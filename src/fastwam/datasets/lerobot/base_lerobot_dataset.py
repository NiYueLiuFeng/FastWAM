import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Literal, Dict, Optional, Any, DefaultDict
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from fastwam.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5


def _stats_all_reduce_device() -> torch.device:
    """Device for the stats all_reduce: NCCL needs CUDA tensors, gloo needs CPU ones."""
    if torch.distributed.get_backend() == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
            return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _resolve_lerobot_key(meta: Dict[str, Any], standard_key: str, default_key: str) -> str:
    """Map a shape_meta entry to its LeRobot column.

    Standard LeRobot names are derived from `key`; an explicit `lerobot_key`
    in shape_meta overrides that, which lets datasets name the column
    differently (e.g. `actions` instead of `action`).
    """
    explicit = meta.get("lerobot_key")
    if explicit:
        return str(explicit)
    return standard_key if meta["key"] != "default" else default_key

class BaseLerobotDataset(torch.utils.data.Dataset):
    metadata_cls = LeRobotDatasetMetadata
    multi_dataset_cls = MultiLeRobotDataset
    presample_images = False

    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1, 
        past_action_size: int = 0, # Excludes the current frame
        obs_size: int = 1, # should be 
        past_obs_size: int = 0,

        # train vs val
        val_set_proportion: float = 0.05, 
        is_training_set: bool = False,
        seed: int = 42,

        # sampling
        global_sample_stride: int = 1,
        image_subsample_stride: int = 1,
        tolerance_s: Optional[float] = None,
        video_backend: Optional[str] = None,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"
        
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.image_subsample_stride = image_subsample_stride
        self.processor = None  # Will be set externally
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = self.metadata_cls(repo_id=repo_id, root=ds_root)
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]
        
        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps = {}
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = _resolve_lerobot_key(
                meta, f"observation.images.{key}", "observation.images")
            image_stride = self.image_subsample_stride if self.presample_images else 1
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps
                for t in range(-past_obs_size, -past_obs_size + obs_size, image_stride)
            ]
        
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = _resolve_lerobot_key(
                meta, f"observation.state.{key}", "observation.state")
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
        
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = _resolve_lerobot_key(
                meta, f"action.{key}", "action")
            delta_timestamps[meta["lerobot_key"]] = [(t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)]

        episodes = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes.update({meta.repo_id: list(range(meta.total_episodes))})
        else:
            for meta in metas:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                # random shuffle episode indices before splitting
                episode_indices = list(range(meta.total_episodes))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx)]})
                else:
                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx, meta.total_episodes)]})

        tolerances_s = None if tolerance_s is None else dict.fromkeys(self.dataset_dirs, tolerance_s)
        self.multi_dataset = self.multi_dataset_cls(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerances_s=tolerances_s,
            video_backend=video_backend,
        )
        
        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        # state = state[..., :-1, :]  # use state_{t} as observation_t
        assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state
    
    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)        
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image
    
    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample
    
    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        # Retry with random indices until we successfully load a frame.
        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    f"Error loading sample {sample_idx} (attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {idx}."
            ) from last_exception

        # Get data from lerobot, organized in nested dict
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Preprocess the sample using the processor
        # for quick data loading
        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor, rank: int = 0, world_size: int = 1):
        """Compute normalization statistics over all episodes.

        With `world_size > 1` the episodes are sharded round-robin across ranks and
        the partial aggregates are combined with one all_reduce, so every rank ends
        up with the same stats and wall time drops roughly `world_size` times.
        Min / max / quantiles are combined with MIN / MAX; per-episode mean and
        variance are combined with SUM and turned into mean / std afterwards, which
        is exactly equivalent to the single-process result.
        """
        episodes_num = self.multi_dataset.num_episodes
        if world_size > episodes_num:
            rank, world_size = 0, 1

        # (group, key, stat) -> per-episode tensors, for min / max / q01 / q99
        extremes = DefaultDict(list)
        # (group, key, stat) -> float64 sum over the local episodes, for mean / var / mean**2
        sums = {}

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        def accumulate(group, meta, cur: torch.Tensor):
            key = meta["key"]
            extremes[(group, key, "min")].append(cur.amin(0))
            extremes[(group, key, "max")].append(cur.amax(0))
            extremes[(group, key, "q01")].append(torch.quantile(cur, 0.01, dim=0, keepdim=False))
            extremes[(group, key, "q99")].append(torch.quantile(cur, 0.99, dim=0, keepdim=False))
            # equal weight per episode, matching the single-process reduction
            mean = cur.mean(0).double()
            for name, value in (("mean", mean), ("var", cur.var(0).double()), ("sq", mean * mean)):
                prev = sums.get((group, key, name))
                sums[(group, key, name)] = value.clone() if prev is None else prev + value

        local_indices = list(range(rank, episodes_num, world_size))
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_episode, num) for num in local_indices]
            progress = tqdm(
                as_completed(futures),
                total=len(local_indices),
                desc=f"Iterating dataset to get normalization (rank {rank}/{world_size}: {len(local_indices)} of {episodes_num} episodes)",
                disable=world_size > 1 and rank != 0,
            )
            for future in progress:
                try:
                    batch = future.result()
                except Exception as e:
                    logger.error(f"Error processing episode: {e}")
                    print(traceback.format_exc())
                    raise e
                for meta in self.state_meta:
                    accumulate("state", meta, batch["state"][meta["key"]])
                for meta in self.action_meta:
                    accumulate("action", meta, batch["action"][meta["key"]])

        if world_size > 1:
            self._combine_sharded_stats(extremes, sums)

        # assume that each minibatch has equal number of samples
        def get_mean_std(group, key):
            sum_mean = sums[(group, key, "mean")]
            sum_var = sums[(group, key, "var")]
            sum_sq = sums[(group, key, "sq")]
            n = float(episodes_num)
            stepwise_mean = sum_mean / n
            stepwise_std = (sum_var / n + sum_sq / n - stepwise_mean ** 2).clamp_min(0).sqrt()
            global_mean = sum_mean.mean(0) / n
            global_std = (sum_var.mean(0) / n + sum_sq.mean(0) / n - global_mean ** 2).clamp_min(0).sqrt()
            return (
                stepwise_mean.float(),
                stepwise_std.float(),
                global_mean.float(),
                global_std.float(),
            )

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
        for group, metas in (("state", self.state_meta), ("action", self.action_meta)):
            for meta in metas:
                key = meta["key"]
                entry = stats[group][key]
                entry["stepwise_min"] = torch.stack(extremes[(group, key, "min")]).amin(0)
                entry["stepwise_max"] = torch.stack(extremes[(group, key, "max")]).amax(0)
                entry["global_min"] = entry["stepwise_min"].amin(0)
                entry["global_max"] = entry["stepwise_max"].amax(0)
                entry["stepwise_q01"] = torch.stack(extremes[(group, key, "q01")]).amin(0)
                entry["stepwise_q99"] = torch.stack(extremes[(group, key, "q99")]).amax(0)
                entry["global_q01"] = entry["stepwise_q01"].amin(0)
                entry["global_q99"] = entry["stepwise_q99"].amax(0)
                (
                    entry["stepwise_mean"],
                    entry["stepwise_std"],
                    entry["global_mean"],
                    entry["global_std"],
                ) = get_mean_std(group, key)

        return stats

    @staticmethod
    def _combine_sharded_stats(extremes, sums):
        """All-reduce the per-rank partial aggregates of `get_dataset_stats`."""
        import torch.distributed as dist

        device = _stats_all_reduce_device()
        min_keys = sorted(k for k in extremes if k[2] in ("min", "q01"))
        max_keys = sorted(k for k in extremes if k[2] in ("max", "q99"))
        sum_keys = sorted(sums)
        if not min_keys or not max_keys or not sum_keys:
            raise RuntimeError(
                f"rank {dist.get_rank()} processed no episodes and cannot contribute to dataset stats; "
                f"set FASTWAM_STATS_SHARDING=0 to disable sharding"
            )

        local = {k: torch.stack(extremes[k]).amin(0) for k in min_keys}
        local.update({k: torch.stack(extremes[k]).amax(0) for k in max_keys})
        local.update(sums)

        def pack(keys):
            shapes = [local[k].shape for k in keys]
            flat = torch.cat([local[k].reshape(-1) for k in keys]).to(device=device, dtype=torch.float64)
            return flat, shapes

        def unpack(flat, keys, shapes):
            out, offset = {}, 0
            for key, shape in zip(keys, shapes):
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                out[key] = flat[offset:offset + numel].reshape(shape)
                offset += numel
            return out

        min_flat, min_shapes = pack(min_keys)
        max_flat, max_shapes = pack(max_keys)
        sum_flat, sum_shapes = pack(sum_keys)
        dist.all_reduce(min_flat, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_flat, op=dist.ReduceOp.MAX)
        dist.all_reduce(sum_flat, op=dist.ReduceOp.SUM)

        reduced = {}
        reduced.update({k: v.float() for k, v in unpack(min_flat.cpu(), min_keys, min_shapes).items()})
        reduced.update({k: v.float() for k, v in unpack(max_flat.cpu(), max_keys, max_shapes).items()})
        reduced.update({k: v.double() for k, v in unpack(sum_flat.cpu(), sum_keys, sum_shapes).items()})

        for key in min_keys + max_keys:
            extremes[key] = [reduced[key]]
        for key in sum_keys:
            sums[key] = reduced[key]


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].
    
    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)
    
    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window
    
    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0
    
    N, D = x.shape
    
    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)            # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices                     # [N, window_size]

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out
