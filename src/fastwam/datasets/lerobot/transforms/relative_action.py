from typing import List, Dict

import torch
from .rotation import quaternion_to_matrix, matrix_to_quaternion

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r


class RelativePoseTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._forward(action, state[..., -1:, :])
        
        return batch
        
    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._backward(action, state[..., -1:, :])

        return batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        position = pose[..., 0: 3]
        quaternion = pose[..., [6, 3, 4, 5]] # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0: 3, 0: 3] = rotation
        matrix[..., 0: 3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0: 3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0: 3, 0: 3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]] # (r i j k) to (i j k r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose
    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix
    

class RelativeActionTransform:
    """Convert selected action dims into offsets from an observation state.

    Unlike :class:`RelativePoseTransform`, this works on raw vector dims and
    does not assume quaternion layout, so it can be applied to any LeRobot
    action column (e.g. x, y, z + Euler angles + gripper + wrench).

    Only the dims listed in ``dims`` are converted; all other dims keep their
    absolute value.  ``base_frame`` selects the reference observation:

    - ``"first"`` (default) uses ``state[..., :1, :]``.  The action chunk of a
      training sample starts at the first observation frame (both
      ``observation.state`` and ``action`` use delta timestamps 0..T-1), and
      ``FastWAM`` conditions on ``proprio[:, 0, :]``, so this is the frame the
      deployed policy sees as "current".
    - ``"last"`` uses ``state[..., -1:, :]``, matching
      :class:`RelativePoseTransform`.

    ``dims`` is either a flat list of dim indices applied to every key, or a
    ``{key: dims}`` mapping.  ``state`` may be ``[..., T, D]`` or ``[..., 1, D]``;
    the reference is taken on the time axis and broadcasts over the action
    horizon.
    """

    def __init__(self, keys: List[str], dims, base_frame: str = "first"):
        if base_frame not in ("last", "first"):
            raise ValueError(f"base_frame must be 'last' or 'first', got {base_frame!r}")
        self.keys = list(keys)
        if isinstance(dims, dict):
            self.dims: Dict[str, List[int]] = {k: list(dims.get(k, [])) for k in self.keys}
        else:
            self.dims = {k: list(dims) for k in self.keys}
        self.base_frame = base_frame

    def _reference(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim < 2:
            return state
        return state[..., -1:, :] if self.base_frame == "last" else state[..., :1, :]

    def _apply(self, batch: Dict, sign: float):
        for k in self.keys:
            dims = self.dims[k]
            if not dims or k not in batch["action"]:
                continue
            # Clone: batch tensors may alias dataset storage.
            action = batch["action"][k].clone()
            reference = self._reference(batch["state"][k])
            action[..., dims] = action[..., dims] + sign * reference[..., dims]
            batch["action"][k] = action
        return batch

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        return self._apply(batch, sign=-1.0)

    def backward(self, batch: Dict):
        return self._apply(batch, sign=1.0)


class RelativeJointTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., :1, :]

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., :1, :]
        
        return batch
