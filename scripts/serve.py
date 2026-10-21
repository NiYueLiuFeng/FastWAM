"""Start the FastWAM cloud inference service.

Example:

    python scripts/serve.py \
        task=mydata_uncond_2cam224_1e-4 \
        serving.ckpt=./runs/mydata_uncond_2cam224_1e-4/<run_id>/checkpoints/weights/<step>.pt \
        serving.dataset_stats_path=./runs/mydata_uncond_2cam224_1e-4/<run_id>/dataset_stats.json
"""

import hydra
from omegaconf import DictConfig

from fastwam.serving.service import run_service
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="serve", version_base="1.3")
def main(cfg: DictConfig):
    run_service(cfg)


if __name__ == "__main__":
    main()
