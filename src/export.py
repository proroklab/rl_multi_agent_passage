import argparse
import json
import ray

from pathlib import Path
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env

from models.model import Model
from envs.env import PassageEnvRender
from rllib_multi_agent_demo.multi_trainer import MultiPPOTrainer
from rllib_multi_agent_demo.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)


def initialize():
    ray.init(include_dashboard=False, object_store_memory=8 * 10 ** 9)
    register_env("passage_env", lambda config: PassageEnvRender(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )


def export():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    initialize()

    checkpoint_path = Path(args.checkpoint)
    with open(checkpoint_path.parent / "params.json") as json_file:
        cfg = json.load(json_file)

    trainer = MultiPPOTrainer(
        env=cfg["env"],
        logger_creator=lambda config: NoopLogger(config, ""),
        config={
            "framework": "torch",
            "seed": 0,
            "num_workers": 0,
            "env_config": cfg["env_config"],
            "model": cfg["model"],
        },
    )
    checkpoint_file = checkpoint_path / (
        "checkpoint-" + str(int(checkpoint_path.name.split("_")[-1]))
    )
    trainer.restore(str(checkpoint_file))
    trainer.export_model("model", str(checkpoint_path.resolve()))


if __name__ == "__main__":
    export()
