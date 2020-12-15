from turtlebot import SimEnv
import time
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env

from model import Model
from ray.rllib.models import ModelCatalog
from adversarial_comms.trainers.multiagent_ppo import MultiPPOTrainer
from adversarial_comms.trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution

if __name__ == '__main__':
    register_env("pybullet", lambda config: SimEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)

    ray.init()
    tune.run(
        MultiPPOTrainer,
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=10,
        config={
            "framework": "torch",
            "_use_trajectory_view_api": False,
            "env": "pybullet",
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "train_batch_size": 1000,
            "rollout_fragment_length": 100,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "num_gpus": 0,
            "model": {
                "custom_model": "model",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "graph_layers": 1,
                    "graph_tabs": 2,
                    "graph_edge_features": 1,

                    "graph_features": 128,

                    "graph_aggregation": "sum",

                    "activation": "relu",
                    "agent_split": 1,
                }
            },
            "env_config": {
                'agent_poses': [
                    [-0.3, -0.5, 0],
                    [0.3, -0.5, 0],
                ],
                'agent_goals': [
                    [0.3, 0.5, 0],
                    [-0.3, 0.5, 0]
                ],
                'max_time_steps': 4000,
                'communication_range': 2.0,
                'render': False,
            }})

