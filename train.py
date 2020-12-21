from turtlebot import SimEnv
import time
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env

from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger
from ray.tune.integration.wandb import WandbLogger

from model import Model
from ray.rllib.models import ModelCatalog
from adversarial_comms.trainers.multiagent_ppo import MultiPPOTrainer
from adversarial_comms.trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution

if __name__ == '__main__':
    ray.init()

    register_env("pybullet", lambda config: SimEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)

    tune.run(
        MultiPPOTrainer,
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        #local_dir="/tmp",
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config={
            "framework": "torch",
            "_use_trajectory_view_api": False,
            "env": "pybullet",
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.2,
            #"entropy_coeff": 0.01,
            "train_batch_size": 80000,
            "sgd_minibatch_size": 16384,
            "num_sgd_iter": 32,
            "num_workers": 4,
            "num_envs_per_worker": 32,
            "lr": 3e-4,
            "gamma": 0.99,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "num_gpus": 1,
            "model": model,
            "logger_config": {
                "wandb": {
                    "project": "rl_dynamic_control",
                    "group": "turtlebots",
                    "api_key_file": "./wandb_api_key_file"
                }
            },
            "env_config": {
                'wall': True,
                'agent_poses': [
                    [-0.3, -0.5, 0],
                    [0.3, -0.5, 0],
                ],
                'agent_goals': [
                    [0.3, 0.5, 0],
                    [-0.3, 0.5, 0]
                ],
                'max_time_steps': 3000,
                'communication_range': 2.0,
                'render': False,
            }})

