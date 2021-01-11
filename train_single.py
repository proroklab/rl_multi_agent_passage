from turtlebot import CentrSimEnv
import time
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger
from ray.tune.integration.wandb import WandbLogger

from ray.rllib.models import ModelCatalog
#from adversarial_comms.trainers.multiagent_ppo import MultiPPOTrainer
#from adversarial_comms.trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution
# Can alternatively pass in p.DIRECT

import random

def train():
    tune.run(
        "PPO",
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        #local_dir="/tmp",
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config={
            "framework": "torch",
            "env": "pybullet_centr", #"pybullet_centr",
            #"lambda": 0.95,
            "kl_coeff": 0.0,
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "train_batch_size": 100000,
            "sgd_minibatch_size": 16384,
            "num_sgd_iter": 20,
            "num_gpus": 0.5,
            "num_workers": 16,
            #"num_gpus_per_worker": 0.5/num_workers,
            "num_envs_per_worker": 1,
            "lr": 1e-4,
            "gamma": 0.995,
            "batch_mode": "complete_episodes", # complete_episodes, truncate_episodes
            "observation_filter": "NoFilter",
            #"normalize_actions": True,
            #"grad_clip": 0.5,
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [128, 256, 128, 32],
            },
            "logger_config": {
                "wandb": {
                    "project": "rl_dynamic_control",
                    "group": "turtlebots",
                    "api_key_file": "./wandb_api_key_file"
                }
            },
            "env_config": {
                'agent_poses': [
                    [-0.3, -0.5, 0],
                    #[0.3, -0.5, 0],
                ],
                'agent_goals': [
                    [0.3, 0.5, 0],
                    #[-0.3, 0.5, 0]
                ],
                'max_time_steps': 3000,
                'communication_range': 2.0,
                'render': False,
            }})


if __name__ == '__main__':
    register_env("pybullet", lambda config: SimEnv(config))
    register_env("pybullet_centr", lambda config: CentrSimEnv(config))

    ray.init(num_cpus=32*4, num_gpus=1)
    train()

