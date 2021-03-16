#from turtlebot import SimEnv
from simple_env import SimpleEnv
import json
import os
import time
import numpy as np
import yaml
import ray
from pathlib import Path
from ray import tune
from ray.tune.registry import register_env

from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger
from ray.tune.integration.wandb import WandbLogger

#from model import Model
from model_2 import Model as Model2
from ray.rllib.models import ModelCatalog

from rllib_multi_agent_demo.multi_trainer import MultiPPOTrainer
from rllib_multi_agent_demo.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)


def initialize():
    #ray.init(local_mode=True)
    ray.init()

    register_env("simple", lambda config: SimpleEnv(config))
    #ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_model("model2", Model2)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)


def continue_experiment(checkpoint_path):
    with open(Path(checkpoint_path) / '..' / 'params.json', "rb") as config_file:
        config = json.load(config_file)

    config['lr'] = 5e-5
    print(config)
    checkpoint_file = Path(checkpoint_path) / ('checkpoint-' + os.path.basename(checkpoint_path).split('_')[-1])

    tune.run(
        MultiPPOTrainer,
        checkpoint_freq=10,
        #keep_checkpoints_num=2,
        #checkpoint_score_attr="episode_reward_mean",
        restore=checkpoint_file,
        config=config,
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        #local_dir="/tmp"
    )


def train():
    num_workers = 16
    tune.run(
        MultiPPOTrainer,
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=10,
        #keep_checkpoints_num=2,
        #checkpoint_score_attr="episode_reward_mean",
        #local_dir="/tmp",
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config={
            "framework": "torch",
            "env": "simple",
            #"lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "train_batch_size": 65536,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 24,
            "num_gpus": 0.5,
            "num_workers": num_workers,
            #"num_gpus_per_worker": 0.5/num_workers,
            "num_envs_per_worker": 1,
            "lr": 5e-5,
            "gamma": 0.995,
            "batch_mode": "complete_episodes", # complete_episodes, truncate_episodes
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "model2",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "graph_tabs": 2,
                    "activation": "relu",
                }
            },
            "logger_config": {
                "wandb": {
                    "project": "rl_dynamic_control",
                    "group": "turtlebots",
                    "api_key_file": "./wandb_api_key_file"
                }
            },
            "env_config": {
                'action_coord_frame': "differential", # global
                'world_shape': (3., 3.),
                'agent_formation': [[-0.2, -0.2], [-0.2, 0.2], [0.2, -0.2], [0.2, 0.2]],
                'max_time_steps': 500,
                'communication_range': 2.0,
                'render': False,
                'max_lateral_speed': 1.0,
                'max_angular_speed': 5*np.pi,
            }})


if __name__ == '__main__':
    initialize()
    train()
    #continue_experiment("../ray_results/MultiPPO_2021-02-01_21-19-50/MultiPPO_simple_383fa_00000_0_2021-02-01_21-19-51/checkpoint_138")

