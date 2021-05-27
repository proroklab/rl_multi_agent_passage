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
from model_2 import Model as Model2
from ray.rllib.models import ModelCatalog
from adversarial_comms.trainers.multiagent_ppo import MultiPPOTrainer
from adversarial_comms.trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution

if __name__ == '__main__':
    #ray.init(local_mode=True)
    ray.init()

    register_env("pybullet", lambda config: SimEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_model("model2", Model2)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)

    num_workers = 16
    tune.run(
        MultiPPOTrainer,
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        local_dir="/tmp",
        #loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config={
            "framework": "torch",
            "env": "pybullet",
            #"lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "train_batch_size": 100000,
            "sgd_minibatch_size": 32768,
            "num_sgd_iter": 20,
            "num_gpus": 0.5,
            "num_workers": 16,
            #"num_gpus_per_worker": 0.5/num_workers,
            "num_envs_per_worker": 1,
            "lr": 1e-4,
            "gamma": 0.995,
            "batch_mode": "complete_episodes", # complete_episodes, truncate_episodes
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "model2",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "graph_tabs": 2,
                    "graph_edge_features": 1,

                    "graph_features": 64,

                    "graph_aggregation": "sum",

                    "activation": "relu",
                    "agent_split": 0,
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
                'wall': True,
                'agent_poses': [
                    [0.0, -0.5, 0],
                    [-0.3, -0.5, 0],
                    [0.3, -0.5, 0],
                    #[0.0, -0.8, 0],
                    #[-0.3, -0.8, 0],
                    #[0.3, -0.8, 0],
                    #[0.0, -1.1, 0],
                    #[-0.3, -1.1, 0],
                    #[0.3, -1.1, 0],
                ],
                'agent_goals': [
                    #[0.0, 1.1, 0],
                    #[-0.3, 1.1, 0],
                    #[0.3, 1.1, 0],
                    #[0.0, 0.8, 0],
                    [0.3, 0.5, 0],
                    [-0.3, 0.5, 0],
                    #[0.3, 0.8, 0],
                    #[-0.3, 0.8, 0],
                    [0.0, 0.5, 0],
                ],
                'max_time_steps': 10000,
                'communication_range': 2.0,
                'render': False,
            }})

