from turtlebot_single import SimEnv
import time
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog
from adversarial_comms.trainers.multiagent_ppo import MultiPPOTrainer
from adversarial_comms.trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution
# Can alternatively pass in p.DIRECT

if __name__ == '__main__':
    register_env("pybullet", lambda config: SimEnv(config))

    ray.init()
    tune.run(
        "PPO",
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
                "fcnet_activation": "relu",
                "fcnet_hiddens": [128, 256, 128, 32],
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
                'max_time_steps': 100,
                'communication_range': 2.0,
                'render': False,
            }})

    #tune.run(
    #    "SAC",
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-02_21-44-56vzulq1wd/checkpoint_190",
        #checkpoint_freq=10,
    #    config=cfg)

'''
d = Drone()
d.set_speed([0,0,0.01])
for i in range(200):
    d.step()
    p.stepSimulation()
    time.sleep(1./240.)

d.set_speed([0,0.01,0.0])

for i in range(1000):
    d.step()
    p.stepSimulation()
    time.sleep(1./240.)

d.set_speed([0,0,-0.005])
for i in range(1000):
    d.step()
    p.stepSimulation()
    time.sleep(1./240.)
#time.sleep(5)
p.disconnect()
'''
