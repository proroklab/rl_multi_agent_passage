import argparse
import collections.abc
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import ray
import time
import torch
import traceback
import tree

from pathlib import Path
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
from ray.util.multiprocessing import Pool

from turtlebot import SimEnv, CentrSimEnv
from simple_env import SimpleEnv
from model import Model
from model_2 import Model as Model2
from rllib_multi_agent_demo.multi_trainer import MultiPPOTrainer
from rllib_multi_agent_demo.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def add_batch_dim(data):
    def mapping(item):
        if isinstance(item, np.ndarray):
            return torch.from_numpy(np.expand_dims(item, 0).astype(np.float32))
        else:
            return item

    return tree.map_structure(mapping, data)


def run_trial(
    trainer_class=MultiPPOTrainer,
    checkpoint_path=None,
    trial=0,
    cfg_update={},
    render=False,
    sampling_probability=0.0,
):
    try:
        t0 = time.time()
        cfg = {"env_config": {}, "model": {}}
        if checkpoint_path is not None:
            # We might want to run policies that are not loaded from a checkpoint
            # (e.g. the random policy) and therefore need this to be optional
            with open(Path(checkpoint_path).parent / "params.json") as json_file:
                cfg = json.load(json_file)

        if "evaluation_config" in cfg:
            # overwrite the environment config with evaluation one if it exists
            cfg = update_dict(cfg, cfg["evaluation_config"])

        cfg = update_dict(cfg, cfg_update)

        # cfg['env_config']["render"] = True
        # cfg['env_config']['agent_formation'] = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])*0.4
        # cfg['env_config']['placement_keepout_wall'] = 1.0
        # cfg['env_config']['placement_keepout_border'] = 0.5
        # cfg['env_config']['gap_length'] = 1.0
        # cfg['env_config']['grid_px_per_m'] = 40
        # cfg['env_config']['dt'] = 0.05
        # cfg['env_config']['world_shape'] = (4,6)

        trainer = trainer_class(
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
        if checkpoint_path is not None:
            checkpoint_file = Path(checkpoint_path) / (
                "checkpoint-" + os.path.basename(checkpoint_path).split("_")[-1]
            )
            trainer.restore(str(checkpoint_file))

        env = {"pybullet_centr": CentrSimEnv, "pybullet": SimEnv, "simple": SimpleEnv}[
            cfg["env"]
        ](cfg["env_config"])
        env.seed(trial)
        obs = env.reset()

        samples = []
        policy = trainer.get_policy()

        """
        cnn_outputs = []
        def record_cnn_output(module, input_, output):
            cnn_outputs.append(output[0].detach().cpu().numpy())
        gnn_outputs = []
        def record_gnn_output(module, input_, output):
            gnn_outputs.append(output[0].detach().cpu().numpy())
        policy.model.coop_convs[-1].register_forward_hook(record_cnn_output)
        policy.model.greedy_convs[-1].register_forward_hook(record_cnn_output)
        policy.model.GFL.register_forward_hook(record_gnn_output)
        """

        results = []
        all_rewards = []
        all_times = []
        i = 0
        rewards = 0
        while True:  # for i in range(cfg['env_config']['max_time_steps']):
            # if render:
            # env.render()
            # time.sleep(1.0)

            # logits, _ = policy.model.forward({"obs": add_batch_dim(obs)}, None, None)
            # dist = policy.dist_class(logits, policy.model)
            # actions = [a.tolist()[0] for a in dist.sample()]
            # print(actions)
            env.render()
            time.sleep(0.01)
            actions = trainer.compute_action(obs)
            """
            n_agents = sum(cfg['env_config']['n_agents'])
            for j in range(n_agents):
                obs['agents'][j]['msg'] = cnn_outputs[j]
                obs['agents'][j]['gnn_features'] = gnn_outputs[0][..., j]
                obs['agents'][j]['logits'] = logits.view(-1, n_agents, 5)[0, j].detach().cpu().numpy()
            cnn_outputs = []
            gnn_outputs = []
            """

            if sampling_probability == 1.0 or np.random.rand() < sampling_probability:
                samples.append(copy.deepcopy({"obs": obs, "actions": actions}))

            obs, r, done, info = env.step(actions)
            print(rewards)
            rewards += r
            # print(info['rewards'].values(), actions)
            # print(env.robots[0].position)
            if done:
                all_rewards.append(rewards)
                all_times.append(env.timestep)
                print(np.mean(all_rewards), rewards, np.mean(all_times), env.timestep)
                env.reset()
                rewards = 0
            """
            for j, reward in enumerate(list(info['rewards'].values())):
                results.append({
                    'step': i,
                    'agent': j,
                    'trial': trial,
                    'reward': reward
                })
            """
            i += 1

        print("Done", time.time() - t0)
    except Exception as e:
        print(e, traceback.format_exc())
        raise
    return pd.DataFrame(results), samples


def path_to_hash(path):
    path_split = path.split("/")
    checkpoint_number_string = path_split[-1].split("_")[-1]
    path_hash = path_split[-2].split("_")[-2]
    return path_hash + "-" + checkpoint_number_string


def serve_config(
    checkpoint_path,
    trials,
    seed=0,
    cfg_change={},
    trainer=MultiPPOTrainer,
    render=False,
    sampling_probability=0.0,
):
    with Pool() as p:
        results, samples = zip(
            *p.starmap(
                run_trial,
                [
                    (
                        trainer,
                        checkpoint_path,
                        seed + t,
                        cfg_change,
                        render,
                        sampling_probability,
                    )
                    for t in range(trials)
                ],
            )
        )
    return pd.concat(results), samples


def initialize():
    ray.init()
    register_env("pybullet", lambda config: SimEnv(config))
    register_env("pybullet_centr", lambda config: CentrSimEnv(config))
    register_env("simple", lambda config: SimpleEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_model("model2", Model2)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-l", "--episode_len", type=int, default=None)
    parser.add_argument("-n", "--num-agents", type=int, default=5)
    args = parser.parse_args()

    initialize()
    cfg_change = {
        "env_config": {
            "render": False,
        }
    }
    if args.episode_len is not None:
        cfg_change["env_config"]["max_episode_len"] = args.episode_len
    for i in range(1):
        df, _ = run_trial(
            trainer_class=MultiPPOTrainer,
            checkpoint_path=args.checkpoint,
            trial=args.seed + i,
            render=True,
            sampling_probability=1.0,
            cfg_update=cfg_change,
        )
        d = (
            df.sort_values(["trial", "step"])
            .groupby(["trial", "step"])["reward"]
            .apply("sum", "step")
            .groupby("trial")
            .cumsum()
            .groupby("step")
        )
        print(d.mean().tail(1))
    plt.ioff()
    plt.show()


def sample():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_file")
    parser.add_argument("-t", "--trials", type=int, default=100)
    parser.add_argument("-l", "--episode_len", type=int, default=None)
    parser.add_argument("-p", "--sample_probability", type=float, default=1.0)
    parser.add_argument("-n", "--num-agents", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    initialize()
    cfg_change = {
        "env_config": {
            "render": False,
        }
    }
    if args.episode_len is not None:
        cfg_change["env_config"]["max_episode_len"] = args.episode_len
    _, samples = serve_config(
        args.checkpoint,
        args.trials,
        seed=args.seed,
        cfg_change=cfg_change,
        sampling_probability=args.sample_probability,
    )

    with open(args.out_file, "wb") as f:
        pickle.dump(samples, f)


serve()
