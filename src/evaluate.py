import argparse
import collections.abc
import json
import pandas as pd
import ray
import time
import traceback

from gym import wrappers
from pathlib import Path
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
from ray.util.multiprocessing import Pool

from models.model import Model
from envs.env import PassageEnvRender
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


def run_trial(checkpoint_path_string, trial, cfg_update={}, render_dir=None):
    try:
        checkpoint_path = Path(checkpoint_path_string)
        t0 = time.time()
        with open(checkpoint_path.parent / "params.json") as json_file:
            cfg = json.load(json_file)

        cfg = update_dict(cfg, cfg_update)

        trainer = MultiPPOTrainer(
            env=cfg["env"],
            logger_creator=lambda config: NoopLogger(config, ""),
            config={
                "framework": "torch",
                "seed": trial,
                "num_workers": 0,
                "env_config": cfg["env_config"],
                "model": cfg["model"],
            },
        )
        checkpoint_file = checkpoint_path / (
            "checkpoint-" + str(int(checkpoint_path.name.split("_")[-1]))
        )
        trainer.restore(str(checkpoint_file))

        env = {"passage_env": PassageEnvRender}[cfg["env"]](cfg["env_config"])
        if render_dir is not None:
            env = wrappers.Monitor(env, Path(checkpoint_path) / render_dir, resume=True)
        env.seed(trial)
        obs = env.reset()

        policy = trainer.get_policy()
        gnn_inputs = []

        def record_gnn_input(module, input_, output):
            gnn_inputs.append(output.detach().cpu().numpy())

        # record layer before GNN input
        # policy.model.gnn.gnn.nns[3].register_forward_hook(record_gnn_input)

        results = []
        for i in range(cfg["env_config"]["max_time_steps"]):
            if render_dir is not None:
                env.render()
            actions = trainer.compute_action(obs)
            obs, r, done, info = env.step(actions)
            # assert len(gnn_inputs) == 1
            for j in range(5):
                results.append(
                    {
                        "episode": trial,
                        "timestep": i,
                        "agent": j,
                        "px": env.ps[0, j, 0],
                        "py": env.ps[0, j, 1],
                        "vx": actions[j][0],
                        "vy": actions[j][1],
                        "reward": info["rewards"][j],
                        # "n_covered_targets": info["n_covered_targets"][j],
                        # "gnn_in_features": gnn_inputs[0][j],
                    }
                )
            gnn_inputs = []

            if done:
                break

        print("Done", time.time() - t0)
    except Exception as e:
        print(e, traceback.format_exc())
        raise
    return pd.DataFrame(results)


def serve_config(
    checkpoint_path,
    trials,
    seed=0,
    cfg_change={},
    render_dir=None,
):
    with Pool(32) as p:
        results = p.starmap(
            run_trial,
            [
                (checkpoint_path, seed + t, cfg_change, render_dir)
                for t in range(trials)
            ],
        )
    return pd.concat(results)


def initialize():
    ray.init(include_dashboard=False, object_store_memory=8 * 10 ** 9)
    register_env("passage_env", lambda config: PassageEnvRender(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode_len", type=int, default=None)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--comm_range", type=float, default=None)
    parser.add_argument(
        "--render_dir", type=str, default=None
    )  # subdir in checkpoint for videos
    parser.add_argument(
        "--dataset", type=str, default=None
    )  # filename for dataset in checkpoint dir
    args = parser.parse_args()

    initialize()
    cfg_update = {
        "env_config": {
            "render_px_per_m": 160,
        },
    }
    if args.episode_len is not None:
        cfg_update["env_config"]["max_episode_len"] = args.episode_len

    if args.comm_range is not None:
        cfg_update["model"] = {
            "custom_model_config": {
                "comm_radius": args.comm_range,
            },
        }

    df = serve_config(
        args.checkpoint, args.trials, args.seed, cfg_update, render_dir=args.render_dir
    )

    ep_lens = df.groupby("episode")["timestep"].max()
    ep_rewards = df.groupby("episode")["reward"].agg("sum")
    print("ep len", ep_lens.mean(), "+-", ep_lens.std())
    print(
        "ep rew",
        ep_rewards.mean(),
        "+-",
        ep_rewards.std(),
        ep_rewards.min(),
        ep_rewards.max(),
        ep_rewards.argmax(),
    )

    if args.dataset is not None:
        df.to_pickle(Path(args.checkpoint) / args.dataset)


if __name__ == "__main__":
    serve()
