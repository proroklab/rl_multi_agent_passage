import time
import gym
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.typing import (
    EnvActionType,
    EnvConfigDict,
    EnvInfoDict,
    EnvObsType,
    EnvType,
    PartialTrainerConfigDict,
)
from typing import Callable, List, Optional, Tuple

from gym.utils import seeding
import torch
import math
import pygame

from scipy.spatial.transform import Rotation as R

X = 0
Y = 1

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)


class PassageEnv(VectorEnv):
    def __init__(self, config):
        self.cfg = config
        action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,), dtype=float
                ),
            )
            * self.cfg["n_agents"]
        )

        observation_space = gym.spaces.Dict(
            {
                "pos": gym.spaces.Box(
                    -6.0, 6.0, shape=(self.cfg["n_agents"], 2), dtype=float
                ),
                "vel": gym.spaces.Box(
                    -100000.0, 100000.0, shape=(self.cfg["n_agents"], 2), dtype=float
                ),
                "goal": gym.spaces.Box(
                    -6.0, 6.0, shape=(self.cfg["n_agents"], 2), dtype=float
                ),
            }
        )

        super().__init__(observation_space, action_space, self.cfg["num_envs"])

        self.device = torch.device(self.cfg["device"])
        self.vec_p_shape = (self.cfg["num_envs"], self.cfg["n_agents"], 2)

        self.ps = self.create_state_tensor()
        self.goal_ps = self.create_state_tensor()
        self.measured_vs = self.create_state_tensor()

        pygame.init()
        size = (
            (torch.Tensor(self.cfg["world_dim"]) * self.cfg["render_px_per_m"])
            .type(torch.int)
            .tolist()
        )
        self.display = pygame.display.set_mode(size)

    def create_state_tensor(self):
        return torch.zeros(self.vec_p_shape, dtype=torch.float32).to(self.device)

    def sample_pos_noise(self):
        if self.cfg["pos_noise_std"] > 0.0:
            return torch.normal(0.0, self.cfg["pos_noise_std"], self.vec_p_shape).to(
                self.device
            )
        else:
            return self.create_state_tensor()

    def check_collisions(self, ps):
        d_agents = torch.cdist(ps, ps)
        diags = torch.eye(ps.shape[1]).unsqueeze(0).repeat(len(d_agents), 1, 1).bool()
        d_agents[diags] = float("inf")
        min_d = torch.min(d_agents, dim=2)[0]
        return min_d > 2 * self.cfg["agent_radius"]

    def rand(self, size, a: float, b: float):
        return (a - b) * torch.rand(size).to(self.device) + b

    def get_starts_and_goals(self, n):
        def generate_rotated_formation():
            rot = torch.empty(n, 2, 2).to(self.device)
            theta = self.rand(n, -math.pi, math.pi)
            c, s = torch.cos(theta), torch.sin(theta)
            rot[:, 0, 0] = c
            rot[:, 0, 1] = -s
            rot[:, 1, 0] = s
            rot[:, 1, 1] = c
            formation = torch.Tensor(self.cfg["agent_formation"]).to(self.device)
            return torch.bmm(formation.repeat(n, 1, 1), rot)

        def rand_n_agents(a, b):
            return self.rand(n, a, b).unsqueeze(1).repeat(1, self.cfg["n_agents"])

        box = (
            torch.Tensor(self.cfg["world_dim"]) / 2
            - self.cfg["placement_keepout_border"]
        )
        starts = generate_rotated_formation()

        starts[:, :, X] += rand_n_agents(-box[X], box[X])
        starts[:, :, Y] += rand_n_agents(-box[Y], -self.cfg["placement_keepout_wall"])
        goals = generate_rotated_formation()
        goals[:, :, X] += rand_n_agents(-box[X], box[X])
        goals[:, :, Y] += rand_n_agents(self.cfg["placement_keepout_wall"], box[Y])
        return starts, goals

    def vector_reset(self) -> List[EnvObsType]:
        """Resets all sub-environments.
        Returns:
            obs (List[any]): List of observations from each environment.
        """
        starts, goals = self.get_starts_and_goals(self.cfg["num_envs"])
        self.ps = starts
        self.goal_ps = goals
        self.measured_vs = self.create_state_tensor()
        return [self.get_obs(index) for index in range(self.cfg["num_envs"])]

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        """Resets a single environment.
        Args:
            index (Optional[int]): An optional sub-env index to reset.
        Returns:
            obs (obj): Observations from the reset sub environment.
        """
        start, goal = self.get_starts_and_goals(1)
        self.ps[index] = start[0]
        self.goal_ps[index] = goal[0]
        self.measured_vs[index] = torch.zeros(self.cfg["n_agents"], 2)
        return self.get_obs(index)

    def get_obs(self, index: int) -> EnvObsType:
        return {
            "pos": self.ps[index].tolist(),
            "vel": self.measured_vs[index].tolist(),
            "goal": self.goal_ps[index].tolist(),
        }

    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        """Performs a vectorized step on all sub environments using `actions`.
        Args:
            actions (List[any]): List of actions (one for each sub-env).
        Returns:
            obs (List[any]): New observations for each sub-env.
            rewards (List[any]): Reward values for each sub-env.
            dones (List[any]): Done values for each sub-env.
            infos (List[any]): Info values for each sub-env.
        """
        assert len(actions) == self.cfg["num_envs"]
        desired_vs = torch.clip(
            torch.Tensor(actions).to(self.device), -self.cfg["max_v"], self.cfg["max_v"]
        )

        desired_as = (desired_vs - self.measured_vs) / self.cfg["dt"]
        possible_as = torch.clip(desired_as, -self.cfg["max_a"], self.cfg["max_a"])
        possible_vs = self.measured_vs + possible_as * self.cfg["dt"]

        previous_ps = self.ps.clone().to(self.device)
        next_ps = self.ps + possible_vs * self.cfg["dt"]
        no_coll = self.check_collisions(next_ps)
        self.ps[no_coll] = next_ps[no_coll]
        self.ps += self.sample_pos_noise()
        dim = torch.Tensor(self.cfg["world_dim"]) / 2
        self.ps[:, :, X] = torch.clip(self.ps[:, :, X], -dim[X], dim[X])
        self.ps[:, :, Y] = torch.clip(self.ps[:, :, Y], -dim[Y], dim[Y])

        self.measured_vs = (self.ps - previous_ps) / self.cfg["dt"]

        obs = [self.get_obs(index) for index in range(self.cfg["num_envs"])]
        rewards = [0.0 for index in range(self.cfg["num_envs"])]
        dones = [False for index in range(self.cfg["num_envs"])]
        infos = [{} for index in range(self.cfg["num_envs"])]
        return obs, rewards, dones, infos

    def get_unwrapped(self) -> List[EnvType]:
        return []


class PassageGymEnv(PassageEnv, gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, config):
        super().__init__(config)

    def reset(self):
        return self.reset_at(0)

    def step(self, actions):
        vector_actions = self.create_state_tensor()
        vector_actions[0] = actions
        self.vector_step(vector_actions)

    def render(self, mode="rgb_array"):
        AGENT_COLOR = BLUE
        BACKGROUND_COLOR = WHITE
        WALL_COLOR = GRAY

        index = 0

        world_dim_2 = torch.Tensor(self.cfg["world_dim"]) / 2

        def point_to_screen(p):
            return (
                (
                    (torch.Tensor([-p[X], p[Y]]) + world_dim_2)
                    * self.cfg["render_px_per_m"]
                )
                .type(torch.int)
                .tolist()
            )

        self.display.fill(BACKGROUND_COLOR)
        img = pygame.Surface(self.display.get_size(), pygame.SRCALPHA)

        for agent_index in range(self.cfg["n_agents"]):
            pygame.draw.circle(
                img,
                AGENT_COLOR,
                point_to_screen(self.ps[index, agent_index]),
                self.cfg["agent_radius"] * self.cfg["render_px_per_m"],
            )
            pygame.draw.line(
                img,
                AGENT_COLOR,
                point_to_screen(self.ps[index, agent_index]),
                point_to_screen(self.goal_ps[index, agent_index]),
                4,
            )

        left_wall_tl = point_to_screen([world_dim_2[X], -(self.cfg["wall_width"] / 2)])
        left_wall_br = point_to_screen(
            [self.cfg["gap_length"] / 2, self.cfg["wall_width"] / 2]
        )
        left_wall_size = (
            torch.Tensor(left_wall_br) - torch.Tensor(left_wall_tl)
        ).tolist()
        pygame.draw.rect(img, WALL_COLOR, left_wall_tl + left_wall_size)
        right_wall_tl = point_to_screen(
            [-(self.cfg["gap_length"] / 2), -self.cfg["wall_width"] / 2]
        )
        right_wall_br = point_to_screen([-world_dim_2[X], self.cfg["wall_width"] / 2])
        right_wall_size = (
            torch.Tensor(right_wall_br) - torch.Tensor(right_wall_tl)
        ).tolist()
        pygame.draw.rect(img, WALL_COLOR, right_wall_tl + right_wall_size)
        self.display.blit(img, (0, 0))

        if mode == "human":
            pygame.display.update()
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(self.display).type(torch.uint8)

    def try_render_at(self, index: Optional[int] = None) -> None:
        """Renders a single environment.
        Args:
            index (Optional[int]): An optional sub-env index to render.
        """
        return self.render(mode="human")


if __name__ == "__main__":
    env = PassageGymEnv(
        {
            "world_dim": (4.0, 6.0),
            "dt": 0.05,
            "num_envs": 3,
            "device": "cpu",
            "n_agents": 5,
            "agent_formation": (
                torch.Tensor([[-1, -1], [-1, 1], [0, 0], [1, -1], [1, 1]]) * 0.6
            ).tolist(),
            "placement_keepout_border": 1.0,
            "placement_keepout_wall": 1.5,
            "pos_noise_std": 0.0,
            "max_time_steps": 500,
            "communication_range": 2.0,
            "wall_width": 0.3,
            "gap_length": 1.0,
            "grid_px_per_m": 40,
            "agent_radius": 0.25,
            "render": False,
            "render_px_per_m": 160,
            "max_v": 10.0,
            "max_a": 5.0,
        }
    )
    import time

    torch.manual_seed(0)
    env.vector_reset()
    # env.reset()
    selected_agent = 0
    while True:

        a = torch.zeros((env.cfg["n_agents"], 2))
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                env.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                selected_agent += 1
                if selected_agent >= env.cfg["n_agents"]:
                    selected_agent = 0
            elif event.type == pygame.MOUSEMOTION:
                v = (
                    torch.clip(torch.Tensor([-event.rel[0], event.rel[1]]), -20, 20)
                    / 20
                )
                a[selected_agent] = v

        # env.ps[0, 0, X] = 1.0
        env.render(mode="human")

        env.step(a)

        # env.reset()
        # time.sleep(1)
        """
        obs, r, done, info = env.step(a)
        for key, agent_reward in info["rewards"].items():
            returns[key] += agent_reward
        print(returns)
        if done:
            env.reset()
            returns = torch.zeros((env.cfg["n_agents"]))
        """
