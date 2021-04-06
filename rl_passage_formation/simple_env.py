import time
import gym
from gym.utils import seeding
import numpy as np
import pygame

from scipy.spatial.transform import Rotation as R


X = 1
Y = 0


class WorldMap:
    def __init__(self, dim, n_agents, wall_width, gap_length, grid_px_per_m, agent_radius):
        self.dim = np.array(dim)
        self.gap_length = gap_length
        self.wall_width = wall_width
        self.px_per_m = grid_px_per_m
        self.map_grid_shape = (self.dim * self.px_per_m).astype(np.int)
        self.agent_radius = agent_radius
        self.n_agents = n_agents

        yy, xx = np.mgrid[: self.map_grid_shape[Y], : self.map_grid_shape[X]]
        self.yy = (yy / self.map_grid_shape[Y]) * self.dim[Y] - (self.dim / 2)[Y]
        self.xx = (xx / self.map_grid_shape[X]) * self.dim[X] - (self.dim / 2)[X]

        self.reset()

    def reset(self):
        self.map = np.zeros(
            (self.map_grid_shape[Y], self.map_grid_shape[X], 1 + self.n_agents),
            dtype=np.bool,
        )
        gap_start = self.pos_to_grid(np.array([-self.gap_length / 2, 0]))[Y]
        gap_end = self.pos_to_grid(np.array([self.gap_length / 2, 0]))[Y]
        center = self.pos_to_grid(np.array([0, 0]))[X]
        wall_width_px_half = max(1, int((self.wall_width * self.px_per_m) / 2))
        self.map[:gap_start, center - wall_width_px_half : center + wall_width_px_half, 0] = True
        self.map[gap_end:, center - wall_width_px_half : center + wall_width_px_half, 0] = True

    def pos_to_grid(self, p):
        return ((np.array(p) + self.dim / 2) / self.dim * self.map_grid_shape).astype(
            np.int
        )

    def set_robot(self, position, agent_idx):
        rob_map = np.zeros(self.map_grid_shape, dtype=np.bool)
        sel = (
            (self.yy - position[Y]) ** 2 + (self.xx - position[X]) ** 2
        ) < self.agent_radius ** 2
        rob_map[sel] = True

        if self.is_colliding_wall(rob_map):
            return "wall"

        if self.is_colliding_other_agent(agent_idx, rob_map):
            return "agent"

        self.map[:, :, agent_idx + 1] = rob_map
        return "ok"

    def is_colliding_wall(self, m):
        return np.any(m & self.map[:, :, 0])

    def is_colliding_other_agent(self, agent_idx, m):
        for other_agent_idx in range(self.n_agents):
            if other_agent_idx == agent_idx:
                continue
            if np.any(m & self.map[:, :, other_agent_idx + 1]):
                return True
        return False

    def render(self):
        m_acc = np.zeros(self.map_grid_shape, dtype=np.bool)
        for i in range(self.map.shape[2]):
            m_acc = m_acc | self.map[:, :, i]
        return ~m_acc


class Turtlebot:
    def __init__(self, index, dt, max_lateral_speed, world_map):
        self.index = index
        self.dt = dt
        self.world_map = world_map
        self.max_lateral_speed = max_lateral_speed

        self.reset(np.array([0, 0]), np.array([0, 0]))

    def reset(self, start_pos, goal_pos):
        self.position = start_pos.copy()
        self.goal_pos = goal_pos.copy()

        self.setpoint_vx = 0
        self.setpoint_vy = 0
        self.passage_state = "before" # before, in, after, reached_goal

    def set_velocity(self, velocity):
        assert not np.any(np.isnan(velocity))
        self.setpoint_vx = np.clip(
            velocity[1], -self.max_lateral_speed, self.max_lateral_speed
        )
        self.setpoint_vy = np.clip(
            velocity[0], -self.max_lateral_speed, self.max_lateral_speed
        )

    def step(self):
        prev_pos = self.position.copy()
        new_pos = (
            self.position + np.array([self.setpoint_vx, self.setpoint_vy]) * self.dt
        )

        pos_map_status = self.world_map.set_robot(new_pos, self.index)
        if pos_map_status == "ok":
            self.position = np.clip(
                new_pos, -self.world_map.dim / 2, self.world_map.dim / 2
            )

        self.v_world = (self.position.copy() - prev_pos) / self.dt

        features = [self.position, self.goal_pos - self.position]

        return np.hstack(features), pos_map_status


class SimpleEnv(gym.Env):
    def __init__(self, config):
        self.seed(0)

        self.cfg = config
        n_agents = self.cfg["n_agents"]
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=float),)
            * n_agents
        )  # velocity world coordinates

        self.observation_space = gym.spaces.Dict(
            {
                # current pose relative to goal (x,y)
                # current pose relative to passage (x, y)
                "agents": gym.spaces.Tuple(
                    (
                        gym.spaces.Dict(
                            {
                                "obs": gym.spaces.Box(
                                    -10000, 10000, shape=(4,), dtype=float
                                ),
                            }
                        ),
                    )
                    * n_agents
                ),
                "gso": gym.spaces.Box(-1, 1, shape=(n_agents, n_agents), dtype=float),
            }
        )

        self.map = WorldMap(
            self.cfg["world_shape"],
            self.cfg["n_agents"],
            self.cfg["wall_width"],
            self.cfg["gap_length"],
            self.cfg["grid_px_per_m"],
            self.cfg["agent_radius"],
        )

        self.robots = []
        for i in range(self.cfg["n_agents"]):
            self.robots.append(
                Turtlebot(i, self.cfg["dt"], self.cfg["max_lateral_speed"], self.map)
            )

        self.display = None
        self.render_frame_index = 0

        self.reset()

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.timestep = 0

        def generate_rotated_formation(theta):
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            return np.dot(self.cfg["agent_formation"], R)

        #theta_start = self.random_state.uniform(-np.pi, np.pi)
        #rotated_formation_start = generate_rotated_formation(theta_start)
        #theta_end = self.random_state.uniform(-np.pi, np.pi)
        #rotated_formation_end = generate_rotated_formation(theta_end)

        keepout_wall = self.cfg["wall_width"] / 2 + self.cfg["agent_radius"] + 0.1
        box = self.map.dim / 2
        def generate_start():
            return self.random_state.uniform(
                [-box[Y], -box[X]],
                [box[Y], -keepout_wall],
            )

        def generate_goal():
            return self.random_state.uniform(
                [-box[Y], keepout_wall],
                [box[Y], box[X]],
            )

        self.map.reset()

        def place_agent(agent_id, positions, gen_fn):
            if agent_id == len(self.robots):
                return True
            for i in range(10):
                p = gen_fn()
                if self.map.set_robot(p, agent_id) == "ok" and place_agent(agent_id + 1, positions, gen_fn):
                    positions.append(p)
                    return True
            return False

        goals = []
        place_agent(0, goals, generate_goal)

        self.map.reset()
        starts = []
        place_agent(0, starts, generate_start)

        for robot, start, goal in zip(self.robots, starts, goals):
            robot.reset(start, goal)

        self.map.reset()
        return self.step([[0, 0]] * len(self.robots))[0]

    def compute_gso(self):
        dists = np.zeros((len(self.robots), len(self.robots)))
        for agent_y in range(len(self.robots)):
            for agent_x in range(len(self.robots)):
                dst = np.sum(
                    np.array(
                        self.robots[agent_x].position - self.robots[agent_y].position
                    )
                    ** 2
                )
                dists[agent_y, agent_x] = dst
                dists[agent_x, agent_y] = dst

        A = dists < (self.cfg["communication_range"] ** 2)
        np.fill_diagonal(A, 0)
        return A.astype(np.float)

    def step(self, actions):
        self.timestep += 1
        obs, infos = [], {"rewards": {}}
        reward = 0

        world_done = self.timestep > self.cfg["max_time_steps"]
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            robot.set_velocity(action)

            o, pos_map_status = robot.step()

            wall_robot_offset = self.cfg["wall_width"] / 2 + self.cfg["agent_radius"]
            if robot.passage_state == "before":
                goal_vector = np.array([0.0, -wall_robot_offset]) - robot.position
                if np.linalg.norm(goal_vector) < 0.05:
                    robot.passage_state = "in"
            elif robot.passage_state == "in":
                goal_vector = np.array([0.0, wall_robot_offset]) - robot.position
                if np.linalg.norm(goal_vector) < 0.05:
                    robot.passage_state = "after"
            elif robot.passage_state == "after":
                goal_vector = robot.goal_pos - robot.position
                if np.linalg.norm(goal_vector) < 0.05:
                    robot.passage_state = "reached_goal"

            world_speed = robot.v_world
            r = 0
            vw = np.linalg.norm(world_speed)
            if vw > 0 and not robot.passage_state == "reached_goal":
                r = (
                    np.dot(goal_vector / np.linalg.norm(goal_vector), world_speed / vw)
                    * vw
                )

            if pos_map_status == "agent" or pos_map_status == "wall":
                r -= 1

            obs.append({"obs": o})
            infos["rewards"][i] = r
            reward += r

        obs = {
            "agents": tuple(obs),
            "gso": self.compute_gso(),
        }

        world_done = world_done or all([robot.passage_state == "reached_goal" for robot in self.robots])
        return obs, reward, world_done, infos

    def clear_patches(self, ax):
        [p.remove() for p in reversed(ax.patches)]
        [t.remove() for t in reversed(ax.texts)]

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode(self.map.map_grid_shape)
        surf = pygame.surfarray.make_surface(self.map.render().astype(np.uint8) * 255)
        self.display.blit(surf, (0, 0))

        for robot in self.robots:
            pygame.draw.line(
                self.display,
                (0, 0, 255),
                self.map.pos_to_grid(robot.position),
                self.map.pos_to_grid(robot.goal_pos),
                2,
            )

        """
        for y in np.arange(-2, 2, 0.4):
            for x in np.arange(-3, 3, 0.4):
                _, goal_vector = get_velocity(np.array([y, x]), self.robots[0].goal_pos)
                pygame.draw.line(self.display, (0,0,255), self.map.pos_to_grid(np.array([y, x])), self.map.pos_to_grid(np.array([y, x]) + goal_vector/5), 2)
        """
        # for p in CYLINDER_POSITIONS:
        #    pygame.draw.circle(self.display, (0,255,0), p/self.map.dim*[200,200], 5)
        if True:
            self.render_frame_index += 1
            pygame.image.save(self.display, f"./img/{self.render_frame_index}.png")
        pygame.display.update()


if __name__ == "__main__":
    env = SimpleEnv(
        {
            "world_shape": (4.0, 6.0),
            "wall_width": 0.5,
            "dt": 0.05,
            #'agent_formation': [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
            #"agent_formation": [[-0.5, -0.5], [-0.5, 0.5], [0.4, 0.0]],
            "n_agents": 1,
            "max_time_steps": 500,
            "communication_range": 2.0,
            "gap_length": 1.0,
            "grid_px_per_m": 40,
            "agent_radius": 0.3,
            "render": False,
            "max_lateral_speed": 2.0,
        }
    )
    import time

    env.reset()
    ret = 0
    while True:
        env.render()
        a = np.ones((env.cfg["n_agents"], 2)) * 0.2
        # env.step(a)
        #time.sleep(1)
        #env.reset()
        #continue

        a = np.ones((env.cfg["n_agents"], 2))

        wall_robot_offset = env.cfg["wall_width"] / 2 + env.cfg["agent_radius"]
        for i, robot in enumerate(env.robots):
            if robot.passage_state == "before":
                goal_vector = np.array([0.0, -wall_robot_offset]) - robot.position
            elif robot.passage_state == "in":
                goal_vector = np.array([0.0, wall_robot_offset]) - robot.position
            elif robot.passage_state == "after":
                goal_vector = robot.goal_pos - robot.position
            a[i] = [goal_vector[1], goal_vector[0]]

        # if env.ts > 100:
        #   a[0][0] = -1.0
        # print(env.ts, a)
        obs, r, done, info = env.step(a)
        ret += r
        print(ret)
        # print(obs["gso"])
        env.render()
        # time.sleep(0.1)
        if done:
            #break
            env.reset()
