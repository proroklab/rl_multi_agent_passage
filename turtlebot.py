import pybullet as p
import pybullet_data
import time
import gym
import numpy as np

class Turtlebot():
    CONFIG = {
        'limits': {
            'forward_speed': (-0.2, 0.2),
            'yaw_rate': (-np.pi/8, np.pi/8),
        },
    }

    def __init__(self, pybullet_client, reset_pos, goal_pos):
        self.pybullet_client = pybullet_client
        self.initial_pos = self.position = np.array(reset_pos)
        self.goal_pos = np.array(goal_pos)
        self.body_id = p.loadURDF("turtlebot3.urdf", basePosition=self.initial_pos, physicsClientId=self.pybullet_client)

        self.reset()	

    def reset(self):
        p.resetBasePositionAndOrientation(self.body_id, self.initial_pos, p.getQuaternionFromEuler([0,0,90*(np.pi/180)]), physicsClientId=self.pybullet_client)
        self.update_state()
        self.closest_dist_to_goal = np.sqrt(np.sum((self.position - self.goal_pos)**2))
        self.setpoint_forward_speed = 0
        self.setpoint_yaw_rate = 0

    def update_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.pybullet_client)
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.orientation_euler = np.array(p.getEulerFromQuaternion(orientation))

        m = self.get_rotation_matrix()
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id, self.pybullet_client)
        self.world_lateral_speed = np.array(lin_vel)
        self.lateral_speed = self.world_lateral_speed @ m
        self.world_ang_speed = np.array(ang_vel)
        self.ang_speed = self.world_ang_speed @ m

    def get_rotation_matrix(self, orientation=None):
        return np.array(p.getMatrixFromQuaternion(self.orientation if orientation is None else orientation)).reshape(3,3)

    def set_velocity(self, angular, lateral):
        assert not np.isnan(angular)
        assert not np.isnan(lateral)
        self.setpoint_forward_speed = np.clip(lateral, *self.CONFIG['limits']['forward_speed'])
        self.setpoint_yaw_rate = np.clip(angular, *self.CONFIG['limits']['yaw_rate'])

    def step(self):
        self.update_state()

        wheel_distance = 0.157
        wheel_radius = 0.033
        v_l = self.setpoint_forward_speed / wheel_radius - ((self.setpoint_yaw_rate * wheel_distance) / (2 * wheel_radius));
        v_r = self.setpoint_forward_speed / wheel_radius + ((self.setpoint_yaw_rate * wheel_distance) / (2 * wheel_radius));

        p.setJointMotorControl2(self.body_id, 1, p.VELOCITY_CONTROL, targetVelocity=v_r, force=1000, physicsClientId=self.pybullet_client)
        p.setJointMotorControl2(self.body_id, 2, p.VELOCITY_CONTROL, targetVelocity=v_l, force=1000, physicsClientId=self.pybullet_client)

        origin_pose = np.array([0, 0, 0])
        m = self.get_rotation_matrix()
        goal_relative = (self.goal_pos - self.position) @ m
        gap_relative = (origin_pose - self.position) @ m
        speed = np.array([self.lateral_speed[0], self.ang_speed[2]])
        obs = np.hstack([gap_relative, goal_relative, speed])
        dst_goal = np.sqrt(np.sum(goal_relative**2))
        reward = 0
        if dst_goal < self.closest_dist_to_goal:
            reward = self.closest_dist_to_goal - dst_goal
            self.closest_dist_to_goal = dst_goal
        done = dst_goal < 0.1
        return obs, reward, done

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        n_agents = len(self.cfg['agent_poses'])
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=float),)*n_agents) # velocity yaw and forward

        self.observation_space = gym.spaces.Dict({
            # current pose relative to goal (x,y)
            # current pose relative to origin (and therefore gap in wall) (x, y, phi)
            # current velocity (lin, ang)
            'agent_obs': gym.spaces.Tuple(
                (gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),)*n_agents
            ),
            'gso': gym.spaces.Box(-np.inf, np.inf, shape=(n_agents, n_agents)),
            # agent obs and current absolute pose
            'state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_agents, 11)),
        })

        self.client = p.connect(p.GUI if self.cfg['render'] else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.wall_id = p.loadURDF("wall.urdf", physicsClientId=self.client)

        self.robots = []
        for initial_pose, goal_pose in zip(self.cfg['agent_poses'], self.cfg['agent_goals']):
            self.robots.append(Turtlebot(self.client, initial_pose, goal_pose))

        self.reset()

    def __del__(self):
        p.disconnect()

    def reset(self):
        self.timestep = 0
        for robot in self.robots:
            robot.reset()
        return self.step([[0, 0]]*len(self.robots))[0]

    def compute_gso(self):
        dists = np.zeros((len(self.robots), len(self.robots)))
        for agent_y in range(len(self.robots)):
            for agent_x in range(len(self.robots)):
                dst = np.sum(np.array(self.robots[agent_x].position - self.robots[agent_y].position)**2)
                dists[agent_y, agent_x] = dst
                dists[agent_x, agent_y] = dst

        A = dists < (self.cfg['communication_range']**2)

        # normalization: refer https://github.com/QingbiaoLi/GraphNets/blob/master/Flocking/Utils/dataTools.py#L601
        np.fill_diagonal(A, 0)
        deg = np.sum(A, axis = 1) # nNodes (degree vector)
        D = np.diag(deg)
        Dp = np.diag(np.nan_to_num(np.power(deg, -1/2)))
        L = A # D-A
        gso = Dp @ L @ Dp
        return gso

    def step(self, actions):
        self.timestep += 1
        obs, dones, infos = [], [], {'rewards': {}}
        reward = 0
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            robot.set_velocity(action[0], action[1])
            o, r, d = robot.step()
            obs.append(o)
            dones.append(d)
            infos['rewards'][i] = r
            reward += r

        p.stepSimulation(physicsClientId=self.client)

        state = []
        for r, o in zip(self.robots, obs):
            state.append(np.concatenate([o, np.array(r.position)]))
        obs = {
            'agent_obs': tuple(obs),
            'gso': self.compute_gso(),
            'state': state
        }
        #print(self.timestep)
        done = all(dones) or self.timestep > self.cfg['max_time_steps']
        #print("INF", actions, infos, len(self.robots))
        return obs, reward, done, infos

def test_env_keyboard():

    env = SimEnv({
        'agent_poses': [
            [-0.3, -0.5, 0],
            #[0.3, -0.5, 0],
        ],
        'agent_goals': [
            [0.3, 0.5, 0],
            #[-0.3, 0.5, 0]
        ],
        'max_time_steps': 40000,
        'communication_range': 2.0,
        'render': True,
    })
    env.reset()
    rewards = 0
    turn = 0
    forward = 0
    while True:
        keys = p.getKeyboardEvents()

        for k,v in keys.items():
            if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    turn = -np.pi/8
            if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                    turn = 0
            if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    turn = np.pi/8
            if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                    turn = 0

            if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    forward=10
            if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                    forward=0
            if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    forward=-10
            if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                    forward=0
        obs, reward, done, info = env.step([[turn, forward]])
        rewards += reward
        
        #print(rewards)
        if done:
            rewards = 0
            turn = 0
            forward = 0
            env.reset()

#test_env_keyboard()

def test_env():
    env = SimEnv({
        'agent_poses': [
            [-0.3, -0.5, 0],
            #[0.3, -0.5, 0],
        ],
        'agent_goals': [
            [0.3, 0.5, 0],
            #[-0.3, 0.5, 0]
        ],
        'max_time_steps': 4000,
        'communication_range': 2.0,
        'render': True,
    })
    env.reset()
    for i in range(1):
        actions = [[[np.pi/8, 0]]]*300 + [[[0, 0.2]]]*800 + [[[-np.pi/8, 0]]]*300 + [[[0, 0.2]]]*100 + [[[np.pi/8, 0]]]*300 + [[[0, 0.2]]]*400
        rewards = 0
        for a in actions:
            obs, reward, done, info = env.step(a)
            rewards += reward
        print(rewards, done)
        np.set_printoptions(suppress=True)
        time.sleep(1/240)
#test_env()
