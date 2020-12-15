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
        reward = -dst_goal
        #reward = 0
        #if dst_goal < self.closest_dist_to_goal:
        #    reward = self.closest_dist_to_goal - dst_goal
        #    self.closest_dist_to_goal = dst_goal
        done = dst_goal < 0.1
        return obs, reward, done

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        n_agents = len(self.cfg['agent_poses'])
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=float)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

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
        print("RESE")
        self.timestep = 0
        for robot in self.robots:
            robot.reset()
        return self.step([0, 0])[0]

    def step(self, action):
        self.timestep += 1
        self.robots[0].set_velocity(action[0], action[1])
        o, r, d = self.robots[0].step()

        p.stepSimulation(physicsClientId=self.client)

        done = d or self.timestep > self.cfg['max_time_steps']
        return o, r, d, {}

