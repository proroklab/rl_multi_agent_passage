import pybullet as p
import pybullet_data
import time
import gym
import numpy as np
from pathlib import Path

class Turtlebot():
    CONFIG = {
        'limits': {
            'forward_speed': (-0.2, 0.2),
            'yaw_rate': (-np.pi/8, np.pi/8)
        },
        'lidar': {
            'range': 4,
            'rays': 8,
            'rays_start': -np.pi/2,
            'rays_end': np.pi/2,
        }
    }

    def __init__(self, pybullet_client, reset_pos, goal_pos):
        self.pybullet_client = pybullet_client
        self.initial_pos = self.position = np.array(reset_pos)
        self.goal_pos = np.array(goal_pos)
        self.body_id = p.loadURDF("turtlebot3.urdf", basePosition=self.initial_pos, physicsClientId=self.pybullet_client)
        self.lidar_lines = [-1]*16
        self.lidar_base = np.array([-0.03, 0, 0.18])

        nearPlane = 0.01
        farPlane = 100
        fov = 60
        self.pixelWidth = 30
        self.pixelHeight = 20
        aspect = self.pixelWidth / self.pixelHeight
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        self.reset()	

    def reset(self):
        self.lateral_speed = np.array([0,0,0])
        self.ang_speed = np.array([0,0,0])

        p.resetBasePositionAndOrientation(self.body_id, self.initial_pos, p.getQuaternionFromEuler([0,0,90*(np.pi/180)]), physicsClientId=self.pybullet_client)
        self.update_state()
        self.setpoint_forward_speed = 0
        self.setpoint_yaw_rate = 0
        self.reached_goal = False

    def update_state(self):
        self.last_lateral_speed = self.lateral_speed.copy()
        self.last_ang_speed = self.ang_speed.copy()

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
        #if angular > 0 and lateral > 0:
        if np.isnan(angular) or np.isnan(lateral):
            import pdb; pdb.set_trace()
        #assert not np.isnan(angular)
        #assert not np.isnan(lateral)
        self.setpoint_forward_speed = np.clip(lateral, *self.CONFIG['limits']['forward_speed'])
        self.setpoint_yaw_rate = np.clip(angular, *self.CONFIG['limits']['yaw_rate'])

    def get_camera(self):
        com_p, com_o, _, _, _, _ = p.getLinkState(self.body_id, 7)
        rot_matrix = p.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0) # z-axis
        init_up_vector = (0, 0, 1) # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        viewMatrix = p.computeViewMatrix(com_p, com_p + camera_vector, up_vector)
        return p.getCameraImage(self.pixelWidth,
                                  self.pixelHeight,
                                  viewMatrix,
                                  self.projectionMatrix,
                                  shadow=1,
                                  lightDirection=[1, 1, 1],
                                  renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                  physicsClientId=self.pybullet_client)
                                      
    def step(self):
        self.update_state()

        wheel_distance = 0.157
        wheel_radius = 0.033
        v_l = self.setpoint_forward_speed / wheel_radius - ((self.setpoint_yaw_rate * wheel_distance) / (2 * wheel_radius));
        v_r = self.setpoint_forward_speed / wheel_radius + ((self.setpoint_yaw_rate * wheel_distance) / (2 * wheel_radius));

        p.setJointMotorControl2(self.body_id, 2, p.VELOCITY_CONTROL, targetVelocity=v_r, force=1000, physicsClientId=self.pybullet_client)
        p.setJointMotorControl2(self.body_id, 3, p.VELOCITY_CONTROL, targetVelocity=v_l, force=1000, physicsClientId=self.pybullet_client)

        m = self.get_rotation_matrix()
        goal_relative = (self.goal_pos - self.position) @ m
        gap_relative_l = (np.array([0, -0.2, 0]) - self.position) @ m
        gap_relative_r = (np.array([0, 0.2, 0]) - self.position) @ m


        cfg_lidar = self.CONFIG['lidar']
        rays_range = cfg_lidar['rays_end'] - cfg_lidar['rays_start']
        base = self.position + m @ self.lidar_base
        start = np.pi/2 + cfg_lidar['rays_start']
        lidar_rays = p.rayTestBatch(
            [base]*cfg_lidar['rays'],
            [base + m @ np.array([
                cfg_lidar['range']*np.sin(start + i*rays_range/cfg_lidar['rays']),
                cfg_lidar['range']*np.cos(start + i*rays_range/cfg_lidar['rays']),
                0]) for i in range(cfg_lidar['rays'])
            ]
        )
        self.lidar_rays = [v[2] for v in lidar_rays]


        obs = np.hstack([
            self.world_lateral_speed,
            self.world_ang_speed,
            #self.last_lateral_speed,
            #self.last_ang_speed,
            np.sin(self.orientation_euler),
            np.cos(self.orientation_euler),
            self.position,
            #gap_relative_l,
            #gap_relative_r,
            #goal_relative,
            self.goal_pos,
            #self.goal_pos,
            #self.lidar_rays
        ])#, self.get_camera()

        return obs

    def render_lidar(self):
        cfg_lidar = self.CONFIG['lidar']
        rays_range = cfg_lidar['rays_end'] - cfg_lidar['rays_start']
        m = self.get_rotation_matrix()
        base = self.position + m @ self.lidar_base
        start = np.pi/2 + cfg_lidar['rays_start']
        for i, r in enumerate(self.lidar_rays):
            self.lidar_lines[i] = p.addUserDebugLine(
                base,
                base + m @ np.array([
                    r*cfg_lidar['range']*np.sin(start + i*rays_range/len(self.lidar_rays)),
                    r*cfg_lidar['range']*np.cos(start + i*rays_range/len(self.lidar_rays)),
                    0
                ]),
                (1,0,0),
                replaceItemUniqueId=self.lidar_lines[i]
            )

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        n_agents = len(self.cfg['agent_poses'])
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=float),)*n_agents) # velocity yaw and forward

        ''''
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Tuple((
                    #gym.spaces.Box(low=-np.pi/8, high=np.pi/8, shape=(1,), dtype=float),
                    #gym.spaces.Box(low=-0.2, high=0.2, shape=(1,), dtype=float)
                    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float),
                    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
                ))
            ,)*n_agents) # velocity yaw and forward
        '''

        self.observation_space = gym.spaces.Dict({
            # current pose relative to goal (x,y)
            # current pose relative to origin (and therefore gap in wall) (x, y, phi)
            # current velocity (lin, ang)
            'agents': gym.spaces.Tuple((
                gym.spaces.Dict({
                    "obs": gym.spaces.Box(-10000, 10000, shape=(18,), dtype=float),
                    #"img": gym.spaces.Box(0, 1, shape=(20, 30, 2), dtype=int),
                    "state": gym.spaces.Box(low=-10000, high=10000, shape=(6,))
                })
            ,)*n_agents),
            'gso': gym.spaces.Box(-1, 1, shape=(n_agents, n_agents), dtype=float),
        })

        self.client = p.connect(p.GUI if self.cfg['render'] else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.setAdditionalSearchPath(str(Path(__file__).parent / "data"))
        self.wall_id = -1 #p.loadURDF("wall.urdf", physicsClientId=self.client)

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
        observations, dones, infos = [], [], {'rewards': {}}
        all_reward = 0
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            robot.set_velocity(action[0], action[1])
            observation = robot.step()
            #segmentation = np.array(img_data[4])
            #segmentation_proc = np.zeros((robot.pixelHeight, robot.pixelWidth, 2), dtype=np.float)
            #segmentation_proc[..., 0][segmentation == self.wall_id] = 1
            #for r in self.robots:
            #    segmentation_proc[..., 1][segmentation == r.body_id] = 1
            
            goal_vector = robot.goal_pos - robot.position
            reward = np.dot(goal_vector/np.linalg.norm(goal_vector), robot.world_lateral_speed)*np.linalg.norm(robot.world_lateral_speed)
            done = np.sqrt(np.sum(goal_vector**2)) < 0.1
            if done and not robot.reached_goal:
                reward = 200
                robot.reached_goal = True
            collision_idx = set([o[2] for o in p.getContactPoints(bodyA=robot.body_id, physicsClientId=self.client)])
            if self.wall_id in collision_idx:
                reward -= 0.1
            if any([not (i == self.wall_id) and not (i == self.plane_id) for i in collision_idx]):
                reward -= 1

            if not np.all(np.isfinite(observation)) or not np.isfinite(reward):
                import pdb; pdb.set_trace()

            observations.append({
                "obs": observation,
                #"img": segmentation_proc,
                "state": np.hstack([
                    #robot.position,
                    np.sin(robot.orientation_euler),
                    np.cos(robot.orientation_euler)
                ])
            })
            dones.append(done)
            infos['rewards'][i] = reward
            all_reward += reward

        #self.robots[0].render_lidar()

        p.stepSimulation(physicsClientId=self.client)

        #print(img_arr[4])
        #import pdb; pdb.set_trace()
        all_obs = {
            'agents': tuple(observations),
            'gso': self.compute_gso(),
        }
        #if not np.all(np.isfinite(obs['state'])) or not np.all(np.isfinite(obs['gso'])):
        #     import pdb; pdb.set_trace()

        done = all(dones) or self.timestep > self.cfg['max_time_steps']
        return all_obs, all_reward, done, infos

class CentrSimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        n_agents = len(self.cfg['agent_poses'])
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2*n_agents,), dtype=float)
        #self.action_space = gym.spaces.Box(low=np.array([-np.pi/8, -0.2]*n_agents), high=np.array([np.pi/8, 0.2]*n_agents), shape=(2*n_agents,), dtype=float)
        #self.action_space = gym.spaces.Tuple((
            #gym.spaces.Box(low=-np.pi/8, high=np.pi/8, shape=(1,), dtype=float),
            #gym.spaces.Box(low=-0.2, high=0.2, shape=(1,), dtype=float)
        #    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float),
        #    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
        #))
        self.observation_space = gym.spaces.Box(-10000, 10000, shape=((9+8)*n_agents,), dtype=float)

        self.client = p.connect(p.GUI if self.cfg['render'] else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.setAdditionalSearchPath(str(Path(__file__).parent / "data"))
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
        return self.step([0, 0]*len(self.robots))[0]

    def step(self, actions):
        self.timestep += 1
        observations, dones = [], []
        all_reward = 0
        #import pdb; pdb.set_trace()
        for i, (robot, action) in enumerate(zip(self.robots, np.array(actions).reshape(len(self.robots), -1))):
            robot.set_velocity(action[0], action[1])
            observation = robot.step()

            goal_vector = robot.goal_pos - robot.position
            reward = np.dot(goal_vector/np.linalg.norm(goal_vector), robot.world_lateral_speed)*np.linalg.norm(robot.world_lateral_speed)
            #if not robot.reached_middle and robot.position[1] > 0:
            #    reward = 5
            #    robot.reached_middle = True

            done = np.sum(goal_vector**2) < 0.05**2
            if done:
                reward = 200
            collision_idx = set([o[2] for o in p.getContactPoints(bodyA=robot.body_id, physicsClientId=self.client)])
            if self.wall_id in collision_idx:
                reward -= 0.1
                #done = True
            #if any([not (i == self.wall_id) and not (i == self.plane_id) for i in collision_idx]):
            #    reward -= 0.01

            if not np.all(np.isfinite(observation)) or not np.isfinite(reward):
                import pdb; pdb.set_trace()

            #if reward < 1e-5:
            #    reward = 0
            observations.append(observation)
            dones.append(done)
            all_reward += reward

        #self.robots[0].render_lidar()

        p.stepSimulation(physicsClientId=self.client)

        done = any(dones) or self.timestep > self.cfg['max_time_steps']
        return np.concatenate(observations), all_reward, done, {}

