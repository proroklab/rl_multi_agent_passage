from turtlebot import SimEnv
import pybullet as p
import numpy as np

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

test_env_keyboard()

