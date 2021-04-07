from simple_env import SimpleEnv
import numpy as np
import rvo2


if __name__ == "__main__":
    env_config = {
        "world_shape": (4.0, 6.0),
        "wall_width": 0.5,
        "dt": 0.05,
        "n_agents": 3,
        "max_time_steps": 500,
        "communication_range": 2.0,
        "gap_length": 1.0,
        "grid_px_per_m": 40,
        "agent_radius": 0.3,
        "render": False,
        "max_lateral_speed": 2.0,
    }

    sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.3, 2)
    end = 0.6
    start = 4.0
    width = 0.1
    sim.addObstacle([(-2., -0.1), (-2., 0.1), (-0.6, 0.1), (-0.6, -0.1)])
    sim.processObstacles()

    rvo_agents = []
    for i in range(env_config["n_agents"]):
        rvo_agents.append(sim.addAgent((0, 0)))

    env = SimpleEnv(env_config)

    env.reset()
    ret = 0
    while True:
        env.render()

        for env_agent, rvo_agent in zip(env.robots, rvo_agents):
            p = env_agent.position
            sim.setAgentPosition(rvo_agent, (p[1], p[0]))
            goal_vector = env_agent.goal_pos - env_agent.position
            v = np.clip(np.array([goal_vector[1], goal_vector[0]]), -1.0, 1.0)
            sim.setAgentPrefVelocity(rvo_agent, tuple(v))
        sim.doStep()

        v = np.zeros((env.cfg["n_agents"], 2))
        for i, rvo_agent in enumerate(rvo_agents):
            v[i] = sim.getAgentVelocity(rvo_agent)

        obs, r, done, info = env.step(v)
        ret += r
        env.render()
        if done or all([np.linalg.norm(r.position - r.goal_pos) < 0.2 for r in env.robots]):
            env.reset()
