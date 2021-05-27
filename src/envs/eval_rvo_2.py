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

    # sim = rvo2.PyRVOSimulator(1/20., 2.0, 5, 1.5, 2, 0.3, 2)

    sim = rvo2.PyRVOSimulator(timeStep=1/20., neighborDist=3.0, maxNeighbors=3,
                              timeHorizon=1.5, timeHorizonObst=2, radius=0.3, maxSpeed=0.5,
                              velocity=(0.1, 0.1))

    # 		RVOSimulator(float timeStep, float neighborDist, size_t maxNeighbors,
    # 					 float timeHorizon, float timeHorizonObst, float radius,
    # 					 float maxSpeed, const Vector2 &velocity = Vector2());
    # 		 * \brief      Constructs a simulator instance and sets the default
    # 		 *             properties for any new agent that is added.
    # 		 * \param      timeStep        The time step of the simulation.
    # 		 *                             Must be positive.
    # 		 * \param      neighborDist    The default maximum distance (center point
    # 		 *                             to center point) to other agents a new agent
    # 		 *                             takes into account in the navigation. The
    # 		 *                             larger this number, the longer he running
    # 		 *                             time of the simulation. If the number is too
    # 		 *                             low, the simulation will not be safe. Must be
    # 		 *                             non-negative.
    # 		 * \param      maxNeighbors    The default maximum number of other agents a
    # 		 *                             new agent takes into account in the
    # 		 *                             navigation. The larger this number, the
    # 		 *                             longer the running time of the simulation.
    # 		 *                             If the number is too low, the simulation
    # 		 *                             will not be safe.
    # 		 * \param      timeHorizon     The default minimal amount of time for which
    # 		 *                             a new agent's velocities that are computed
    # 		 *                             by the simulation are safe with respect to
    # 		 *                             other agents. The larger this number, the
    # 		 *                             sooner an agent will respond to the presence
    # 		 *                             of other agents, but the less freedom the
    # 		 *                             agent has in choosing its velocities.
    # 		 *                             Must be positive.
    # 		 * \param      timeHorizonObst The default minimal amount of time for which
    # 		 *                             a new agent's velocities that are computed
    # 		 *                             by the simulation are safe with respect to
    # 		 *                             obstacles. The larger this number, the
    # 		 *                             sooner an agent will respond to the presence
    # 		 *                             of obstacles, but the less freedom the agent
    # 		 *                             has in choosing its velocities.
    # 		 *                             Must be positive.
    # 		 * \param      radius          The default radius of a new agent.
    # 		 *                             Must be non-negative.
    # 		 * \param      maxSpeed        The default maximum speed of a new agent.
    # 		 *                             Must be non-negative.
    # 		 * \param      velocity        The default initial two-dimensional linear
    # 		 *                             velocity of a new agent (optional).
    end = 0.6
    start = 4.0
    width = 0.1



    # Counter CLockerwise

    # y x
    # sim.addObstacle([(0.1, -0.5), (0.1, -2.0), (-0.1, -2.0), (-0.1, 0.5)])

    # sim.addObstacle([(0.1, -1.5), (0.1, -2.0), (-0.1, -2.0), (-0.1, -1.5)])

    # sim.addObstacle([(0.1, 2.), ( 0.1, -2.0), ( -0.1 ,-2.0), ( -0.1, 2.0)])

    num_obstalce = 10
    for i in range(num_obstalce):
        sim.addObstacle([(0.1, -1.5), (0.1, -2.0), (-0.1, -2.0), (-0.1, -1.5)])


    # x y
    # sim.addObstacle([(-0.5, 0.1), (-2.0, 0.1), (-2.0, -0.1), (0.5,-0.1)])
    # sim.addObstacle([(-2, 0.1), (-2.0, 0.1), (-2.0, -0.1), (2.0,-0.1)])


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
