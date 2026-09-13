import numpy as np

def integrate(dynamics, timestep, sim_time, initial_state, params):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        k1 = dynamics(t, state_traj[:, step], params)
        k2 = dynamics(t + timestep / 2, state_traj[:, step] + (timestep / 2) * k1, params)
        k3 = dynamics(t + timestep / 2, state_traj[:, step] + (timestep / 2) * k2, params)
        k4 = dynamics(t + timestep, state_traj[:, step] + timestep * k3, params)

        state_traj[:, step + 1] = state_traj[:, step] + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return time_traj, state_traj