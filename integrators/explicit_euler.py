import numpy as np

def integrate(dynamics, timestep, sim_time, initial_state, params):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = state_traj[:, step] + timestep * dynamics(
            t, state_traj[:, step], params
        )
    return time_traj, state_traj


    