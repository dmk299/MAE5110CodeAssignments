import numpy as np


def explicit_euler(dynamics, t, state, timestep, params):
    """Advance state by one explicit Euler step."""
    return state + timestep * dynamics(t, state, params)


def integrate(dynamics, timestep, sim_time, initial_state, params):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((np.size(initial_state), n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = explicit_euler(
            dynamics, t, state_traj[:, step], timestep, params
        )
    return time_traj, state_traj
