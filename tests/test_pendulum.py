import numpy as np

from integrators.rk4 import integrate
from models import pendulum

timestep = 0.01
sim_time = 100 * timestep  # a hundred steps


def test_energy():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    initial_state = np.array([0.5, 1.0])

    _, state_traj = integrate(
        pendulum.dynamics, timestep, sim_time, initial_state, params
    )
    kinetic_energy, potential_energy = pendulum.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy

    #checking that the total energy change is 0
    assert np.all(np.isclose(total_energy, total_energy[0]))


def test_torque_work():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 1.5
    initial_state = np.array([0.5, 1.0])

    _, state_traj = integrate(
        pendulum.dynamics, timestep, sim_time, initial_state, params
    )
    kinetic_energy, potential_energy = pendulum.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy
    work_adjusted_energy = total_energy - params["torque"] * state_traj[0]

    #Checking that the torque actually changes the energy
    assert not np.isclose(total_energy[-1], total_energy[0])
    #Checks the right magnitude of the energy after torque applied
    assert np.all(np.isclose(work_adjusted_energy, work_adjusted_energy[0]))


def test_damping_dissipation():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.3
    params["torque"] = 0.0
    initial_state = np.array([0.5, 1.0])

    time_traj, state_traj = integrate(
        pendulum.dynamics, timestep, sim_time, initial_state, params
    )
    kinetic_energy, potential_energy = pendulum.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy

    #checks that the energy decreases after every step
    assert np.all(np.diff(total_energy) < 0.0)
    #makes sure the energy loss is the same as the work done by tghe damper
    dissipated = params["damping_coeff"] * np.trapezoid(state_traj[1] ** 2, time_traj)
    assert np.isclose(total_energy[0] - total_energy[-1], dissipated, rtol=1e-4)
