import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import explicit_euler as integrator

# Basic Simulation of rimless wheel

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # spoke length (m)
    "mass": 0.2,  # point mass at center of wheel (kg)
    "number_of_spokes": 6,  # number of spokes
    "slope_angle": np.radians(30),  # slope of the ground (radians): 
    "initial_angle": 0.2,  # initial angle of the wheel (radians)
}

# set-up
initial_state = np.array([params["initial_angle"], 0.0])

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop

for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
        t, state_traj[:, step], params
    )
    if state_traj[0, step] >= np.pi / params["number_of_spokes"] + params["slope_angle"]:
        state_traj[0, step + 1] = -np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * model.dynamics(t, state_traj[:, step], params)[0]
        state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))
    elif state_traj[0, step] <= -np.pi / params["number_of_spokes"] + params["slope_angle"]:
        state_traj[0, step + 1] = np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * model.dynamics(t, state_traj[:, step], params)[0]
        state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

print("impact 1 ", state_traj[1, 100] * np.cos(2 * (np.pi / params["number_of_spokes"])))
print("impact 0 ", state_traj[1, 2] * np.cos(2 * (np.pi / params["number_of_spokes"])))
print("state_traj1 ", state_traj[0, 100])
print("state_traj0 ", state_traj[1, 100])
print("acceleration", model.dynamics(t, state_traj[:, 0], params))
print("reverse", -np.pi / params["number_of_spokes"] + params["slope_angle"])
print("forward", np.pi / params["number_of_spokes"] + params["slope_angle"])

plt.figure()
plt.plot(time_traj, potential_energy, label="Potenti energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()


# plt.figure()
# plt.plot(state_traj[0], state_traj[1], label="Phase portrait")
# plt.ylabel("Angular Velocity (rads/s)")
# plt.xlabel("Angle (rads)")
# plt.axvline(x=params["slope_angle"] + np.pi/params["number_of_spokes"], color="r", linestyle="--", label="Right Bound")
# plt.axvline(x=params["slope_angle"] - np.pi/params["number_of_spokes"], color="r", linestyle="--", label="Left Bound")
# plt.title("Rimless wheel phase portrait")
# plt.legend()
# plt.tight_layout()
# plt.show()