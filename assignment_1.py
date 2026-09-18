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
    "initial_angular_velocity": 0.0,  # initial angular velocity of the wheel (rads/s)
}

# set-up
initial_state = np.array([params["initial_angle"], params["initial_angular_velocity"]])

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop

# for step, t in enumerate(time_traj[:-1]):
#     state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
#         t, state_traj[:, step], params
#     )
#     if state_traj[0, step] >= np.pi / params["number_of_spokes"] + params["slope_angle"]:
#         state_traj[0, step + 1] = -np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * model.dynamics(t, state_traj[:, step], params)[0]
#         state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))
#     elif state_traj[0, step] <= -np.pi / params["number_of_spokes"] + params["slope_angle"]:
#         state_traj[0, step + 1] = np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * model.dynamics(t, state_traj[:, step], params)[0]
#         state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))

# kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

# Plotting Energy Graph
# plt.figure()
# plt.plot(time_traj, potential_energy, label="Potenti energy")
# plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
# plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
# plt.xlabel("Time (s)")
# plt.ylabel("Energy (J)")
# plt.title("Pendulum energy")
# plt.legend()
# plt.tight_layout()
# plt.show()

# Plotting Phase Portrait Graph
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

# Simulating attractor grid
red_tracker = 0
green_tracker = 0
for step_initial_angle in range(0, 10):
    print(f"Simulating initial angle: {step_initial_angle}")
    initial_angle = params["slope_angle"] - np.pi / params["number_of_spokes"] + step_initial_angle * (2 * np.pi / params["number_of_spokes"]) / 9
    for step_initial_angular_velocity in range(0, 7):
        print(f"Simulating initial angular velocity: {step_initial_angular_velocity}")
        initial_angular_velocity = params["initial_angular_velocity"] - 0.3 + 0.1 * step_initial_angular_velocity
        initial_state = np.array([initial_angle, initial_angular_velocity])
        print(f"Initial state: {initial_state}")
        state_traj, final_step = model.simulate_to_attractor(initial_state, params, timestep, sim_time)
        print(f"Final state: {state_traj[:, final_step]}")
        if state_traj[1, final_step] <= 0.05 and state_traj[1, final_step] >= -0.05:
            plt.scatter(
                state_traj[0, 0], state_traj[1, 0],
                color="red",
                label="Converged to rest" if red_tracker == 0 else None
                )
            red_tracker = red_tracker + 1
        else:
            plt.scatter(
                state_traj[0, 0], state_traj[1, 0],
                color="green",
                label="Converged to limit cycle" if green_tracker == 0 else None
                )
            green_tracker = green_tracker + 1

        step_initial_angular_velocity = step_initial_angular_velocity + 1
    step_initial_angle = 1
    step_initial_angle = step_initial_angle + 1

print(f"Final state: {state_traj[:, final_step]}")
plt.ylabel("Angular Velocity (rads/s)")
plt.xlabel("Angle (rads)")
plt.axvline(x=params["slope_angle"] + np.pi/params["number_of_spokes"], color="r", linestyle="--", label="Right Bound")
plt.axvline(x=params["slope_angle"] - np.pi/params["number_of_spokes"], color="r", linestyle="--", label="Left Bound")
plt.title("Rimless wheel phase portrait")
plt.legend()
plt.tight_layout()
plt.show()