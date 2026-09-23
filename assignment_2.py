from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3
tolerance = [0,0]

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

def find_tolerance(tolerance, params):
    gravity = params["gravity"]
    length = params["length"]   
    
    tolerance[0] = np.arcsin(0.1)
    tolerance[1] = 0.1 * (gravity / length)
    return tolerance

def compute_ankle_torque(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    gravity_cancellation = -mass * gravity * length * np.sin(angle)
    inertia_cancellation = -mass * length ** 2 * angular_velocity 

    torque_min = -0.1 * mass * gravity * length
    torque_max = 0.05 * mass * gravity * length

    return np.clip(gravity_cancellation + inertia_cancellation, torque_min, torque_max)

def ankle_balance_convergence(state, params, timestep, sim_time, tolerance):
    alpha = params["angle_of_attack"]
    gamma = params["incline"]
    impact_angle = alpha + gamma
    local_state = state

    for step in range(round(sim_time/timestep)):
        time = step * timestep
        local_params["ankle_torque"] = compute_ankle_torque(local_state, params) 
        next_state = local_state + timestep * model.dynamics(t, local_state, params)
        if abs(next_state[0]) >= impact_angle:
            return False
        local_state = next_state   
    return abs(state[0]) <= tolerance[0] and abs(state[1]) <= tolerance[1]

def compute_roa


# Simulation loop. Replace this Euler step with your own integrator as needed.
for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]
    next_state = state + timestep * model.dynamics(t, state, params)

    if model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        completed_steps += 1

    state_traj[:, step + 1] = next_state
    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()

