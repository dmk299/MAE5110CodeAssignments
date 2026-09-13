import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import explicit_euler as integrator

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

timestep = 1e-3
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state


#Spokes Sweep

print("=" * 60)
print("SWEEP 1: Number of spokes")
print("=" * 60)

base_params = dict(params)
spoke_counts = np.arange(6, 12)

percent_converged_spokes = []
floquet_spokes = []
fixed_point_spokes = []

for n_spokes in spoke_counts:
    p = dict(base_params)
    p["number_of_spokes"] = n_spokes

    omega_fixed, floquet = model.compute_fixed_point_and_floquet(p, timestep)
    if omega_fixed is None:
        print(f"n_spokes={n_spokes}: no fixed point found, skipping")
        percent_converged_spokes.append(np.nan)
        floquet_spokes.append(np.nan)
        fixed_point_spokes.append(np.nan)
        continue

    pct = model.compute_roa_percent(p, timestep, sim_time, omega_fixed)

    percent_converged_spokes.append(pct)
    floquet_spokes.append(floquet)
    fixed_point_spokes.append(omega_fixed)

    print(f"n_spokes={n_spokes} -> {pct:.1f}% converged, "
          f"omega*={omega_fixed:.3f}, lambda={floquet:.3f}")


#Angle Sweep


print("=" * 60)
print("SWEEP 2: Ramp angle")
print("=" * 60)

ramp_angles = np.linspace(-0.7, -0.1, 13)

percent_converged_ramp = []
floquet_ramp = []
fixed_point_ramp = []

for angle in ramp_angles:
    p = dict(base_params)
    p["slope_angle"] = angle

    omega_fixed, floquet = model.compute_fixed_point_and_floquet(p, timestep)
    if omega_fixed is None:
        print(f"slope={angle:.3f}: no fixed point found, skipping")
        percent_converged_ramp.append(np.nan)
        floquet_ramp.append(np.nan)
        fixed_point_ramp.append(np.nan)
        continue

    pct = model.compute_roa_percent(p, timestep, sim_time, omega_fixed)

    percent_converged_ramp.append(pct)
    floquet_ramp.append(floquet)
    fixed_point_ramp.append(omega_fixed)

    print(f"slope={angle:.3f} -> {pct:.1f}% converged, "
          f"omega*={omega_fixed:.3f}, lambda={floquet:.3f}")



p = dict(base_params)
p["number_of_spokes"] = 6
p["slope_angle"] = -0.1   # shallow, per your target plot

upper = np.pi / p["number_of_spokes"] + p["slope_angle"]
lower = -np.pi / p["number_of_spokes"] + p["slope_angle"]

test_angles = np.array([
    lower + 0.05,
    lower + (upper - lower) * 0.25,
    p["slope_angle"],
    lower + (upper - lower) * 0.75,
    upper - 0.05,
])

test_velocities = np.linspace(-0.3, 0.3, 9)

print(f"Wedge bounds: lower={lower:.3f}, upper={upper:.3f}")
print(f"{'angle':>8} | {'v0':>6} | {'final_omega':>12} | result")
print("-" * 50)

results_grid = []
for angle in test_angles:
    row = []
    for v0 in test_velocities:
        initial_state = np.array([angle, v0])
        state_traj, final_step = model.simulate_to_attractor_fast(
            initial_state, p, timestep, sim_time,
            warmup_time=15.0, compare_time=3.0, energy_tol=0.0001
        )
        final_omega = state_traj[1, final_step]
        converged = not (-0.05 <= final_omega <= 0.05)
        row.append(converged)
        print(f"{angle:8.3f} | {v0:6.2f} | {final_omega:12.4f} | {'ROLLING' if converged else 'RESTED'}")
    results_grid.append(row)
    print()

results_grid = np.array(results_grid)
print(f"Total tested: {results_grid.size}, ROLLING: {results_grid.sum()}, RESTED: {(~results_grid).sum()}")

model.plot_sweep_with_floquet(
    spoke_counts, percent_converged_spokes, floquet_spokes,
    "Number of Spokes", "Effect of Number of Spokes on Convergence and Stability"
)

model.plot_sweep_with_floquet(
    ramp_angles, percent_converged_ramp, floquet_ramp,
    "Ramp Angle (rad)", "Effect of Ramp Angle on Convergence and Stability"
)