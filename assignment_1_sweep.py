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

ramp_angles = np.linspace(0.1, 1.0, 15)  # sweep from shallow to steep, all positive

percent_converged_ramp = []
floquet_ramp = []
fixed_point_ramp = []

for angle in ramp_angles:
    p = dict(base_params)
    p["slope_angle"] = angle

    omega_fixed, floquet = model.compute_fixed_point_and_floquet(
    p, timestep, omega_search_range=(0.1, 30.0), n_search_points=100
    )
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



p = dict(base_params)  # don't override slope_angle -- use whatever Sweep 1 uses
initial_theta = -np.pi / p["number_of_spokes"] + p["slope_angle"]

print(f"Testing slope_angle = {p['slope_angle']}")
for w0 in np.linspace(0.1, 10.0, 40):
    w1 = model.one_step_return(w0, initial_theta, p, timestep)
    if w1 is None:
        print(f"{w0:8.2f} | fell back")
    else:
        print(f"{w0:8.2f} | P={w1:.4f} | residual={w1 - w0:.4f}")





model.plot_sweep_with_floquet(
    spoke_counts, percent_converged_spokes, floquet_spokes,
    "Number of Spokes", "Effect of Number of Spokes on Convergence and Stability"
)

model.plot_sweep_with_floquet(
    ramp_angles, percent_converged_ramp, floquet_ramp,
    "Ramp Angle (rad)", "Effect of Ramp Angle on Convergence and Stability"
)