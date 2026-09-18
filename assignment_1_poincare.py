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


# ---- Build the return map curve ----
initial_theta = -np.pi / params["number_of_spokes"] + params["slope_angle"]  # just after a reset
initial_omegas = np.linspace(0.1, 3.5, 40)

omega_n_list = []
omega_np1_list = []

for w0 in initial_omegas:
    w1 = model.one_step_return(w0, initial_theta, params, timestep)
    if w1 is not None:
        omega_n_list.append(w0)
        omega_np1_list.append(w1)

# ---- Find the fixed point: P(omega) - omega = 0 ----
def residual(omega0):
    w1 = model.one_step_return(omega0, initial_theta, params, timestep)
    return w1 - omega0

omega_fixed = model.bisection(residual, initial_omegas[0], initial_omegas[-1])

# ---- Estimate the Floquet multiplier via finite difference ----
epsilon = 1e-2  # perturbation size, relative to omega_fixed's scale

omega_minus = omega_fixed - epsilon
omega_plus = omega_fixed + epsilon

P_minus = model.one_step_return(omega_minus, initial_theta, params, timestep)
P_plus = model.one_step_return(omega_plus, initial_theta, params, timestep)

if P_minus is None or P_plus is None:
    print("Perturbed omega values did not produce a valid crossing — try a smaller epsilon or check bounds.")
else:
    floquet_multiplier = (P_plus - P_minus) / (2 * epsilon)

    print(f"omega_fixed = {omega_fixed:.6f}")
    print(f"P(omega_fixed - eps) = {P_minus:.6f}")
    print(f"P(omega_fixed + eps) = {P_plus:.6f}")
    print(f"Estimated Floquet multiplier: {floquet_multiplier:.6f}")

    if abs(floquet_multiplier) < 1:
        print("=> Limit cycle is locally stable (|lambda| < 1).")
    else:
        print("=> Limit cycle is locally unstable (|lambda| >= 1).")

# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.plot(omega_n_list, omega_np1_list, label=r"Return map $P(\omega)$")
plt.plot([initial_omegas[0], initial_omegas[-1]],
         [initial_omegas[0], initial_omegas[-1]],
         '--', label=r"Identity $\omega_{n+1} = \omega_n$")
plt.scatter([omega_fixed], [omega_fixed], color="green", s=100, zorder=5,
            label="Fixed point")
plt.xlabel(r"$\omega_n^+$ [rad/s]")
plt.ylabel(r"$\omega_{n+1}^+$ [rad/s]")
plt.title("Rimless Wheel Step-to-Step Return Map")
plt.legend()
plt.grid(True)
plt.show()

print(f"Fixed point: omega* = {omega_fixed:.4f} rad/s")