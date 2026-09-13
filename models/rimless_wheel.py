from pyexpat import model

import numpy as np
import matplotlib.pyplot as plt

def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        gravity * np.sin(angle)
    ) / (length)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy

def simulate_to_attractor(initial_state, params, timestep, sim_time):
    """Simulate the rimless wheel dynamics until it reaches an attractor."""

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        if step <= 50000:
            state_traj[:, step + 1] = state_traj[:, step] + timestep * dynamics(t, state_traj[:, step], params)
            if state_traj[0, step] >= np.pi / params["number_of_spokes"] + params["slope_angle"]:
                state_traj[0, step + 1] = -np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * dynamics(t, state_traj[:, step], params)[0]
                state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))
            elif state_traj[0, step] <= -np.pi / params["number_of_spokes"] + params["slope_angle"]:
                state_traj[0, step + 1] = np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * dynamics(t, state_traj[:, step], params)[0]
                state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))
        else:
            #print(f"Step: {step}, Time: {t:.4f} s, Angle: {state_traj[0, step]:.4f} rad, Angular Velocity: {state_traj[1, step]:.4f} rad/s")
            kinetic_energy, potential_energy = calculate_energy(state_traj, params) 
        if step >= 50001 and 0.999 * (kinetic_energy[step-10000] + potential_energy[step-10000]) <= kinetic_energy[step] + potential_energy[step] <= 1.001 * (kinetic_energy[step-10000] + potential_energy[step-10000]):
            final_step = step
            
            break
        else:
            state_traj[:, step + 1] = state_traj[:, step] + timestep * dynamics(
                t, state_traj[:, step], params
            )
            if state_traj[0, step] >= np.pi / params["number_of_spokes"] + params["slope_angle"]:
                state_traj[0, step + 1] = -np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * dynamics(t, state_traj[:, step], params)[0]
                state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))
            elif state_traj[0, step] <= -np.pi / params["number_of_spokes"] + params["slope_angle"]:
                state_traj[0, step + 1] = np.pi / params["number_of_spokes"] + params["slope_angle"] + timestep * dynamics(t, state_traj[:, step], params)[0]
                state_traj[1, step + 1] = state_traj[1, step] * np.cos(2 * (np.pi / params["number_of_spokes"]))
    return state_traj, final_step


def simulate_until_n_crossings(initial_omega, initial_theta, params, timestep,
                                 n_crossings=2, max_steps=100000):
    """
    Simulate the rimless wheel starting from a given angle/angular velocity,
    and return the list of angular velocities recorded at each contact
    (Poincare section) event, stopping once n_crossings have been recorded.
    """
    state = np.array([initial_theta, initial_omega])
    omega_crossings = []
    t = 0.0

    for _ in range(max_steps):
        new_state = state + timestep * dynamics(t, state, params)

        if state[0] > np.pi / params["number_of_spokes"] + params["slope_angle"]:
            new_state[0] = (-np.pi / params["number_of_spokes"] + params["slope_angle"]
                             + timestep * dynamics(t, state, params)[0])
            new_state[1] = state[1] * np.cos(2 * (np.pi / params["number_of_spokes"]))
            omega_crossings.append(new_state[1])

        elif state[0] < -np.pi / params["number_of_spokes"] + params["slope_angle"]:
            new_state[0] = (np.pi / params["number_of_spokes"] + params["slope_angle"]
                             + timestep * dynamics(t, state, params)[0])
            new_state[1] = state[1] * np.cos(2 * (np.pi / params["number_of_spokes"]))
            omega_crossings.append(new_state[1])

        state = new_state
        t += timestep

        if len(omega_crossings) >= n_crossings:
            break

    return omega_crossings


def one_step_return(omega0, initial_theta, params, timestep):
    """Given omega_n at a crossing, return omega_{n+1} at the next crossing."""
    crossings = simulate_until_n_crossings(omega0, initial_theta, params, timestep,
                                             n_crossings=1)
    if len(crossings) < 1:
        return None
    return crossings[0]

def bisection(f, a, b, tol=1e-8, max_iter=200):
    """Find a root of f between a and b using bisection. Assumes f(a) and f(b) have opposite signs."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for bisection to work.")

    for _ in range(max_iter):
        midpoint = (a + b) / 2
        fm = f(midpoint)

        if abs(fm) < tol or (b - a) / 2 < tol:
            return midpoint

        if fa * fm < 0:
            b = midpoint
            fb = fm
        else:
            a = midpoint
            fa = fm

    return (a + b) / 2

def simulate_to_attractor_fast(initial_state, params, timestep, sim_time,
                                 warmup_time=5.0, compare_time=1.0, energy_tol=0.001):
    """Faster version of simulate_to_attractor: single integration per step,
    incremental energy tracking instead of recomputing the whole trajectory."""

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    warmup_steps = int(warmup_time / timestep)
    compare_window = int(compare_time / timestep)

    kinetic_energy = np.zeros(n_timesteps)
    potential_energy = np.zeros(n_timesteps)
    kinetic_energy[0], potential_energy[0] = calculate_energy(state_traj[:, 0], params)

    upper = np.pi / params["number_of_spokes"] + params["slope_angle"]
    lower = -np.pi / params["number_of_spokes"] + params["slope_angle"]

    final_step = n_timesteps - 1  # fallback if it never converges

    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        deriv = dynamics(t, state, params)
        new_state = state + timestep * deriv

        if state[0] >= upper:
            new_state[0] = lower + timestep * deriv[0]
            new_state[1] = state[1] * np.cos(2 * np.pi / params["number_of_spokes"])
        elif state[0] <= lower:
            new_state[0] = upper + timestep * deriv[0]
            new_state[1] = state[1] * np.cos(2 * np.pi / params["number_of_spokes"])

        state_traj[:, step + 1] = new_state
        ke, pe = calculate_energy(new_state, params)
        kinetic_energy[step + 1] = ke
        potential_energy[step + 1] = pe

        if step + 1 >= warmup_steps + compare_window:
            e_now = kinetic_energy[step + 1] + potential_energy[step + 1]
            e_prev = (kinetic_energy[step + 1 - compare_window]
                      + potential_energy[step + 1 - compare_window])
            if (1 - energy_tol) * e_prev <= e_now <= (1 + energy_tol) * e_prev:
                final_step = step + 1
                break

    return state_traj, final_step


def estimate_slope(omega0, eps, initial_theta, params, timestep):
    """Central-difference estimate of the local slope of P(omega) at omega0."""
    P_minus = one_step_return(omega0 - eps, initial_theta, params, timestep)
    P_plus = one_step_return(omega0 + eps, initial_theta, params, timestep)
    if P_minus is None or P_plus is None:
        return None
    return (P_plus - P_minus) / (2 * eps)


def compute_fixed_point_and_floquet(params, timestep,
                                      omega_search_range=(0.1, 4.0), n_search_points=40,
                                      slope_eps=1e-2):
    """Find the return-map fixed point and local slope (Floquet multiplier)."""
    initial_theta = -np.pi / params["number_of_spokes"] + params["slope_angle"]

    search_omegas = np.linspace(*omega_search_range, n_search_points)
    residuals, valid_omegas = [], []
    for w0 in search_omegas:
        w1 = one_step_return(w0, initial_theta, params, timestep)
        if w1 is not None:
            residuals.append(w1 - w0)
            valid_omegas.append(w0)

    residuals = np.array(residuals)
    valid_omegas = np.array(valid_omegas)

    if len(residuals) < 2:
        return None, None

    sign_changes = np.where(np.diff(np.sign(residuals)) != 0)[0]
    if len(sign_changes) == 0:
        return None, None

    idx = sign_changes[0]
    a, b = valid_omegas[idx], valid_omegas[idx + 1]

    def residual(w0):
        w1 = one_step_return(w0, initial_theta, params, timestep)
        return w1 - w0 if w1 is not None else np.nan

    try:
        omega_fixed = bisection(residual, a, b)
    except ValueError:
        return None, None

    floquet = estimate_slope(omega_fixed, slope_eps, initial_theta, params, timestep)
    return omega_fixed, floquet


def compute_roa_percent(params, timestep, sim_time, omega_fixed,
                          n_angle_steps=5, n_velocity_steps=6,
                          rest_threshold=0.05):
    converged_count = 0
    total_count = 0

    for step_initial_angle in range(n_angle_steps):
        initial_angle = (params["slope_angle"] - np.pi / params["number_of_spokes"]
                          + step_initial_angle * (2 * np.pi / params["number_of_spokes"])
                          / (n_angle_steps - 1))

        # span from near-zero up to a bit above omega_fixed
        velocities = np.linspace(0.0, max(omega_fixed * 1.5, 1.0), n_velocity_steps)

        for v0 in velocities:
            initial_state = np.array([initial_angle, v0])
            state_traj, final_step = simulate_to_attractor_fast(
                initial_state, params, timestep, sim_time,
                warmup_time=15.0, compare_time=3.0, energy_tol=0.0001
            )
            total_count += 1

            if not (-rest_threshold <= state_traj[1, final_step] <= rest_threshold):
                converged_count += 1

    return 100 * converged_count / total_count

def plot_sweep_with_floquet(x_values, percent_converged, floquet_vals, xlabel, title):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(x_values, percent_converged, 'o-', color='tab:blue', label="Percent converged")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Percent Converged", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    floquet_plot_vals = [f if f is not None else np.nan for f in floquet_vals]
    ax2 = ax1.twinx()
    ax2.plot(x_values, floquet_plot_vals, 's--', color='tab:orange', label="Floquet multiplier")
    ax2.axhline(1, color='gray', linestyle=':')
    ax2.axhline(-1, color='gray', linestyle=':')
    ax2.set_ylabel("Floquet Multiplier (lambda)", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
