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
timestep = 1e-3
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
    local_params = dict(params)
    impact_angle = alpha + gamma
    local_state = np.array(state, dtype=float)

    for step in range(round(sim_time/timestep)):
        time = step * timestep
        local_params["ankle_torque"] = compute_ankle_torque(local_state, params) 
        local_state = local_state + timestep * model.dynamics(time, local_state, local_params)
        if abs(local_state[0]) >= impact_angle:
            return False
    return abs(local_state[0]) <= tolerance[0] and abs(local_state[1]) <= tolerance[1]

def compute_roa(params, timestep, sim_time, tolerance):
    roa = []
    failed_roa = []
    for angle in np.linspace(-tolerance[0], tolerance[0], 41):
        for angular_velocity in np.linspace(-tolerance[1], tolerance[1], 41):
            state = np.array([angle, angular_velocity])
            if ankle_balance_convergence(state, params, timestep, sim_time, tolerance):
                roa.append(state)
            else:
                failed_roa.append(state)
    return np.array(roa), np.array(failed_roa)

def plot_roa(roa, failed_roa, save_path=None):
    roa = np.asarray(roa).reshape(-1, 2)
    failed_roa = np.asarray(failed_roa).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(failed_roa[:, 0], failed_roa[:, 1], color="tab:red", marker="s", label="Falls")
    ax.scatter(roa[:, 0], roa[:, 1], color="tab:green", marker="s", label="Converges")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_title("Ankle-balance controller: region of attraction")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax

def advance_to_next_step(angular_velocity, alpha, params, timestep, max_time=5.0):
    local_params = dict(params)
    local_params["angle_of_attack"] = alpha
    local_params["ankle_torque"] = 0.0
    local_state = np.array([0.0, angular_velocity])
    impacted = False

    for step in range(round(max_time / timestep)):
        time = step * timestep
        next_state = local_state + timestep * model.dynamics(time, local_state, local_params)

        if not impacted and model.event_guard(local_state, next_state, local_params):
            impact_angle = local_params["angle_of_attack"] + local_params["incline"]
            fraction = (impact_angle - local_state[0]) / (next_state[0] - local_state[0])
            impact_state = local_state + fraction * (next_state - local_state)  # state exactly at impact
            next_state = model.event_dynamics(impact_state, local_params)
            impacted = True
        elif impacted and local_state[0] < 0 and next_state[0] >= 0:
            fraction = -local_state[0] / (next_state[0] - local_state[0])
            return local_state[1] + fraction * (next_state[1] - local_state[1])

        if next_state[1] <= 0:
            return np.nan
        local_state = next_state

    return np.nan

def find_standing_threshold(params, timestep, sim_time, tolerance, precision=1e-3):
    low = 0.0   # theta_dot = 0 on the section always stands
    high = 1.0  # well above the standing range
    if not ankle_balance_convergence(np.array([0.0, low]), params, timestep, sim_time, tolerance):
        raise ValueError("theta_dot = 0 doesn't stand; check the ankle controller")
    if ankle_balance_convergence(np.array([0.0, high]), params, timestep, sim_time, tolerance):
        raise ValueError("theta_dot = 1.0 stands; increase high")

    while high - low > precision:
        middle = (low + high) / 2
        if ankle_balance_convergence(np.array([0.0, middle]), params, timestep, sim_time, tolerance):
            low = middle   # middle stands, so the threshold is above it
        else:
            high = middle  # middle falls, so the threshold is below it
    return low             # fastest speed confirmed to stand

def analytic_next_section(theta_dot, angle_of_attack, params):
    gravity = params["gravity"]
    length = params["length"]
    gamma = params["incline"]
    alpha = angle_of_attack

    before_impact_squared = theta_dot ** 2 + 2 * (gravity / length) * (1 - np.cos(alpha + gamma))
    after_impact = np.cos(2 * alpha) * np.sqrt(before_impact_squared)
    next_squared = after_impact ** 2 - 2 * (gravity / length) * (1 - np.cos(alpha - gamma))

    if next_squared <= 0:
        return None  # stalls before reaching theta = 0
    return np.sqrt(next_squared)

def build_step_table(params, timestep, n_velocity=30, n_alpha=10):
    gravity = params["gravity"]
    length = params["length"]

    max_angular_velocity = np.sqrt(2 * gravity / length)          # Froude number of 2
    velocities = np.linspace(0, max_angular_velocity, n_velocity)  # state grid
    alphas = np.linspace(np.pi / 8, np.pi / 7, n_alpha)            # control grid
    table = np.full((n_velocity, n_alpha), np.nan)                 # rows = states, columns = actions

    for i, angular_velocity in enumerate(velocities):
        for j, alpha in enumerate(alphas):
            table[i, j] = advance_to_next_step(angular_velocity, alpha, params, timestep)
    return velocities, alphas, table


def find_standing_states(velocities, threshold):
    return velocities <= threshold  # one comparison per grid state


def plot_step_table(velocities, alphas, table, save_path=None):
    d_alpha = alphas[1] - alphas[0]
    d_velocity = velocities[1] - velocities[0]
    extent = [alphas[0] - d_alpha / 2, alphas[-1] + d_alpha / 2,
              velocities[0] - d_velocity / 2, velocities[-1] + d_velocity / 2]

    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("lightgray")  # nan cells (stalled steps) show as gray

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(table, origin="lower", extent=extent, aspect="auto",
                      cmap=colormap, interpolation="nearest")
    fig.colorbar(image, ax=ax, label=r"$\dot{\theta}_{k+1}$ (rad/s)")
    ax.set_xlabel(r"$\alpha$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}_k$ (rad/s)")
    ax.set_title("Step-to-step table: next section velocity")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax

def nearest_index(velocities, value):
    return int(np.argmin(np.abs(velocities - value)))  # grid point closest to value


def find_one_step_cells(table, threshold):
    return table <= threshold       # nan (stalled) cells compare as False


def compute_steps_to_stand(velocities, alphas, table, standing, one_step_cells, threshold, max_steps=50):
    n_velocity = len(velocities)
    steps_to_stand = np.full(n_velocity, np.inf)  # inf = can't reach standstill (yet)
    best_alpha = np.full(n_velocity, np.nan)      # nan = no step needed, or no policy
    steps_to_stand[standing] = 0                  # layer 0: already standing

    # Layer 1: some action lands exactly in the standing range
    for i in range(n_velocity):
        if steps_to_stand[i] == np.inf:
            good_actions = np.where(one_step_cells[i])[0]
            if len(good_actions) > 0:
                steps_to_stand[i] = 1
                best_alpha[i] = alphas[good_actions[len(good_actions) // 2]]  # middle of the working range

    # Layers 2, 3, ...: some action lands on a grid state from the previous layer
    for n in range(2, max_steps + 1):
        found_new_state = False
        for i in range(n_velocity):
            if steps_to_stand[i] != np.inf:
                continue
            best_error = np.inf
            for j, alpha in enumerate(alphas):
                next_velocity = table[i, j]
                if np.isnan(next_velocity):
                    continue                                   # this step stalls
                k = nearest_index(velocities, next_velocity)
                rounding_error = abs(next_velocity - velocities[k])
                if steps_to_stand[k] == n - 1 and rounding_error < best_error:
                    best_error = rounding_error                # tie-break: closest to a grid point
                    best_alpha[i] = alpha
            if best_error < np.inf:
                steps_to_stand[i] = n
                found_new_state = True
        if not found_new_state:
            break                                              # no new layer, so no later ones either
    return steps_to_stand, best_alpha


def compute_action_steps(velocities, table, steps_to_stand, one_step_cells):
    action_steps = np.full(table.shape, np.nan)  # nan = this step stalls
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if one_step_cells[i, j]:
                action_steps[i, j] = 1
            elif not np.isnan(table[i, j]):
                k = nearest_index(velocities, table[i, j])
                # Rounding onto a standing state doesn't count; layer 1 is checked exactly
                action_steps[i, j] = 1 + steps_to_stand[k] if steps_to_stand[k] > 0 else np.inf
    return action_steps


def rollout(angular_velocity, velocities, best_alpha, params, timestep, threshold, max_steps=20):
    current_velocity = angular_velocity
    for step_count in range(max_steps + 1):
        if current_velocity <= threshold:
            return step_count                                  # standing
        alpha = best_alpha[nearest_index(velocities, current_velocity)]
        if np.isnan(alpha):
            return np.inf                                      # no policy for this state
        current_velocity = advance_to_next_step(current_velocity, alpha, params, timestep)
        if np.isnan(current_velocity):
            return np.inf                                      # the step stalled
    return np.inf

def plot_steps_to_stand(velocities, alphas, action_steps, steps_to_stand, best_alpha, save_path=None):
    d_alpha = alphas[1] - alphas[0]
    d_velocity = velocities[1] - velocities[0]
    extent = [alphas[0] - d_alpha / 2, alphas[-1] + d_alpha / 2,
              velocities[0] - d_velocity / 2, velocities[-1] + d_velocity / 2]

    finite_steps = np.where(np.isfinite(action_steps), action_steps, np.nan)
    unreachable_cells = np.where(np.isinf(action_steps), 1.0, np.nan)
    max_steps = int(np.nanmax(finite_steps))

    step_colormap = plt.get_cmap("viridis", max_steps).copy()
    step_colormap.set_bad("lightgray")

    fig, (ax_table, ax_steps) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: state-action table colored by total steps to standstill
    image = ax_table.imshow(finite_steps, origin="lower", extent=extent, aspect="auto",
                            cmap=step_colormap, vmin=0.5, vmax=max_steps + 0.5,
                            interpolation="nearest")
    ax_table.imshow(unreachable_cells, origin="lower", extent=extent, aspect="auto",
                    cmap="binary", vmin=0, vmax=1, interpolation="nearest")  # 1 = black, nan = transparent
    colorbar = fig.colorbar(image, ax=ax_table, ticks=range(1, max_steps + 1))
    colorbar.set_label("Steps to standstill with this action")
    has_policy = ~np.isnan(best_alpha)
    ax_table.plot(best_alpha[has_policy], velocities[has_policy], "o",
                  color="tab:red", markersize=4, label="Chosen action")
    ax_table.plot([], [], "s", color="lightgray", markersize=10, label="Step stalls")  # legend-only entry
    ax_table.plot([], [], "s", color="black", markersize=10, label="Lands on unreachable state")
    ax_table.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax_table.set_xlabel(r"$\alpha$ (rad)")
    ax_table.set_ylabel(r"$\dot{\theta}_k$ (rad/s)")
    ax_table.set_title("State-action table")

    # Right: fewest steps to standstill from each state
    reachable = np.isfinite(steps_to_stand)
    ax_steps.scatter(velocities[reachable], steps_to_stand[reachable],
                     color="tab:green", label="Reachable")
    ax_steps.scatter(velocities[~reachable], np.full(np.sum(~reachable), -0.5),
                     marker="x", color="tab:red", label="Unreachable")
    ax_steps.set_yticks(range(0, int(np.max(steps_to_stand[reachable])) + 1))
    ax_steps.set_xlabel(r"$\dot{\theta}_k$ (rad/s)")
    ax_steps.set_ylabel("Steps to standstill")
    ax_steps.set_title("Fewest steps to standstill")
    ax_steps.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax_table, ax_steps

def evaluate_grid(n_velocity, n_alpha, test_velocities, params, timestep, threshold,
                  rollout_timestep=None, fixed_velocity=3.0):
    if rollout_timestep is None:
        rollout_timestep = timestep  # test at the same timestep the table was built with

    # Build the table and policy for this grid
    velocities, alphas, table = build_step_table(params, timestep, n_velocity, n_alpha)
    standing = find_standing_states(velocities, threshold)
    one_step_cells = find_one_step_cells(table, threshold)
    steps_to_stand, best_alpha = compute_steps_to_stand(velocities, alphas, table, standing, one_step_cells)

    # What the table promises vs what real simulations do, from off-grid starts
    predicted = np.array([steps_to_stand[nearest_index(velocities, v)] for v in test_velocities])
    actual = np.array([rollout(v, velocities, best_alpha, params, rollout_timestep, threshold)
                       for v in test_velocities])

    promised = np.isfinite(predicted)  # only score starts the table says are reachable
    if np.any(promised):
        reaches_rate = np.mean(np.isfinite(actual[promised]))
        agreement_rate = np.mean(actual[promised] == predicted[promised])
    else:
        reaches_rate = 0.0
        agreement_rate = 0.0

    # Classmate-style single example for the report
    fixed_predicted = steps_to_stand[nearest_index(velocities, fixed_velocity)]
    fixed_actual = rollout(fixed_velocity, velocities, best_alpha, params, rollout_timestep, threshold)

    return {
        "n_velocity": n_velocity,
        "n_alpha": n_alpha,
        "velocity_spacing": velocities[1] - velocities[0],
        "alpha_spacing": alphas[1] - alphas[0],
        "n_promised": int(np.sum(promised)),
        "reaches_rate": reaches_rate,
        "agreement_rate": agreement_rate,
        "fixed_predicted": fixed_predicted,
        "fixed_actual": fixed_actual,
    }


def passes_criterion(result, min_reaches=0.80, min_agreement=0.80):
    return result["reaches_rate"] >= min_reaches and result["agreement_rate"] >= min_agreement


def print_result(result):
    print(f"n_velocity={result['n_velocity']:4d}  n_alpha={result['n_alpha']:4d}  "
          f"reaches={100 * result['reaches_rate']:5.1f}%  "
          f"as predicted={100 * result['agreement_rate']:5.1f}%  "
          f"passes={passes_criterion(result)}")


def search_resolution(evaluate, candidates, relative_gap=0.1):
    results = {}
    for n in candidates:                       # coarse-to-fine doubling
        results[n] = evaluate(n)
        print_result(results[n])

    passing = [passes_criterion(results[n]) for n in candidates]
    chosen = None
    for index in range(len(candidates)):
        if all(passing[index:]):               # this one and every finer one pass
            chosen = index
            break

    if chosen is None:
        print("The finest candidate fails; add finer candidates.")
        return None, results
    if chosen == 0:
        print("The coarsest candidate already passes; add coarser candidates.")
        return candidates[0], results

    low = candidates[chosen - 1]               # fails
    high = candidates[chosen]                  # passes
    while high - low > relative_gap * high:    # bisect until the gap is within 10%
        middle = (low + high) // 2
        results[middle] = evaluate(middle)
        print_result(results[middle])
        if passes_criterion(results[middle]):
            high = middle
        else:
            low = middle
    return high, results


def results_to_markdown(results):
    lines = ["| n_velocity | n_alpha | velocity spacing (rad/s) | alpha spacing (rad) "
             "| reaches RoA | steps as predicted | passes | steps from 3.0 (predicted / actual) |",
             "|---|---|---|---|---|---|---|---|"]
    for n in sorted(results):
        r = results[n]
        lines.append(f"| {r['n_velocity']} | {r['n_alpha']} | {r['velocity_spacing']:.3f} "
                     f"| {r['alpha_spacing']:.5f} | {100 * r['reaches_rate']:.1f}% "
                     f"| {100 * r['agreement_rate']:.1f}% | {'yes' if passes_criterion(r) else 'no'} "
                     f"| {r['fixed_predicted']:g} / {r['fixed_actual']:g} |")
    return "\n".join(lines)


def plot_resolution_search(results, xlabel, title, save_path=None):
    grid_sizes = sorted(results)
    reaches = [100 * results[n]["reaches_rate"] for n in grid_sizes]
    agreement = [100 * results[n]["agreement_rate"] for n in grid_sizes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid_sizes, reaches, "o-", color="tab:green", label="Reaches RoA")
    ax.plot(grid_sizes, agreement, "s-", color="tab:blue", label="Steps as predicted")
    ax.axhline(98, color="tab:green", linestyle="--", label="95% threshold")
    ax.axhline(95, color="tab:blue", linestyle="--", label="95% threshold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test starts (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax

def compute_max_steps(velocities, alphas, table, standing, one_step_cells, threshold):
    n_velocity = len(velocities)
    max_steps = np.full(n_velocity, -np.inf)     # -inf = can't reach standstill
    longest_alpha = np.full(n_velocity, np.nan)
    max_steps[standing] = 0
    target = threshold ** 2 / 2                  # middle of the standing window, in energy

    for i in range(n_velocity):                  # slow to fast: each step lands on a slower state
        if standing[i]:
            continue
        best_steps = -np.inf
        best_error = np.inf
        for j, alpha in enumerate(alphas):
            next_velocity = table[i, j]
            if np.isnan(next_velocity):
                continue                                         # this step stalls
            if one_step_cells[i, j]:
                steps = 1                                        # stops after this step
                error = abs(next_velocity ** 2 - target)         # prefer the safest final landing
            else:
                k = nearest_index(velocities, next_velocity)
                if k >= i or max_steps[k] < 1:
                    continue                                     # must land on a slower state that can still stop
                steps = 1 + max_steps[k]
                error = abs(next_velocity - velocities[k])       # prefer landing close to a grid point
            if steps > best_steps or (steps == best_steps and error < best_error):
                best_steps = steps
                best_error = error
                longest_alpha[i] = alpha
        max_steps[i] = best_steps
    return max_steps, longest_alpha


def make_walk(times, angles, angular_velocities, footstrike_times, n_steps, handover_time, status):
    return {
        "time": np.array(times),
        "angle": np.array(angles),
        "angular_velocity": np.array(angular_velocities),
        "footstrike_times": footstrike_times,
        "n_steps": n_steps,
        "handover_time": handover_time,
        "status": status,
    }


def simulate_walk(start_velocity, velocities, policy_alpha, params, timestep, sim_time,
                  tolerance, threshold, max_steps=30, max_step_time=5.0):
    times = [0.0]
    angles = [0.0]
    angular_velocities = [start_velocity]
    footstrike_times = []
    time = 0.0
    current_velocity = start_velocity
    n_steps = 0

    # Walking phase: one passive step at a time, alpha chosen from the policy
    while current_velocity > threshold:
        if n_steps >= max_steps:
            return make_walk(times, angles, angular_velocities, footstrike_times, n_steps, time, "too many steps")
        alpha = policy_alpha[nearest_index(velocities, current_velocity)]
        if np.isnan(alpha):
            return make_walk(times, angles, angular_velocities, footstrike_times, n_steps, time, "no policy")

        local_params = dict(params)
        local_params["angle_of_attack"] = alpha
        local_params["ankle_torque"] = 0.0
        local_state = np.array([0.0, current_velocity])
        impacted = False
        crossed = False

        for step in range(round(max_step_time / timestep)):
            next_state = local_state + timestep * model.dynamics(time, local_state, local_params)

            if not impacted and model.event_guard(local_state, next_state, local_params):
                impact_angle = alpha + params["incline"]
                fraction = (impact_angle - local_state[0]) / (next_state[0] - local_state[0])
                impact_state = local_state + fraction * (next_state - local_state)
                impact_time = time + fraction * timestep
                times.append(impact_time)                         # just before impact
                angles.append(impact_state[0])
                angular_velocities.append(impact_state[1])
                footstrike_times.append(impact_time)
                times.append(np.nan)                              # break the line at the leg swap
                angles.append(np.nan)
                angular_velocities.append(np.nan)
                next_state = model.event_dynamics(impact_state, local_params)
                impacted = True
            elif impacted and local_state[0] < 0 and next_state[0] >= 0:
                fraction = -local_state[0] / (next_state[0] - local_state[0])
                current_velocity = local_state[1] + fraction * (next_state[1] - local_state[1])
                time = time + fraction * timestep
                times.append(time)                                # back on the section
                angles.append(0.0)
                angular_velocities.append(current_velocity)
                crossed = True
                break

            if next_state[1] <= 0:
                return make_walk(times, angles, angular_velocities, footstrike_times, n_steps, time, "stalled")

            time = time + timestep
            local_state = next_state
            times.append(time)
            angles.append(local_state[0])
            angular_velocities.append(local_state[1])

        if not crossed:
            return make_walk(times, angles, angular_velocities, footstrike_times, n_steps, time, "timed out")
        n_steps += 1

    # Standing phase: hand over to the ankle controller on the section
    handover_time = time
    local_params = dict(params)
    local_state = np.array([0.0, current_velocity])
    impact_angle = params["angle_of_attack"] + params["incline"]
    for step in range(round(sim_time / timestep)):
        local_params["ankle_torque"] = compute_ankle_torque(local_state, params)
        local_state = local_state + timestep * model.dynamics(time, local_state, local_params)
        time = time + timestep
        times.append(time)
        angles.append(local_state[0])
        angular_velocities.append(local_state[1])
        if abs(local_state[0]) >= impact_angle:
            return make_walk(times, angles, angular_velocities, footstrike_times, n_steps,
                             handover_time, "fell while standing")

    stood = abs(local_state[0]) <= tolerance[0] and abs(local_state[1]) <= tolerance[1]
    status = "standing" if stood else "didn't settle"
    return make_walk(times, angles, angular_velocities, footstrike_times, n_steps, handover_time, status)


def plot_walks(walks, labels, colors, save_path=None):
    fig, (ax_angle, ax_velocity, ax_phase) = plt.subplots(1, 3, figsize=(17, 5))
    for walk, label, color in zip(walks, labels, colors):
        full_label = f"{label}: {walk['n_steps']} steps, {walk['status']}"
        ax_angle.plot(walk["time"], walk["angle"], color=color, label=full_label)
        ax_velocity.plot(walk["time"], walk["angular_velocity"], color=color, label=full_label)
        ax_phase.plot(walk["angle"], walk["angular_velocity"], color=color, label=full_label)
        for footstrike_time in walk["footstrike_times"]:
            ax_angle.axvline(footstrike_time, color=color, linestyle=":", linewidth=0.8)
        ax_angle.axvline(walk["handover_time"], color=color, linestyle="--", linewidth=1.2)
        ax_velocity.axvline(walk["handover_time"], color=color, linestyle="--", linewidth=1.2)

    ax_angle.set_xlabel("Time (s)")
    ax_angle.set_ylabel(r"$\theta$ (rad)")
    ax_angle.set_title("Angle (dotted: footstrikes, dashed: ankle takes over)")
    ax_velocity.set_xlabel("Time (s)")
    ax_velocity.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax_velocity.set_title("Angular velocity")
    ax_phase.set_xlabel(r"$\theta$ (rad)")
    ax_phase.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax_phase.set_title("Phase portrait")
    ax_phase.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_steps_vs_initial_condition(velocities, steps_to_stand, max_steps, save_path=None):
    fewest = np.where(np.isfinite(steps_to_stand), steps_to_stand, np.nan)
    most = np.where(np.isfinite(max_steps), max_steps, np.nan)
    unreachable = ~np.isfinite(steps_to_stand)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(velocities, most, where="mid", color="tab:orange", label="Most steps (longest walk)")
    ax.step(velocities, fewest, where="mid", color="tab:green", label="Fewest steps (fastest stop)")
    ax.scatter(velocities[unreachable], np.full(np.sum(unreachable), -0.5),
               marker="x", color="tab:red", s=12, label="Can't reach standstill")
    ax.set_xlabel(r"Initial $\dot{\theta}$ on the section (rad/s)")
    ax.set_ylabel("Steps to standstill")
    ax.set_title("Steps to standstill from each initial condition")
    ax.set_yticks(range(0, int(np.nanmax(most)) + 1))
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_phase_portrait(walks, labels, colors, linestyles, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for walk, label, color, linestyle in zip(walks, labels, colors, linestyles):
        keep = ~np.isnan(walk["angle"])  # drop the line breaks so each impact is drawn as a jump
        ax.plot(walk["angle"][keep], walk["angular_velocity"][keep], color=color,
                linestyle=linestyle, label=f"{label} ({walk['n_steps']} steps)")

    start = walks[0]
    ax.scatter(start["angle"][0], start["angular_velocity"][0], color="black",
               zorder=3, label="Initial state")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_title("State-Space Trajectory")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig



output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

tolerance = find_tolerance(tolerance, params)
threshold = find_standing_threshold(params, timestep, sim_time, tolerance)

n_velocity_final = 360   # your chosen grid
n_alpha_final = 100
velocities, alphas, table = build_step_table(params, timestep, n_velocity_final, n_alpha_final)
standing = find_standing_states(velocities, threshold)
one_step_cells = find_one_step_cells(table, threshold)
steps_to_stand, best_alpha = compute_steps_to_stand(velocities, alphas, table, standing,
                                                    one_step_cells, threshold)
max_steps, longest_alpha = compute_max_steps(velocities, alphas, table, standing,
                                             one_step_cells, threshold)

start_velocity = 3.0
i = nearest_index(velocities, start_velocity)
print(f"Start {start_velocity}: table predicts fewest={steps_to_stand[i]}, most={max_steps[i]}")

fewest_walk = simulate_walk(start_velocity, velocities, best_alpha, params, timestep,
                            sim_time, tolerance, threshold)
longest_walk = simulate_walk(start_velocity, velocities, longest_alpha, params, timestep,
                             sim_time, tolerance, threshold)
print(f"Fewest-steps walk: {fewest_walk['n_steps']} steps, {fewest_walk['status']}")
print(f"Most-steps walk:   {longest_walk['n_steps']} steps, {longest_walk['status']}")

plot_walks([fewest_walk, longest_walk], ["Fewest steps", "Most steps"],
           ["tab:green", "tab:orange"], save_path=output / "trajectories.png")
plot_steps_vs_initial_condition(velocities, steps_to_stand, max_steps,
                                save_path=output / "steps_vs_initial_condition.png")
plt.show()

plot_phase_portrait([fewest_walk, longest_walk], ["fastest", "slowest"],
                    ["tab:blue", "tab:orange"], ["-", "--"],
                    save_path=output / "phase_portrait.png")
plt.show()


# #ROA computation and plotting
# tolerance = find_tolerance(tolerance, params)
# roa, failed_roa = compute_roa(params, timestep, sim_time, tolerance)
# plot_roa(roa, failed_roa, save_path=Path("roa.png"))
# plt.show()

# #Step table computation and plotting
# tolerance = find_tolerance(tolerance, params)
# velocities, alphas, table = build_step_table(params, timestep)
# standing = find_standing_states(velocities, params, timestep, sim_time, tolerance)
# plot_step_table(velocities, alphas, table, save_path=Path("step_table.png"))
# plt.show()

# #Step-to-step convergence computation and plotting
# tolerance = find_tolerance(tolerance, params)
# threshold = find_standing_threshold(params, timestep, sim_time, tolerance)
# velocities, alphas, table = build_step_table(params, timestep, n_velocity=120, n_alpha=100)
# standing = find_standing_states(velocities, threshold)
# one_step_cells = find_one_step_cells(table, threshold)
# steps_to_stand, best_alpha = compute_steps_to_stand(velocities, alphas, table, standing, one_step_cells)
# action_steps = compute_action_steps(velocities, table, steps_to_stand, one_step_cells)
# plot_steps_to_stand(velocities, alphas, action_steps, steps_to_stand, best_alpha,
#                     save_path=Path("steps_to_stand.png"))
# plt.show()

# # Resolution search for velocity and alpha grid sizes
# output = Path("output/assignment_2")
# output.mkdir(parents=True, exist_ok=True)

# tolerance = find_tolerance(tolerance, params)
# threshold = find_standing_threshold(params, timestep, sim_time, tolerance)

# max_angular_velocity = np.sqrt(2 * params["gravity"] / params["length"])
# rng = np.random.default_rng(seed=0)
# test_velocities = rng.uniform(0, max_angular_velocity, 50)   # 50 test starts instead of 200

# # Search 1: velocity resolution, alpha held at 60
# def evaluate_velocity(n):
#     return evaluate_grid(n, 60, test_velocities, params, timestep, threshold)

# best_n_velocity, velocity_results = search_resolution(evaluate_velocity, [120, 240, 360, 480],
#                                                       relative_gap=0.25)
# if best_n_velocity is None:
#     raise SystemExit("No velocity grid passed; add finer candidates.")

# # Search 2: alpha resolution at the chosen velocity grid
# def evaluate_alpha(n):
#     return evaluate_grid(best_n_velocity, n, test_velocities, params, timestep, threshold)

# best_n_alpha, alpha_results = search_resolution(evaluate_alpha, [20, 40, 60], relative_gap=0.25)
# if best_n_alpha is None:
#     raise SystemExit("No alpha grid passed; add finer candidates.")

# print(f"\nChosen grid: n_velocity={best_n_velocity}, n_alpha={best_n_alpha}")

# velocity_table = results_to_markdown(velocity_results)
# alpha_table = results_to_markdown(alpha_results)
# print("\n" + velocity_table + "\n\n" + alpha_table)
# (output / "resolution_tables.md").write_text(velocity_table + "\n\n" + alpha_table + "\n")

# plot_resolution_search(velocity_results, "Velocity grid points (n_velocity)",
#                        "Velocity resolution search (n_alpha = 60)",
#                        save_path=output / "velocity_resolution.png")
# plot_resolution_search(alpha_results, "Alpha grid points (n_alpha)",
#                        f"Alpha resolution search (n_velocity = {best_n_velocity})",
#                        save_path=output / "alpha_resolution.png")
# plt.show


"""
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

"""