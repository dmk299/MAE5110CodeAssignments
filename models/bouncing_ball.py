import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    velocity = state[1]
    stiffness = params["stiffness"]

    acceleration = -gravity

    if height <= 0:
        acceleration += stiffness * (-height) / mass

    state_derivative = np.array([velocity, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "mass": 1,  # mass of ball (kg)
        "stiffness": 100,  # stiffness of ball (kg/s^2)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    spring_energy = 0.5 * params["stiffness"] * np.maximum(0, -height) ** 2

    kinetic_energy = 0.5 * mass * velocity ** 2
    potential_energy = mass * gravity * height + spring_energy
    return kinetic_energy, potential_energy
