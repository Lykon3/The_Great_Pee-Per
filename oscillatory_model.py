import numpy as np

def oscillatory_collapse(t, A, k, t_c, B, gamma, omega, phi, offset):
    """
    Oscillatory collapse model:
    f(t) = (A / (1 + exp(-k(t - t_c)))) * (1 + B * exp(-gamma(t - t_c)) * cos(omega(t - t_c) + phi)) + offset
    """
    sigmoid = A / (1 + np.exp(-k * (t - t_c)))
    oscillation = 1 + B * np.exp(-gamma * (t - t_c)) * np.cos(omega * (t - t_c) + phi)
    return sigmoid * oscillation + offset
