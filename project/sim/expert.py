import numpy as np
from numba import njit


@njit
def _act_jit(x, y, cos_t, sin_t, dist, k_v, k_omega, max_v, max_omega,
             final_phase_dist, noise_std):
    if dist < final_phase_dist:
        theta = np.arctan2(sin_t, cos_t)
        v     = min(max(k_v * dist,       0.0,      ), max_v)
        omega = min(max(-k_omega * theta, -max_omega), max_omega)
    else:
        r_norm = np.sqrt(x * x + y * y)

        cross = x * sin_t - y * cos_t
        rp    = x * cos_t + y * sin_t

        v_raw     = -rp / (r_norm + 1e-8)
        omega_raw = -np.sign(rp) * cross / (r_norm + 1e-8)

        v     = min(max(k_v * v_raw,         -max_v),     max_v)
        omega = min(max(k_omega * omega_raw, -max_omega), max_omega)

    if noise_std > 0.0:
        v     += np.random.normal(0.0, noise_std)
        omega += np.random.normal(0.0, noise_std)

    return v, omega


class ExpertController:
    def __init__(self, k_v=0.6, k_omega=1.0, max_v=1.0, max_omega=2.0,
                 final_phase_dist=0.015, noise_std=0.0):
        self.k_v = k_v
        self.k_omega = k_omega
        self.max_v = max_v
        self.max_omega = max_omega
        self.final_phase_dist = final_phase_dist
        self.noise_std = noise_std

    def act(self, obs, state):
        # state: [x, y, cos_theta, sin_theta]
        # obs:   [dist, cos_angle, sin_angle]
        v, omega = _act_jit(
            state[0], state[1], state[2], state[3], obs[0],
            self.k_v, self.k_omega, self.max_v, self.max_omega,
            self.final_phase_dist, self.noise_std,
        )
        return np.array([v, omega])
