import numpy as np
from numba import njit


@njit
def _step_jit(x, y, cos_t, sin_t, action, max_v, max_omega, dt,
              goal_threshold_pos, goal_threshold_angle):
    v     = min(max(action[0], -max_v), max_v)
    omega = min(max(action[1], -max_omega), max_omega)

    x += cos_t * v * dt
    y += sin_t * v * dt

    theta = np.arctan2(sin_t, cos_t) + omega * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    dist          = np.sqrt(x * x + y * y)
    angle_to_goal = np.arctan2(-y, -x) - theta
    angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi

    done = dist < goal_threshold_pos and abs(theta) < goal_threshold_angle
    return x, y, cos_t, sin_t, dist, np.cos(angle_to_goal), np.sin(angle_to_goal), done


class DiffDriveEnv:
    def __init__(self, dt=0.05, max_v=1.0, max_omega=2.0,
                 goal_threshold_pos=0.05, goal_threshold_angle=0.1):
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.goal_threshold_pos = goal_threshold_pos
        self.goal_threshold_angle = goal_threshold_angle
        self.x = 0.0
        self.y = 0.0
        self.cos_theta = 1.0
        self.sin_theta = 0.0

    def reset(self):
        self.x = np.random.uniform(-3, 3)
        self.y = np.random.uniform(-3, 3)
        theta = np.random.uniform(-np.pi, np.pi)
        self.cos_theta = np.cos(theta)
        self.sin_theta = np.sin(theta)
        return self._get_obs()

    def step(self, action):
        self.x, self.y, self.cos_theta, self.sin_theta, dist, cos_a, sin_a, done = _step_jit(
            self.x, self.y, self.cos_theta, self.sin_theta, action,
            self.max_v, self.max_omega, self.dt,
            self.goal_threshold_pos, self.goal_threshold_angle,
        )
        return np.array([dist, cos_a, sin_a], dtype=np.float64), done

    def get_state(self):
        return np.array([self.x, self.y, self.cos_theta, self.sin_theta])

    def _get_obs(self):
        dist = np.sqrt(self.x ** 2 + self.y ** 2)
        angle_to_goal = np.arctan2(-self.y, -self.x) - np.arctan2(self.sin_theta, self.cos_theta)
        angle_to_goal = self._wrap_angle(angle_to_goal)
        return np.array([dist, np.cos(angle_to_goal), np.sin(angle_to_goal)])

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
