import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sim.env import DiffDriveEnv
from sim.expert import ExpertController

N_EPISODES = 5
MAX_STEPS = 500
LOG_EVERY = 20


def main():
    env = DiffDriveEnv()
    expert = ExpertController()

    for episode in range(N_EPISODES):
        obs = env.reset()
        print(f"\n=== Episode {episode + 1} ===")
        state = env.get_state()
        theta0 = np.arctan2(state[3], state[2])
        print(f"Start: x={state[0]:.2f}, y={state[1]:.2f}, theta={theta0:.2f}")

        done = False
        for step in range(MAX_STEPS):
            state = env.get_state()
            action = expert.act(obs, state)

            if step % LOG_EVERY == 0:
                theta = np.arctan2(state[3], state[2])
                print(f"  step={step:3d}  x={state[0]:6.2f}  y={state[1]:6.2f}  "
                      f"theta={theta:5.2f}  dist={obs[0]:.3f}  "
                      f"v={action[0]:.2f}  omega={action[1]:.2f}")

            obs, done = env.step(action)
            if done:
                break

        state = env.get_state()
        result = "DONE" if done else "FAILED"
        theta_f = np.arctan2(state[3], state[2])
        print(f"  {result} at step {step + 1}: "
              f"x={state[0]:.3f}, y={state[1]:.3f}, theta={theta_f:.3f}")


if __name__ == '__main__':
    main()
