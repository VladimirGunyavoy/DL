import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sim.env import DiffDriveEnv
from sim.expert import ExpertController

N_EPISODES = 1000
MAX_STEPS = 500


def main():
    env = DiffDriveEnv()
    expert = ExpertController()

    observations = []
    actions = []
    success_count = 0
    episode_lengths = []

    for episode in range(N_EPISODES):
        obs = env.reset()
        ep_obs = []
        ep_actions = []

        for step in range(MAX_STEPS):
            state = env.get_state()
            action = expert.act(obs, state)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            obs, done = env.step(action)
            if done:
                success_count += 1
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(MAX_STEPS)

        observations.extend(ep_obs)
        actions.extend(ep_actions)

    observations = np.array(observations)
    actions = np.array(actions)

    os.makedirs('data', exist_ok=True)
    np.save('data/observations.npy', observations)
    np.save('data/actions.npy', actions)

    print(f"Total samples: {len(observations)}")
    print(f"Success rate: {success_count}/{N_EPISODES} ({100*success_count/N_EPISODES:.1f}%)")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")


if __name__ == '__main__':
    main()
