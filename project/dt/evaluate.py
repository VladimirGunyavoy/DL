import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm.auto import tqdm
from sim.env import DiffDriveEnv


def evaluate_dt(model, dataset, device, n_episodes=200, desired_rtg=-150, max_steps=1000):
    model.eval()
    env = DiffDriveEnv()
    K   = model.context_len

    rtg_init = desired_rtg

    successes = 0
    lengths   = []

    obs_dim = dataset.obs_mean.shape[0]
    act_dim = dataset.act_mean.shape[0]

    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc='rollout', leave=False):
            obs_np = env.reset()

            obs_buf = np.zeros((K, obs_dim), dtype=np.float32)
            act_buf = np.zeros((K, act_dim), dtype=np.float32)
            rtg_buf = np.zeros((K, 1),       dtype=np.float32)
            ts_buf  = np.zeros(K,             dtype=np.int64)

            rtg_t = rtg_init
            done  = False

            for step in range(max_steps):
                obs_norm = (obs_np - dataset.obs_mean) / dataset.obs_std

                obs_buf = np.roll(obs_buf, -1, axis=0)
                act_buf = np.roll(act_buf, -1, axis=0)
                rtg_buf = np.roll(rtg_buf, -1, axis=0)
                ts_buf  = np.roll(ts_buf,  -1)

                obs_buf[-1] = obs_norm
                rtg_buf[-1] = rtg_t
                ts_buf[-1]  = min(step, 999)

                obs_t = torch.from_numpy(obs_buf).unsqueeze(0).to(device)
                act_t = torch.from_numpy(act_buf).unsqueeze(0).to(device)
                rtg_t_ = torch.from_numpy(rtg_buf).unsqueeze(0).to(device)
                ts_t  = torch.from_numpy(ts_buf).unsqueeze(0).to(device)

                pred = model(obs_t, act_t, rtg_t_, ts_t)   # (1, K, act_dim)
                action_norm = pred[0, -1].cpu().numpy()

                act_buf[-1] = action_norm
                action = action_norm * dataset.act_std + dataset.act_mean

                obs_np, done = env.step(action)
                rtg_t = rtg_t - (-1.0) / dataset.rtg_scale

                if done:
                    break

            successes += int(done)
            lengths.append(step + 1)

    model.train()
    return {
        'success_rate': successes / n_episodes,
        'mean_length':  float(np.mean(lengths)),
    }
