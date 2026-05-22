import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class SequenceDataset(Dataset):
    def __init__(self, h5_path, context_len=20, samples_per_traj=20):
        self.context_len = context_len

        with h5py.File(h5_path, 'r') as f:
            n = int(f['metadata']['n_trajectories'][()])
            trajs = []
            for i in range(n):
                g = f[f'trajectory_{i}']
                trajs.append({
                    'obs':       g['obs'][:].astype(np.float32),
                    'actions':   g['actions'][:].astype(np.float32),
                    'rtg':       g['rtg'][:].astype(np.float32),
                    'timesteps': g['timesteps'][:].astype(np.int64),
                })

        all_obs = np.concatenate([t['obs']     for t in trajs])
        all_act = np.concatenate([t['actions'] for t in trajs])
        all_rtg = np.concatenate([t['rtg']     for t in trajs])

        self.obs_mean  = all_obs.mean(0).astype(np.float32)
        self.obs_std   = (all_obs.std(0) + 1e-8).astype(np.float32)
        self.act_mean  = all_act.mean(0).astype(np.float32)
        self.act_std   = (all_act.std(0) + 1e-8).astype(np.float32)
        self.rtg_scale = float(np.abs(all_rtg).max())

        for t in trajs:
            t['obs']     = (t['obs']     - self.obs_mean) / self.obs_std
            t['actions'] = (t['actions'] - self.act_mean) / self.act_std
            t['rtg']     = (t['rtg'] / self.rtg_scale)[:, None]

        # Pre-generate samples → big numpy cache, DataLoader просто индексирует
        K        = context_len
        obs_dim  = trajs[0]['obs'].shape[1]
        act_dim  = trajs[0]['actions'].shape[1]
        N        = len(trajs) * samples_per_traj

        cache_obs = np.zeros((N, K, obs_dim), dtype=np.float32)
        cache_act = np.zeros((N, K, act_dim), dtype=np.float32)
        cache_rtg = np.zeros((N, K, 1),       dtype=np.float32)
        cache_ts  = np.zeros((N, K),           dtype=np.int64)

        idx = 0
        for t in trajs:
            T = len(t['obs'])
            for _ in range(samples_per_traj):
                start  = np.random.randint(0, T)
                end    = min(start + K, T)
                length = end - start
                pad    = K - length

                obs = t['obs'][start:end]
                act = t['actions'][start:end]
                rtg = t['rtg'][start:end]
                ts  = t['timesteps'][start:end]

                if pad > 0:
                    obs = np.concatenate([np.zeros((pad, obs_dim), dtype=np.float32), obs])
                    act = np.concatenate([np.zeros((pad, act_dim), dtype=np.float32), act])
                    rtg = np.concatenate([np.zeros((pad, 1),       dtype=np.float32), rtg])
                    ts  = np.concatenate([np.zeros(pad,            dtype=np.int64),   ts])

                cache_obs[idx] = obs
                cache_act[idx] = act
                cache_rtg[idx] = rtg
                cache_ts[idx]  = ts
                idx += 1

        self.cache_obs = cache_obs
        self.cache_act = cache_act
        self.cache_rtg = cache_rtg
        self.cache_ts  = cache_ts

    def __len__(self):
        return len(self.cache_obs)

    def __getitem__(self, idx):
        return {
            'obs':       torch.from_numpy(self.cache_obs[idx]),
            'actions':   torch.from_numpy(self.cache_act[idx]),
            'rtg':       torch.from_numpy(self.cache_rtg[idx]),
            'timesteps': torch.from_numpy(self.cache_ts[idx]),
        }
