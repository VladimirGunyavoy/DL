# Decision Transformer for Differential-Drive Robot Navigation

**Skoltech Deep Learning Course — Project**

Imitation learning approach to robot navigation: a Decision Transformer trained on expert demonstrations steers a differential-drive robot from any random starting pose to the origin `(0, 0, θ=0)`.

---

## Problem

**Robot:** differential-drive (unicycle kinematics)

```
x_new     = x + cos(θ) · v · dt
y_new     = y + sin(θ) · v · dt
θ_new     = θ + ω · dt
```

**Goal:** reach `(x, y, θ) = (0, 0, 0)` from a random start in `[-3, 3]² × [-π, π]`.

| | Space |
|---|---|
| **Observation** | `[dist, cos(α), sin(α)]` — distance to goal and heading error (periodic encoding) |
| **Action** | `[v, ω]` — linear and angular velocity |
| **Done** | `dist < 0.05` and `\|θ\| < 0.1 rad` |

The observation uses `(cos α, sin α)` instead of raw angle α to avoid the discontinuity at ±π.

---

## Expert Controller

A geometric controller used to collect demonstrations. It does not use a finite-state machine — instead it computes actions directly from the position vector **r** = (x, y) and heading **p** = (cos θ, sin θ):

```python
cross  = r × p          # signed angle between r and p
rp     = r · p          # projection of r onto heading

v     = -k_v  * rp / |r|              # forward when facing goal
omega = -k_ω  * sign(rp) * cross / |r|  # steer toward goal
```

The sign of `rp` tells the controller whether to go forward or backward, so the robot always takes the shorter path. Near the goal (`dist < 0.015`) it switches to pure theta-alignment.

Expert success rate: **≈ 99%** on 1 000 random episodes.

---

## Decision Transformer

Standard DT architecture with the `[RTG, obs, act]` token interleaving.

| Hyperparameter | Value |
|---|---|
| Observation dim | 3 |
| Action dim | 2 |
| Context length K | 20 |
| Embedding dim | 64 |
| Transformer layers | 3 |
| Attention heads | 4 |
| Desired RTG | −150 |

The model is trained with supervised MSE loss on sequences sampled from recorded expert trajectories.

---

## Results

Models trained on datasets of different sizes, evaluated over 200 rollouts each.

| Dataset | Train loss (final) | Success Rate |
|---|---|---|
| 1 k episodes | 0.0343 | **100%** |
| 2 k episodes | 0.0221 | **100%** |
| 5 k episodes | 0.0151 | **100%** |
| 10 k episodes | 0.0124 | **100%** |

All four models reach 100% success rate by the end of training. Even 1k expert episodes is sufficient.

### Trajectories — Expert vs Decision Transformer

![Expert trajectories](figures/eval_trajectories.png)

### Action distribution — Expert vs DT

![Speed boxplot](figures/eval_speed_boxplot.png)

---

## Repository Structure

```
.
├── sim/
│   ├── env.py            # DiffDriveEnv — numba-accelerated kinematics
│   ├── expert.py         # Geometric expert controller (numba JIT)
│   ├── generate_data.py  # Collect expert trajectories → data/
│   └── test_expert.py    # Quick sanity check (5 episodes)
├── dt/
│   ├── model.py          # DecisionTransformer (nn.Module)
│   ├── dataset.py        # SequenceDataset — HDF5 + RTG preprocessing
│   └── evaluate.py       # Rollout evaluation
├── checkpoints/          # Saved model weights (*.pt, not tracked)
├── data/                 # Generated datasets (*.hdf5, not tracked)
├── figures/              # Plots
├── tests/
│   └── visualize_expert.ipynb   # Expert trajectory visualization
├── scripts/
│   └── monitor.py        # Live training monitor (reads results.json)
├── results.json          # Training history
└── environment.yml
```

---

## How to Run

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate dl_project
```

### 2. Generate expert dataset

```bash
python sim/generate_data.py
```

Saves trajectories to `data/` in HDF5 format.

### 3. Visualize expert trajectories

Open `tests/visualize_expert.ipynb` and run all cells.

### 4. Monitor training (while training runs)

```bash
python scripts/monitor.py
```

---

## Dependencies

- Python 3.10
- PyTorch
- NumPy
- Numba
- h5py
- matplotlib
- tqdm

See `environment.yml` for the full environment.
