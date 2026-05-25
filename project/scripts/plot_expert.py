"""
Generate figures/trajectories_expert.png
Run from project root: python scripts/plot_expert.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sim.env import DiffDriveEnv
from sim.expert import ExpertController

N = 25
MAX_STEPS = 1000

env    = DiffDriveEnv(goal_threshold_pos=0.2, goal_threshold_angle=0.3)
expert = ExpertController()
rng    = np.random.default_rng(42)

trajectories = []
for ep in range(N):
    env.x         = rng.uniform(-3, 3)
    env.y         = rng.uniform(-3, 3)
    theta         = rng.uniform(-np.pi, np.pi)
    env.cos_theta = np.cos(theta)
    env.sin_theta = np.sin(theta)
    obs  = env._get_obs()
    xs, ys, thetas = [], [], []
    done = False
    for _ in range(MAX_STEPS):
        state = env.get_state()
        xs.append(state[0]);  ys.append(state[1])
        thetas.append(np.arctan2(state[3], state[2]))
        action = expert.act(obs, state)
        obs, done = env.step(action)
        if done:
            state = env.get_state()
            xs.append(state[0]); ys.append(state[1])
            thetas.append(np.arctan2(state[3], state[2]))
            break
    trajectories.append({'ep': ep, 'xs': np.array(xs), 'ys': np.array(ys),
                         'thetas': np.array(thetas), 'done': done})

success = sum(1 for t in trajectories if t['done'])

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
colors = cm.tab20(np.linspace(0, 1, N))

for t in trajectories:
    xs, ys, thetas = t['xs'], t['ys'], t['thetas']
    c   = colors[t['ep']] if t['done'] else 'crimson'
    lw  = 1.6 if t['done'] else 2.4
    ax.plot(xs, ys, color=c, linewidth=lw, alpha=0.85, solid_capstyle='round')

    # start dot + heading arrow
    ax.plot(xs[0], ys[0], 'o', color=c, markersize=6, zorder=4)
    dx, dy = np.cos(thetas[0]) * 0.22, np.sin(thetas[0]) * 0.22
    ax.annotate('', xy=(xs[0]+dx, ys[0]+dy), xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.6))

    if not t['done']:
        ax.plot(xs[-1], ys[-1], 'x', color='crimson',
                markersize=10, markeredgewidth=2.2, zorder=5)

# goal
ax.plot(0, 0, 'k*', markersize=16, zorder=6, label='goal')
ax.add_patch(plt.Circle((0, 0), 0.05, color='black', fill=False,
                         linestyle='--', linewidth=1.2))

ax.set_xlim(-3.7, 3.7); ax.set_ylim(-3.7, 3.7)
ax.set_aspect('equal')
ax.grid(True, alpha=0.25, linewidth=0.7)
ax.axhline(0, color='gray', linewidth=0.6, zorder=0)
ax.axvline(0, color='gray', linewidth=0.6, zorder=0)
ax.set_title(f'Expert controller — {success}/{N} episodes\n'
             f'● start  → heading  ★ goal', fontsize=12)
ax.set_xlabel('x, m'); ax.set_ylabel('y, m')

os.makedirs('figures', exist_ok=True)
out = 'figures/trajectories_expert.png'
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
plt.show()
