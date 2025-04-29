import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Settings
np.random.seed(42)
n_anchors = 3
n_points = 10
n_frames = 100
alpha = 0.1

# Generate initial points and anchors
points = np.random.rand(n_points, 2)
anchors = np.random.rand(n_frames, n_anchors, 2)

# anchors mean
anchors_mean = np.mean(anchors, axis=1)
# anchors std
anchors_std = np.std(anchors, axis=1)

# EMA anchors
ema_anchors = np.zeros_like(anchors)
ema_anchors[0] = anchors[0]
for t in range(1, n_frames):
    anchors_normalized = (anchors[t] - anchors_mean[t]) / anchors_std[t]
    ema_anchors[t] = (alpha * anchors_normalized + (1 - alpha) * ema_anchors[t - 1]) * anchors_std[t] + anchors_mean[t]
    # ema_anchors[t] = alpha * anchors[t] + (1 - alpha) * ema_anchors[t - 1]

# Compute relative representations (distances to anchors)
def compute_rel_repr(points, anchor_set, i):
    local_mean = anchors_mean[i]
    local_std = anchors_std[i]
    # perform centering and standard scaling of points and anchors using local statistics
    points_norm = (points - local_mean) / local_std
    anchor_set_norm = (anchor_set - local_mean) / local_std
    dists = np.linalg.norm(points_norm[:, None, :] - anchor_set_norm[None, :, :], axis=-1)
    return dists

# Colors for points
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

# Setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_title('Unstable Anchors (No EMA)')
ax2.set_title('Stable Anchors (With EMA)')

for ax in [ax1, ax2]:
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True)

# Initialize scatter plots with dummy data
scatter_no_ema = ax1.scatter(np.zeros((n_points,)), np.zeros((n_points,)), c=colors, label='Points')
scatter_anchors_no_ema = ax1.scatter(np.zeros((n_anchors,)), np.zeros((n_anchors,)), c='black', marker='x', s=100, label='Anchors')

scatter_ema = ax2.scatter(np.zeros((n_points,)), np.zeros((n_points,)), c=colors, label='Points')
scatter_anchors_ema = ax2.scatter(np.zeros((n_anchors,)), np.zeros((n_anchors,)), c='black', marker='x', s=100, label='Anchors')

for ax in [ax1, ax2]:
    ax.legend()

# Update function for animation
def animate(frame):
    # Without EMA
    rel_no_ema = compute_rel_repr(points, anchors[frame], frame)
    scatter_no_ema.set_offsets(rel_no_ema)
    scatter_anchors_no_ema.set_offsets(anchors[frame])

    # With EMA
    rel_ema = compute_rel_repr(points, ema_anchors[frame], frame)
    scatter_ema.set_offsets(rel_ema)
    scatter_anchors_ema.set_offsets(ema_anchors[frame])

    return scatter_no_ema, scatter_anchors_no_ema, scatter_ema, scatter_anchors_ema

# Animation
ani = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=True)

# ani.save('experiments/ema_relative_animation.gif', writer='imagemagick', fps=10)
plt.tight_layout()
plt.show()