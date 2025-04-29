import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Settings
np.random.seed(0)
n_points = 10
n_frames = 50
alpha = 0.1  # EMA factor

# Initialize anchor points randomly
anchors_true_positions = np.random.rand(n_frames, n_points, 2)
anchors_ema_positions = np.zeros_like(anchors_true_positions)
anchors_ema_positions[0] = anchors_true_positions[0]

# Compute EMA positions
for t in range(1, n_frames):
    anchors_ema_positions[t] = (alpha * anchors_true_positions[t] +
                                (1 - alpha) * anchors_ema_positions[t - 1])

# Setup figure
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax1, ax2 = axes
ax1.set_title('Without EMA (unstable)')
ax2.set_title('With EMA (stable)')

scatter_true = ax1.scatter([], [], c='red')
scatter_ema = ax2.scatter([], [], c='blue')

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

# Animation update function
def update(frame):
    scatter_true.set_offsets(anchors_true_positions[frame])
    scatter_ema.set_offsets(anchors_ema_positions[frame])
    return scatter_true, scatter_ema

# Create animation
ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)

# Save animation to disk (requires FFmpeg installed)
ani.save('experiments/ema_animation.mp4',
         writer='ffmpeg', dpi=200)

plt.tight_layout()
plt.show()