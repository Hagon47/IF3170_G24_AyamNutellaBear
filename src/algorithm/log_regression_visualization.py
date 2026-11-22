from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from .log_regression import LogisticRegression

# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contour.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FFMpegWriter.html

def save_loss_video(X, y, path, feature_name="x", seed=7):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Pilih tepat satu kolom fitur untuk video.")
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Pasang FFmpeg agar video MP4 bisa dibuat.")

    model = LogisticRegression(learning_rate=0.00002, max_iteration=50, penalty=None, random=seed)
    weights = []
    model.fit(X, y, history_loss=weights)
    theta0, theta1 = np.asarray(weights).T

    pad0 = max(0.05, 0.25 * np.ptp(theta0))
    pad1 = max(0.05, 0.25 * np.ptp(theta1))
    grid0 = np.linspace(theta0.min() - pad0, theta0.max() + pad0, 60)
    grid1 = np.linspace(theta1.min() - pad1, theta1.max() + pad1, 60)
    losses = np.empty((len(grid1), len(grid0)))
    for row, slope in enumerate(grid1):
        for col, bias in enumerate(grid0):
            losses[row, col] = model.loss(bias + slope * X[:, 0], y)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    try:
        contours = ax.contour(grid0, grid1, losses, levels=14, cmap="Blues")
        ax.clabel(contours, fmt="%.3f", fontsize=8)
        ax.set_xlabel("θ₀ (bias)")
        ax.set_ylabel(f"θ₁ (bobot {feature_name})")
        ax.plot(theta0[0], theta1[0], "ks", label="Awal")
        line, = ax.plot([], [], color="crimson", label="Lintasan training")
        point, = ax.plot([], [], "o", color="crimson")
        ax.legend(loc="upper right")
        writer = FFMpegWriter(fps=5, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        with TemporaryDirectory(dir=path.parent) as temp:
            video = Path(temp) / "training.mp4"
            with writer.saving(fig, str(video), dpi=120):
                for epoch, loss in enumerate(model.loss_history):
                    line.set_data(theta0[:epoch + 1], theta1[:epoch + 1])
                    point.set_data([theta0[epoch]], [theta1[epoch]])
                    ax.set_title(f"Logistic Regression   |   Epoch {epoch}   |   Log-loss {loss:.4f}")
                    writer.grab_frame()
            video.replace(path)
    finally:
        plt.close(fig)
    return path
