import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_comparison(
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    title: str,
) -> plt.Figure:
    """
    Creates a matplotlib figure comparing predicted and ground-truth
    trajectories.

    The figure includes:
    - A 3D plot of both trajectories with point-to-point lines.
    - A 2D plot showing Euclidean distance at each timestep.
    - The overall mean squared error (MSE) displayed on the plot.

    Args:
        gt_xyz: Ground truth trajectory
        pred_xyz: Predicted trajectory
        emb_name: The name of the embodiment for labeling
        epoch: The training epoch for annotation

    Returns:
        A matplotlib Figure object containing both subplots.
    """
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    ax3d.plot(
        gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
        label="Ground Truth", color="blue"
    )
    ax3d.plot(
        pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
        label="Predicted", color="red", linestyle="dashed"
    )

    for j in range(gt_xyz.shape[0]):
        ax3d.plot(
            [gt_xyz[j, 0], pred_xyz[j, 0]],
            [gt_xyz[j, 1], pred_xyz[j, 1]],
            [gt_xyz[j, 2], pred_xyz[j, 2]],
            color="gray", linestyle="dotted", linewidth=0.5
        )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(title)
    ax3d.legend()

    distances = np.linalg.norm(gt_xyz - pred_xyz, axis=1)
    mse = np.mean((gt_xyz - pred_xyz) ** 2)

    ax2d.plot(range(len(distances)), distances, marker="o", color="purple")
    ax2d.set_xlabel("Time Step")
    ax2d.set_ylabel("Euclidean Distance")
    ax2d.set_title("Distance per Time Step")
    ax2d.text(
        0.05, 0.95, f"MSE: {mse:.4f}",
        transform=ax2d.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6)
    )

    fig.tight_layout()
    return fig