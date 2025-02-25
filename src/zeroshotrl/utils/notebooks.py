from typing import Optional
import torch
import torch.nn.functional as F

from torch.nn.functional import mse_loss, pairwise_distance
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from torch import cosine_similarity
from enum import auto
from backports.strenum import StrEnum
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go

CMAP = "jet"


class DistMethod(StrEnum):
    COSINE = auto()
    INNER = auto()
    MSE = auto()


def self_sim_comparison(
    space1: torch.Tensor,
    space2: torch.Tensor,
    normalize: bool = False,
):
    if normalize:
        space1 = F.normalize(space1, p=2, dim=-1)
        space2 = F.normalize(space2, p=2, dim=-1)

    self_sim1 = space1 @ space1.T
    self_sim2 = space2 @ space2.T

    spearman = spearman_corrcoef(self_sim1.T, self_sim2.T)
    pearson = pearson_corrcoef(self_sim1.T, self_sim2.T)
    cosine = cosine_similarity(self_sim1, self_sim2)

    return dict(
        spearman_mean=spearman.mean().item(),
        spearman_std=spearman.std().item(),
        pearson_mean=pearson.mean().item(),
        pearson_std=pearson.std().item(),
        cosine_mean=cosine.mean().item(),
        cosine_std=cosine.std().item(),
    )


def pairwise_dist(space1: torch.Tensor, space2: torch.Tensor, method: DistMethod):
    if method == DistMethod.COSINE:
        dists = cosine_similarity(space1, space2)
    elif method == DistMethod.INNER:
        dists = pairwise_distance(space1, space2, p=2)
    elif method == DistMethod.MSE:
        dists = mse_loss(space1, space2, reduction="none").mean(dim=1)
    else:
        raise NotImplementedError

    return dists


def all_dist(space1: torch.Tensor, space2: torch.Tensor, method: DistMethod):
    if method == DistMethod.COSINE:
        space1 = F.normalize(space1, p=2, dim=-1)
        space2 = F.normalize(space2, p=2, dim=-1)
        dists = space1 @ space2.T
    elif method == DistMethod.INNER:
        dists = space1 @ space2.T
    elif method == DistMethod.MSE:
        dists = ((space1[:, None, :] - space2[None, :, :]) ** 2).mean(dim=-1)
    else:
        raise NotImplementedError

    return dists


def plot_pairwise_dist(
    space1: torch.Tensor,
    space2: torch.Tensor,
    prefix: str,
    dist_method: Optional[str] = None,
    show_title: bool = True,
):
    if dist_method is None:
        dist_methods = list(DistMethod)
    else:
        if dist_method.lower() == "cosine":
            dist_methods = [DistMethod.COSINE]
        elif dist_method.lower() == "inner":
            dist_methods = [DistMethod.INNER]
        elif dist_method.lower() == "mse":
            dist_methods = [DistMethod.MSE]
        else:
            raise ValueError(f"Unsupported distance method: {dist_method}")

    if len(dist_methods) == 1:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=False)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=len(dist_methods), figsize=(20, 6), sharey=False)

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for method, ax in zip(dist_methods, axs):
        dists = pairwise_dist(space1=space1, space2=space2, method=method)
        ax.hist(dists, bins=42)
        if show_title:
            ax.set_title(f"{prefix} pairwise similarities ({method.value})")

        ax.axvline(dists.mean(), color="k", linestyle="dashed", linewidth=1)
        min_ylim, max_ylim = ax.get_ylim()
        min_xlim, max_xlim = ax.get_xlim()

        mean_val = dists.mean()
        # Shift the text slightly from the mean, then clamp to avoid going outside the plot
        text_x = mean_val + 0.05 * (max_xlim - min_xlim)
        text_x = min(text_x, max_xlim * 0.95)
        text_x = max(text_x, min_xlim * 1.05)

        ax.text(
            text_x,
            max_ylim * 0.95,
            f"Mean: {mean_val:.2f}",
            ha="center",
        )

    plt.close()
    return fig


def plot_pairwise_dist_old(space1: torch.Tensor, space2: torch.Tensor, prefix: str):
    fig, axs = plt.subplots(
        nrows=1, ncols=len(DistMethod), figsize=(20, 6), sharey=False
    )

    for dist_method, ax in zip(DistMethod, axs):
        dists = pairwise_dist(space1=space1, space2=space2, method=dist_method)
        ax.hist(dists, bins=42)
        ax.set_title(f"{prefix} pairwise similarities ({dist_method})")
        ax.axvline(dists.mean(), color="k", linestyle="dashed", linewidth=1)

        min_ylim, max_ylim = ax.get_ylim()
        min_xlim, max_xlim = ax.get_xlim()
        ax.text(
            dists.mean() + (max_xlim - min_xlim) / 20,
            max_ylim * 0.95,
            "Mean: {:.2f}".format(dists.mean()),
        )

    plt.close()

    return fig


def plot_self_dist(space1: torch.Tensor, space2: torch.Tensor, prefix: str):
    fig, axs = plt.subplots(
        nrows=2, ncols=len(DistMethod), figsize=(20, 11), sharey=False
    )
    for dist_method, ax in zip(DistMethod, axs[0]):
        dists = all_dist(space1=space1, space2=space2, method=dist_method)
        ax.set_title(f"{prefix} self-similarities ({dist_method})")
        img = ax.imshow(dists, cmap=CMAP)
        plt.colorbar(img, ax=ax)

    for dist_method, ax in zip(DistMethod, axs[1]):
        dists = all_dist(
            space1=F.normalize(space1, p=2, dim=-1),
            space2=F.normalize(space2, p=2, dim=-1),
            method=dist_method,
        )
        ax.set_title(f"L2({prefix}) self-similarities ({dist_method})")
        img = ax.imshow(dists, cmap=CMAP)
        plt.colorbar(img, ax=ax)

    plt.close()

    return fig

def plot_latent_space(space1, space2, color1, color2, method='PCA', save_fig=False, save_path=None):
    # Assuming obs_set1 and obs_set2 are your two sets of observations
    # Combine the observations
    plot_space1 = space1
    plot_space2 = space2
    all_obs = np.concatenate((plot_space1, plot_space2), axis=0)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3)
    latent_pca = pca.fit_transform(all_obs)

    # Separate the latent encodings back into two sets
    latent_pca_set1 = latent_pca[:len(plot_space1)]
    latent_pca_set2 = latent_pca[len(plot_space2):]

    # Plot the combined latent space in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent_pca_set1[:, 0], latent_pca_set1[:, 1], latent_pca_set1[:, 2], label=f'{color1}', s=10)
    ax.scatter(latent_pca_set2[:, 0], latent_pca_set2[:, 1], latent_pca_set2[:, 2], label=f'{color2}', s=10)
    ax.set_title('Shared Mapped Latent Space Visualization (PCA)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.legend()
    if save_fig and save_path:
        plt.savefig(save_path)
    plt.show()


def plot_latent_space_plotly(space1, space2, color1, color2, method='PCA'):
    # Assuming space1, space2, color1, and color2 are already defined
    # Combine the observations
    plot_space1 = space1
    plot_space2 = space2
    all_obs = np.concatenate((plot_space1, plot_space2), axis=0)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3)
    latent_pca = pca.fit_transform(all_obs)

    # Separate the latent encodings back into two sets
    latent_pca_set1 = latent_pca[:len(plot_space1)]
    latent_pca_set2 = latent_pca[len(plot_space2):]


    # Create Plotly traces for each set
    trace1 = go.Scatter3d(
        x=latent_pca_set1[:, 0],
        y=latent_pca_set1[:, 1],
        z=latent_pca_set1[:, 2],
        mode='markers',
        marker=dict(size=4, color=color1),
        name=f'{color1}',
        text=[f"Index: {i}" for i in range(len(latent_pca_set1))],
        hoverinfo='text'
    )

    trace2 = go.Scatter3d(
        x=latent_pca_set2[:, 0],
        y=latent_pca_set2[:, 1],
        z=latent_pca_set2[:, 2],
        mode='markers',
        marker=dict(size=4, color=color2),
        name=f'{color2}',
        text=[f"Index: {i}" for i in range(len(latent_pca_set2))],
        hoverinfo='text'
    )

    # Create the figure and update layout
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title='Shared Translated Latent Space Visualization (PCA)',
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
        )
    )
    fig.layout.yaxis.scaleanchor="x"
    fig.show()

def set_cmap(cmap):
    global CMAP
    CMAP = cmap
