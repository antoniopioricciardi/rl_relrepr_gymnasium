import torch
import torch.nn.functional as F

from torch.nn.functional import mse_loss, pairwise_distance
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from torch import cosine_similarity
from enum import auto
from strenum import StrEnum

import matplotlib.pyplot as plt

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



def plot_pairwise_dist(space1: torch.Tensor, space2: torch.Tensor, prefix: str):
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

def set_cmap(cmap):
    global CMAP
    CMAP = cmap