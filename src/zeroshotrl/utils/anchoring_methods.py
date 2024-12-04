import torch
import torch.nn.functional as F
# from torch_cluster import fps as torchfps
from typing import Sequence
from pathlib import Path


# def fps(x: torch.Tensor, k: int, device="cpu"):
#     x = F.normalize(x, p=2, dim=-1)
#     x = torchfps(x.to(device), random_start=False, ratio=k / x.size(0))
#     anchor_ids: Sequence[int] = x.cpu().tolist()
#     return anchor_ids


# def fps(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         idx, farthest = torch.max(distance, -1)
#     return centroids


def get_anchors(
    space1_vectors,
    space2_vectors,
    num_anchors,
    subset_indices,
    anchoring_method,
    file_path,
    use_saved=True,
    device="cpu",
):
    if anchoring_method == "kmeans":
        anchor_file = Path(
            file_path
            # f"alignment_indices/kmeans_{file_path}"
        )
        if use_saved:
            if not anchor_file.exists():
                subset = F.normalize(space1_vectors[subset_indices])
                print("computing k-means on the anchors...\n")
                # num_anchors = 3136
                """ KMEANS """
                from sklearn.cluster import MiniBatchKMeans as KMeans
                from sklearn.metrics import pairwise_distances_argmin_min

                kmeans = KMeans(
                    n_clusters=num_anchors, random_state=42, verbose=1, batch_size=2048
                )
                kmeans.fit(subset.cpu())

                # get centroid ids from space1 and used as anchors for both spaces
                closest, _ = pairwise_distances_argmin_min(
                    kmeans.cluster_centers_, subset
                )
                torch.save(
                    closest,
                    anchor_file,
                )
            else:
                print("loading kmeans anchors")
                closest = torch.load(anchor_file)
        else:
            subset = F.normalize(space1_vectors[subset_indices])
            print("computing k-means on the anchors...\n")
            # num_anchors = 3136
            """ KMEANS """
            from sklearn.cluster import MiniBatchKMeans as KMeans
            from sklearn.metrics import pairwise_distances_argmin_min

            kmeans = KMeans(
                n_clusters=num_anchors, random_state=42, verbose=1, batch_size=2048
            )
            kmeans.fit(subset.cpu())

            # get centroid ids from space1 and used as anchors for both spaces
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, subset)
            closest = torch.tensor(closest)
            print("kmeans done")

        space1_anchors = space1_vectors[subset_indices][closest]
        space2_anchors = space2_vectors[subset_indices][closest]
        """ KMEANS """
    # elif anchoring_method == "fps":
    #     anchor_file = Path(
    #         file_path
    #         # f"alignment_indices/fps_{file_path}"
    #     )
    #     # Furthest Point Sampling over space1_vectors

    #     if use_saved:
    #         if anchor_file.exists():
    #             print("loading fps anchors")
    #             anchor_indices = torch.load(anchor_file)

    #         else:
    #             print("fps starting")
    #             anchor_indices = fps(
    #                 x=space1_vectors[subset_indices], k=num_anchors, device=device
    #             )
    #             torch.save(
    #                 anchor_indices,  # .cpu(),
    #                 anchor_file,
    #             )
    #             print(anchor_indices)
    #             print("fps done and saved")
    #     else:
    #         print("fps starting")
    #         anchor_indices = fps(
    #             x=space1_vectors[subset_indices], k=num_anchors, device=device
    #         )
    #         print("fps done")
    #     space1_anchors = space1_vectors[subset_indices][anchor_indices]
    #     space2_anchors = space2_vectors[subset_indices][anchor_indices]
    if anchoring_method == "random":
        anchor_file = Path(
            file_path
            # f"alignment_indices/random_{file_path}"
        )
        if use_saved:
            if anchor_file.exists():
                print("loading random anchors")
                anchor_indices = torch.load(anchor_file)
            else:
                print("gettin random anchors")
                anchor_indices = torch.randperm(len(subset_indices))[:num_anchors]
                torch.save(
                    anchor_indices,
                    anchor_file,
                )
                print("random done")
                # print(anchor_indices[anchor_indices > 4000])
        else:
            print("gettin random anchors")
            anchor_indices = torch.randperm(len(subset_indices[:num_anchors]))
            print("random done")
        space1_anchors = space1_vectors[subset_indices][anchor_indices]
        space2_anchors = space2_vectors[subset_indices][anchor_indices]

    return space1_anchors, space2_anchors
