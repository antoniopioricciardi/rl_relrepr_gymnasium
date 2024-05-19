import torch
import numpy as np
import pickle
from pathlib import Path
import torch.nn.functional as F

def init_anchors(use_anchors, anchors_path):
    anchors = None
    anchors_mean = None
    # anchors_std = None
    if use_anchors:
        # obs_set = pickle.load(Path(f"data/tanh_bwgreen_model/rgb_transitions_{background}_r.pkl").open("rb"))
        obs_set = pickle.load(Path(anchors_path).open("rb"))

        anchors = torch.cat(
            [encoding.detach().cpu() for encoding in obs_set['encoded_repr']], dim=0
        )
        # NUM_ANCHORS: int = 3136
        # anchor_indices = list(range(anchors.size(0)))
        # random.shuffle(anchor_indices)
        # anchor_indices = anchor_indices[:NUM_ANCHORS]
        # # save anchor_indices to a txt file
        # text = ''
        # for item in anchor_indices:
        #     text += str(item) + '\n'
        # with open('anchor_indices.txt', 'w') as f:
        #     f.write(text)
        # exit(2)

        # open anchor_indices.txt and read the indices
        with open('anchor_indices_ppo3136.txt', 'r') as f:
            anchor_indices = f.readlines()
        anchor_indices = [int(item.strip()) for item in anchor_indices]
        anchors = anchors[anchor_indices, :]
        
        anchors_mean = anchors.mean(dim=0)
        # anchors_std = anchors.std(dim=0)

        # Center absolute space
        anchors = (anchors - anchors_mean)#  / anchors_std
        anchors: torch.Tensor = F.normalize(anchors, p=2, dim=-1)

    return anchors, anchors_mean


def init_anchors_from_obs(use_anchors, anchors_path, model):
    anchors = None
    anchors_mean = None
    encoded_obs = None
    # anchors_std = None
    if use_anchors:
        # obs_set = pickle.load(Path(f"data/tanh_bwgreen_model/rgb_transitions_{background}_r.pkl").open("rb"))
        obs_set = pickle.load(Path(anchors_path).open("rb"))

        obs = torch.cat(
            [encoding.detach().cpu() for encoding in obs_set['obs']], dim=0
        )

        encoded_obs = model.forward(obs, flatten=True)
        # anchors = resnet_model.update_anchors(encoded_obs)

        # open anchor_indices.txt and read the indices
        with open('anchor_indices_ppo3136.txt', 'r') as f:
            anchor_indices = f.readlines()
        anchor_indices = [int(item.strip()) for item in anchor_indices]
        encoded_obs = encoded_obs[anchor_indices, :]
        # anchors = F.normalize(anchors, p=2, dim=-1)
        
        anchors_mean = anchors.mean(dim=0)
        # anchors_std = anchors.std(dim=0)

        # Center absolute space
        anchors = (anchors - anchors_mean)# / anchors_std
        anchors: torch.Tensor = F.normalize(anchors, p=2, dim=-1)
    # del resnet_model
    return anchors, anchors_mean, encoded_obs


def get_obs_anchors(anchors_path):#, anchors_indices_path='anchor_indices_ppo3136.txt'):
    obs_set = pickle.load(Path(anchors_path).open("rb"))  # [30:2000]
    print('\n#####\nObs loaded\n#####\n')
    obs_set = obs_set#[:4000]
    print('Converting obs to torch tensor')
    # convert the (4000, 3, 84, 84) numpy array to a torch tensor
    obs_set = torch.tensor(np.array(obs_set), dtype=torch.float32)
    obs_set = obs_set.to('cuda' if torch.cuda.is_available() else 'cpu')

    print('Done converting obs to torch tensor\n#####\n')
    return obs_set



def get_obs_anchors_old(anchors_path, anchors_indices_path='anchor_indices_ppo3136.txt'):
    obs_set = pickle.load(Path(anchors_path).open("rb"))
    anchor_obs = torch.cat(obs_set, dim=0)

    # open anchor_indices.txt and read the indices
    with open(anchors_indices_path, 'r') as f:
        anchor_indices = f.readlines()
    anchor_indices = [int(item.strip()) for item in anchor_indices]
    anchor_obs = anchor_obs[anchor_indices, :]
    return anchor_obs

def get_obs_anchors_totensor(anchors_path, anchors_indices_path='anchor_indices_ppo3136.txt'):
    """
    Convert the anchors observations to tensors before returning them.
    """
    obs_set = pickle.load(Path(anchors_path).open("rb"))
    obs_set = [torch.tensor(obs, dtype=torch.float32).unsqueeze(0) for obs in obs_set]

    anchor_obs = torch.cat(obs_set, dim=0)

    # open anchor_indices.txt and read the indices
    with open(anchors_indices_path, 'r') as f:
        anchor_indices = f.readlines()
    anchor_indices = [int(item.strip()) for item in anchor_indices]
    anchor_obs = anchor_obs[anchor_indices, :]
    return anchor_obs



def svd_align_state(x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
    assert x.size(1) == y.size(
        1
    ), f"Dimension mismatch between {x.size(1)} and {y.size(1)}. Forgot some padding/truncation transforms?"

    #  Compute the translation vector that aligns A to B using SVD.
    u, sigma, vt = torch.svd((y.T @ x).T)
    translation_matrix = u @ vt.T

    translation_matrix = torch.as_tensor(translation_matrix, dtype=x.dtype, device=x.device)
    u, sigma, vt = torch.svd((translation_matrix))
    translation_matrix = u[:, :k] @ torch.diag(sigma[:k]) @ vt[:, :k].T
    return dict(matrix=translation_matrix, sigma=sigma)