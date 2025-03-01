from latentis.space import LatentSpace
from latentis.utils import seed_everything

# from latentis import transforms
from latentis.estimate.dim_matcher import ZeroPadding
from latentis.estimate.orthogonal import SVDEstimator
from latentis.translate.translator import LatentTranslator
from latentis.estimate.affine import SGDAffineTranslator

import os
import pickle
from pathlib import Path
import numpy as np
import argparse

import torch
from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor, Agent

# from natural_rl_environment.natural_env import NaturalEnvWrapper

from zeroshotrl.utils.models import (
    get_algo_instance,
    get_algo_instance_bw,
    load_model_from_path,
    load_encoder_from_path,
    load_policy_from_path,
)

# from utils.preprocess_env import PreprocessFrameRGB

from zeroshotrl.utils.env_initializer import init_env


def translate(anchors_file1, anchors_file2, encoder_dir, encoder1, encoder2, policy2, model_color_1, model_color_2, anchoring_method, use_resnet, num_envs, device):
    obs_set_1 = pickle.load(Path(anchors_file1).open("rb"))  # [30:2000]
    obs_set_2 = pickle.load(Path(anchors_file2).open("rb"))  # [30:2000]
    
    print("\n#####\nObs loaded\n#####\n")
    # subset_indices = np.random.randint(0, len(obs_set_1), 5000)
    obs_set_1 = obs_set_1  # [:4000]
    obs_set_2 = obs_set_2  # [:4000]

    print("Converting obs to torch tensor")
    # convert the (4000, 3, 84, 84) numpy array to a torch tensor
    obs_set_1 = torch.tensor(np.array(obs_set_1), dtype=torch.float32)
    obs_set_2 = torch.tensor(np.array(obs_set_2), dtype=torch.float32)
    print("Done converting obs to torch tensor\n#####\n")

    # obs_set_1 = torch.cat([obs_set_1, obs_set_2], dim=0)  # [anch_indices
    # obs_set_2 = obs_set_1

    subset_indices = np.arange(len(obs_set_1))  # [:4000]

    # obs_set_1 = torch.cat(obs_set_1, dim=0).cpu()  # [anch_indices]
    # obs_set_2 = torch.cat(obs_set_2, dim=0).cpu()  # [anch_indices]
    space1 = encoder1.forward_single(obs_set_1.to(device))#.detach().cpu()
    space2 = encoder2.forward_single(obs_set_2.to(device))#.detach().cpu()

    # print('AAAAA', obs_set_1.shape, obs_set_2.shape, space1.shape, space2.shape)

    from collections import namedtuple

    Space = namedtuple("Space", ["name", "vectors"])
    # space1 = Space(name=model_color_1, vectors=space1)
    # space2 = Space(name=model_color_2, vectors=space2)

    # space1_vectors = space1.vectors
    # space2_vectors = space2.vectors

    space1_anchors = space1 # space1_vectors[:]
    space2_anchors = space2 # space2_vectors[:]

    # compute mean distance between anchors
    diff = space1_anchors - space2_anchors
    print("mean distance between anchors: ", diff.mean())

    from zeroshotrl.utils.anchoring_methods import get_anchors

    """ CHANGE ANCHOR SAMPLING METHOD HERE """
    # if not os.path.exists(f"alignment_indices/{env_id}/{env_info}"):
    #     os.makedirs(f"alignment_indices/{env_id}/{env_info}")
    # translation_path = f'alignment_indices/{env_id}/{env_info}/{anchoring_method}_{model_color_1}_{model_seed_1}_closest.pt'#{model_color_2}_closest.pt'

    align_path = os.path.join(
        "alignment_indices", str(encoder_dir).replace("models/", "")
    )
    if not os.path.exists(align_path):
        os.makedirs(align_path)
    translation_path = os.path.join(align_path, f"{anchoring_method}_closest.pt")

    num_anchors = 3136  # len(space1_anchors) # 3136
    space1_anchors, space2_anchors = get_anchors(
        space1,
        space2,
        num_anchors,
        subset_indices,
        anchoring_method,
        translation_path,
        device,
        # use_saved=True,
    )

    from latentis.estimate.linear import LSTSQEstimator
    import latentis

    # translation = LatentTranslator(
    #     random_seed=42,
    #     estimator=SVDEstimator(
    #         dim_matcher=ZeroPadding()
    #     ),  # SGDAffineTranslator(),#SVDEstimator(dim_matcher=ZeroPadding()),
    #     source_transforms=[latentis.transform.StandardScaling()],
    #     target_transforms=[latentis.transform.StandardScaling()],
    # )


    translation = LatentTranslator(
        random_seed=42,
        estimator=LSTSQEstimator(),
        # estimator=SVDEstimator(
        #     dim_matcher=ZeroPadding()
        # ),  # SGDAffineTranslator(),#SVDEstimator(dim_matcher=ZeroPadding()),
        source_transforms=[latentis.transform.Centering()],#, latentis.transform.StandardScaling()], # [latentis.transform.Centering()], # [latentis.transform.StandardScaling()], #None
        target_transforms=[latentis.transform.Centering()],#, latentis.transform.StandardScaling()], # [latentis.transform.Centering()], # [latentis.transform.StandardScaling()],
    )
    translation = LatentTranslator(
    random_seed=42,
    estimator=SGDAffineTranslator(),
    # estimator=SVDEstimator(
    #     dim_matcher=ZeroPadding()
    # ),  # SGDAffineTranslator(),#SVDEstimator(dim_matcher=ZeroPadding()),
    source_transforms=[latentis.transform.Centering()],#, latentis.transform.StandardScaling()], # [latentis.transform.Centering()], # [latentis.transform.StandardScaling()], #None
    target_transforms=[latentis.transform.Centering()],#, latentis.transform.StandardScaling()], # [latentis.transform.Centering()], # [latentis.transform.StandardScaling()],
    )

    space1_anchors = space1_anchors.to(device)  # [:3136]
    space2_anchors = space2_anchors.to(device)  # [:3136]
    # space1 = LatentSpace(vectors=space1_anchors, name="space1")
    # space2 = LatentSpace(vectors=space2_anchors, name="space2")
    # remove gradients from the anchors
    space1_anchors = space1_anchors.detach()
    space2_anchors = space2_anchors.detach()

    print(f"fitting translation layer between {model_color_1} and {model_color_2} spaces...")
    translation.fit(source_data=space1_anchors, target_data=space2_anchors)
    print("done.\n\n")
    print("\n##############################################\n")
    
    # print mse and cosine similarity between the two spaces
    mse = torch.nn.MSELoss()
    cos = torch.nn.CosineSimilarity()
    print(
        f"mean squared error between the two spaces: {mse(space1_anchors, space2_anchors)}"
    )
    print(
        f"cosine similarity between the two spaces: {cos(space1_anchors, space2_anchors).mean()}"
    )

    space1 = space1[:900]
    space2 = space2[:900]

    translated_space1 = translation(space1)
    print("Computing avg pairwise distances between space1 and space2...")
    print(torch.cdist(space1, space2, p=2).mean())

    print("\n##############################################\n")
    print("Computing avg pairwise distances between translated space1 and space2...")
    pairwise_dist_translated = torch.cdist(translated_space1, space2, p=2).mean()
    print(pairwise_dist_translated)

    # print cosine similarities between space1 and space2, and translated space1 and space2
    cos = torch.nn.CosineSimilarity(dim=1)
    cos_sim = cos(space1, space2)
    print("cosine similarity between space1 and space2: ", cos_sim.mean())

    cos = torch.nn.CosineSimilarity(dim=1)
    cos_sim = cos(translated_space1, space2)
    print(
        "cosine similarity between translated space1 and space2: ",
        cos_sim.mean(),
    )
    # agent = Agent(encoder1, policy2, translation=translation).to(device)
    if use_resnet:
        from rl_agents.ppo.ppo_resnet import AgentResNet

        agent = AgentResNet(encoder1, policy2).to(device)
    else:
        agent = Agent(encoder1, policy2, translation=translation, num_envs=num_envs).to(device)

    return agent, encoder1, policy2, translation