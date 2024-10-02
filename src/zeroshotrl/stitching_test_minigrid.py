import os
import pickle
from pathlib import Path
import numpy as np
import argparse

import torch
from zeroshotrl.utils.testing import test_rel_repr_vec

# from ppo_naturalenv_discrete_rgb_nostack_relrepr_end_to_end import make_env


# from rl_agents.ppo.ppo_resnet_fc import FeatureExtractorResNet, PolicyResNet, AgentResNet
# from rl_agents.ppo.ppo_end_to_end_relu import FeatureExtractor, Policy, Agent
# from rl_agents.ddqn.ddqn_end_to_end import FeatureExtractorDDQN, PolicyDDQN, AgentDDQN

from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor, Agent

# from natural_rl_environment.natural_env import NaturalEnvWrapper

from zeroshotrl.utils.models import (
    get_algo_instance,
    get_algo_instance_bw,
    load_model_from_path,
    load_encoder_from_path,
    load_policy_from_path,
)

from zeroshotrl.utils.env_initializer import make_env_atari

from pytorch_lightning import seed_everything

seed_everything(42)


# parse args
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # parser.add_argument("--single-test", default=True, type=bool, help="Perform a single test or an extensive stitching test")
    parser.add_argument("--stitching-mode", default="absolute", type=str, help="Stitching mode: absolute, relative, translate")
    parser.add_argument("--env-id", default="CarRacing-v2", type=str, help="Environment ID")
    parser.add_argument("--env-seed", default=0, type=int, help="Environment seed")

    # env specific arguments
    parser.add_argument("--grid-size", type=int, default=8,
                        help="the size of the grid world")
    # # goal-pos is a tuple of two ints, default is 6,6
    # parser.add_argument("--goal-pos", type=int, nargs=2, default=(6,6),
    #                     help="the position of the goal in x,y coordinates (integers)")
    parser.add_argument("--goal-shape", type=str, default="square",
                        help="the shape of the goal. Can be: square, circle")
    parser.add_argument("--goal-pos", type=str, default="right",
                        help="the position of the goal. Can be: right, left")
    parser.add_argument("--goal-color", type=str, default="green",
                        help="the color of the goal. Can be: green, red")
    parser.add_argument("--item-color", type=str, default="red",
                        help="the color of the othe item in the env. Can be: green, red")
    parser.add_argument("--wall-color", type=str, default="grey",
                        help="color of the walls. Can be: grey, red, blue")

    parser.add_argument("--use-resnet", default=False, type=bool, help="Use resnet model")
    # mandatory args
    parser.add_argument("--encoder-dir", default=None, type=str, help="Path to the encoder model to test", required=False)
    parser.add_argument("--policy-dir", default=None, type=str, help="Path to the policy model to test", required=True)

    # args mandatory only if stitching mode is translate
    parser.add_argument("--anchors-file1", default=None, type=str, help="Path to the anchors to use for stitching")
    parser.add_argument("--anchors-file2", default=None, type=str, help="Path to the anchors to use for stitching")
    parser.add_argument("--anchors-alpha", default=None, type=str, help="Alpha value to use for anchors")
    parser.add_argument("--anchors-method", default="random", type=str, help="Method to use for anchors: fps, kmeans, random")

    parser.add_argument("--render-mode", default="human", type=str, help="Render mode: human, rgb_array")

    args = parser.parse_args()
    return args


""" CARRACING """
""" abs/transl/relative/resnet (green blue)"""
# python stitching_test.py --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/blue/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_blue_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
# python stitching_test.py --stitching-mode relative --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/relative/relu/alpha_0_999/seed_2 --anchors-alpha None --anchors-method fps --render-mode human
# python stitching_test.py --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color green --use-resnet True --policy-dir models/CarRacing-v2/rgb/green/ppo_resnet/absolute/relu/seed_1
""" multicolor """
# python stitching_test.py --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color multicolor --encoder-dir models/CarRacing-v2/rgb/multicolor/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/multicolor/ppo/absolute/relu/seed_1
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2 --env-seed 1 --background-color multicolor --encoder-dir models/CarRacing-v2/rgb/multicolor/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/blue/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_multicolor_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_blue_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
# python stitching_test.py --stitching-mode relative --env-id CarRacing-v2 --env-seed 1 --background-color multicolor --encoder-dir models/CarRacing-v2/rgb/multicolor/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/CarRacing-v2/rgb/multicolor/ppo/relative/relu/alpha_0_999/seed_2 --anchors-alpha None --anchors-method fps --render-mode human
""" no_noop_4as """
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2-no_noop_4as --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-no_noop_4as/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" scrambled """
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2-scrambled --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-scrambled/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" onlyleft """
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2-onlyleft --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-onlyleft/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" slow """
# python stitching_test.py --stitching-mode absolute --env-id CarRacing-v2-slow --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1
""" slow: abs/transl/relative """
# python stitching_test.py --stitching-mode absolute --env-id CarRacing-v2-slow --env-seed 4 --background-color green --encoder-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2-slow --env-seed 4 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
# python stitching_test.py --stitching-mode relative --env-id CarRacing-v2-slow --env-seed 4 --background-color green --encoder-dir models/CarRacing-v2-slow/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --anchors-alpha None --anchors-method fps --render-mode human
""" camera far"""
# python stitching_test.py --stitching-mode absolute --env-id CarRacing-v2-camera_far --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2-camera_far --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2-camera_far/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" camera_far (policy no_noop_4as) """
# python stitching_test.py --stitching-mode translate --env-id CarRacing-v2-no_noop_4as --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-no_noop_4as/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2-camera_far/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --zoom 2.7 --render-mode human


""" ATARI Breakout """
""" abs/transl """
# python stitching_test.py --stitching-mode absolute --env-id BreakoutNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1
# python stitching_test.py --stitching-mode translate --env-id BreakoutNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-file2 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" different cols """
# python stitching_test.py --stitching-mode translate --env-id BreakoutNoFrameskip-v4 --env-seed 1 --background-color green --encoder-dir models/BreakoutNoFrameskip-v4/rgb/green/ppo/absolute/relu/seed_0 --policy-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human

""" ATARI boxing """
""" abs/transl """
# python stitching_test.py --stitching-mode absolute --env-id BoxingNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1
# python stitching_test.py --stitching-mode translate --env-id BoxingNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-file2 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" transl different cols """
# python stitching_test.py --stitching-mode translate --env-id BoxingNoFrameskip-v4 --env-seed 1 --background-color green --encoder-dir models/BoxingNoFrameskip-v4/rgb/green/ppo/absolute/relu/seed_0 --policy-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human

""" ATARI pong """
""" abs/transl """
# python stitching_test.py --stitching-mode absolute --env-id PongNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/PongNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/PongNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1

""" MINIGRID """
""" abs/transl """
# python stitching_test_minigrid.py --stitching-mode absolute --env-id minigrid --env-seed 1 --grid-size 6 --goal-shape square --goal-pos right --goal-color green --item-color red --wall-color grey --encoder-dir models/minigrid_dual/rgb/goal_square-right-green/item_square-left-red/absolute/seed_2 --policy-dir models/minigrid_dual/rgb/goal_square-right-green/item_square-left-red/absolute/seed_2


args = parse_args()
if args.stitching_mode == "translate":
    assert (
        args.anchors_file1 is not None and args.anchors_file2 is not None
    ), "Anchors file must be specified for stitching mode translate"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Single test allows to choose a single combination of encoder and policy to test.
If False, perform an extensive stitching test over all combinations of encoder and policy colors. """

# single_test = args.single_test
# stitching_modes = ["absolute", "relative", "translate"]
stitching_md = args.stitching_mode

anchoring_method = args.anchors_method  # "fps"  # "fps", "kmeans", "random"


relative = False
if stitching_md == "relative":
    relative = True

pretrained = False
# model_type = 'ppo'

env_id = args.env_id  # "CarRacing-v2" # "Wolfenstein-basic" # "StarGunnerNoFrameskip-v4  # "CarRacing-v2" #"BoxingNoFrameskip-v4"# "BreakoutNoFrameskip-v4"
env_info = "rgb"
cust_seed = args.env_seed

env_seeds_totest = [0]
if env_id.startswith("CarRacing"):
    env_seeds_totest = [40, 41, 42, 43]  # 39
# torch.Size([4000, 3, 84, 84])

""" Parameters to change for single test """
# background_color = "green"
model_color_1 = "green"
model_color_2 = "green"

model_algo_1 = "ppo"
model_algo_2 = "ppo"
""" ----- """

video_path = "data/track_bg_videos/0.mp4"
image_path = "data/track_bg_images/0.jpg"

# imgsource = "plain" # "color" "plain"
render_md = args.render_mode
num_envs = 1

from minigrid.envs.empty_dual import EmptyDualEnv
from minigrid.wrappers import *
from utils.preprocess_env import FilterFromDict

env = EmptyDualEnv(
    size=args.grid_size,
    goal_shape=args.goal_shape,
    goal_pos=args.goal_pos,
    goal_color=args.goal_color,
    item_color=args.item_color,
    wall_color=args.wall_color,
    render_mode="human",
)
# env = EmptyEnv(size=8)
env = RGBImgPartialObsWrapper(env)
env = FilterFromDict(env, "image")

envs = gym.vector.SyncVectorEnv(
    [
        make_env_atari(
            env,
            seed=cust_seed,
            rgb=True,
            stack=4,
            no_op=0,
            action_repeat=0,
            max_frames=False,
            episodic_life=False,
            clip_reward=False,
            check_fire=False,
            idx=i,
            capture_video=False,
            filter_dict=None,
            run_name="test",
        )
        for i in range(num_envs)
    ]
)

# envs = init_env(env_id, env_info, background_color=args.background_color, image_path=image_path, zoom=args.zoom, cust_seed=args.env_seed, render_md=render_md, num_envs=num_envs)

if env_info == "rgb":
    encoder_instance, policy_instance, agent_instance = get_algo_instance(
        model_algo_1, model_algo_2, use_resnet=args.use_resnet
    )
else:
    encoder_instance, policy_instance, agent_instance = get_algo_instance_bw(
        model_algo_1, model_algo_2
    )
if not args.use_resnet:
    path1_enc = os.path.join(args.encoder_dir, "encoder.pt")
    path2_enc = os.path.join(args.policy_dir, "encoder.pt")
# path1_pol = os.path.join(args.encoder_dir, "policy.pt")
path2_pol = os.path.join(args.policy_dir, "policy.pt")

random_encoder = False
if random_encoder:
    obs_anchors = None
    # if is_relative:
    #     obs_anchors = encoder_params["obs_anchors"]
    encoder1 = FeatureExtractor(
        use_relative=relative,
        pretrained=False,
        obs_anchors=obs_anchors,
        anchors_alpha=None,
    ).to(device)
elif args.use_resnet:
    from rl_agents.ppo.ppo_resnet import FeatureExtractorResNet

    encoder1 = FeatureExtractorResNet().to(device)
else:
    encoder1 = load_encoder_from_path(
        path1_enc,
        encoder_instance,
        is_relative=relative,
        is_pretrained=False,
        anchors_alpha=None,
        encoder_eval=True,
        device=device,
    )
if args.use_resnet:
    policy2 = load_policy_from_path(
        path2_pol,
        envs.single_action_space.n,
        policy_instance,
        policy_eval=True,
        encoder_out_dim=encoder1.out_dim,
        repr_dim=3136,
        device=device,
    )
else:
    encoder2, policy2, agent2 = load_model_from_path(
        path2_enc,
        path2_pol,
        envs.single_action_space.n,
        encoder_instance,
        policy_instance,
        agent_instance,
        is_relative=False,
        is_pretrained=False,
        device=device,
    )

translation = None
if stitching_md == "translate":
    from latentis.space import LatentSpace
    from latentis.utils import seed_everything

    # from latentis import transforms
    from latentis.estimate.dim_matcher import ZeroPadding
    from latentis.estimate.orthogonal import SVDEstimator
    from latentis.translate.translator import LatentTranslator

    obs_set_1 = pickle.load(Path(args.anchors_file1).open("rb"))  # [30:2000]
    obs_set_2 = pickle.load(Path(args.anchors_file2).open("rb"))  # [30:2000]
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
    space1 = encoder1.forward_single(obs_set_1.to(device)).detach().cpu()
    space2 = encoder2.forward_single(obs_set_2.to(device)).detach().cpu()

    # print('AAAAA', obs_set_1.shape, obs_set_2.shape, space1.shape, space2.shape)

    from collections import namedtuple

    Space = namedtuple("Space", ["name", "vectors"])
    space1 = Space(name=model_color_1, vectors=space1)
    space2 = Space(name=model_color_2, vectors=space2)

    space1_vectors = space1.vectors
    space2_vectors = space2.vectors

    space1_anchors = space1_vectors[:]
    space2_anchors = space2_vectors[:]

    # compute mean distance between anchors
    diff = space1_anchors - space2_anchors
    print("mean distance between anchors: ", diff.mean())

    from utils.anchoring_methods import get_anchors

    """ CHANGE ANCHOR SAMPLING METHOD HERE """
    # if not os.path.exists(f"alignment_indices/{env_id}/{env_info}"):
    #     os.makedirs(f"alignment_indices/{env_id}/{env_info}")
    # translation_path = f'alignment_indices/{env_id}/{env_info}/{anchoring_method}_{model_color_1}_{model_seed_1}_closest.pt'#{model_color_2}_closest.pt'

    align_path = os.path.join(
        "alignment_indices", str(args.encoder_dir).replace("models/", "")
    )
    if not os.path.exists(align_path):
        os.makedirs(align_path)
    translation_path = os.path.join(align_path, f"{args.anchors_method}_closest.pt")

    num_anchors = 3136  # len(space1_anchors) # 3136
    space1_anchors, space2_anchors = get_anchors(
        space1_vectors,
        space2_vectors,
        num_anchors,
        subset_indices,
        anchoring_method,
        translation_path,
        device,
    )

    translation = LatentTranslator(
        random_seed=42,
        estimator=SVDEstimator(
            dim_matcher=ZeroPadding()
        ),  # SGDAffineTranslator(),#SVDEstimator(dim_matcher=ZeroPadding()),
        source_transforms=None,  # [transforms.StandardScaling()],
        target_transforms=None,  # [transforms.StandardScaling()],
    )
    # translation = LatentTranslation(
    #     seed=42,
    #     translator=SVDTranslator(),
    #     source_transforms=None, #[Transforms.StandardScaling()],
    #     target_transforms=None, #[Transforms.StandardScaling()],
    # )
    space1_anchors = space1_anchors.to(device)  # [:3136]
    space2_anchors = space2_anchors.to(device)  # [:3136]
    space1 = LatentSpace(vectors=space1_anchors, name="space1")
    space2 = LatentSpace(vectors=space2_anchors, name="space2")
    print("\n##############################################\n")
    print(
        f"fitting translation layer between {model_color_1} and {model_color_2} spaces..."
    )
    translation.fit(source_data=space1, target_data=space2)
    print("done.\n\n")
    print(translation(space1))
# agent = Agent(encoder1, policy2, translation=translation).to(device)
if args.use_resnet:
    from rl_agents.ppo.ppo_resnet import AgentResNet

    agent = AgentResNet(encoder1, policy2).to(device)
else:
    agent = Agent(encoder1, policy2, translation=translation).to(device)

# translated_obs = translation(agent.encoder.forward_single(obs_set_1.to(device)).detach().cpu())
# print('######')
# print(space1_anchors)
# print('######')
# print(space2_anchors)
# print('######')
# print(translated_obs)

# from scipy.spatial.distance import cdist

# dist1 = cdist(space1_vectors, space2_vectors)
# dist2 = cdist(space1_vectors, translated_obs['target'])
# dist3 = cdist(space2_vectors, translated_obs['target'])

# print(f"mean distance between {model_color_1} and {model_color_2} spaces: {dist1.mean()}")
# print(f"mean distance between {model_color_1} and {model_color_2} spaces translated: {dist2.mean()}")
# print(f"mean distance between {model_color_1} and {model_color_2} spaces translated: {dist3.mean()}")
# exit(3)

# env_type = "tanh_rgb_nostack"

test_rel_repr_vec(
    envs,
    agent,
    policy_algo=model_algo_2,
    limit_episode_length=4000,
    device=device,
    seed=args.env_seed,
    num_envs=num_envs,
)
# test_rel_repr(envs, agent)
