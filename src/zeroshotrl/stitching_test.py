import os
import pickle
from pathlib import Path
import numpy as np
import argparse

import torch
from zeroshotrl.utils.testing import test_rel_repr_vec
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
    parser.add_argument("--background-color", default="green", type=str, help="Background color of the environment")
    parser.add_argument("--zoom", default=2.7, type=float, help="Zoom factor of the environment")

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
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/blue/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_blue_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode relative --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/relative/relu/alpha_0_999/seed_2 --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color green --use-resnet True --policy-dir models/CarRacing-v2/rgb/green/ppo_resnet/absolute/relu/seed_1
""" multicolor """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color multicolor --encoder-dir models/CarRacing-v2/rgb/multicolor/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/multicolor/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2 --env-seed 1 --background-color multicolor --encoder-dir models/CarRacing-v2/rgb/multicolor/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/blue/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_multicolor_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_blue_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode relative --env-id CarRacing-v2 --env-seed 1 --background-color multicolor --encoder-dir models/CarRacing-v2/rgb/multicolor/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/CarRacing-v2/rgb/multicolor/ppo/relative/relu/alpha_0_999/seed_2 --anchors-alpha None --anchors-method random --render-mode human
""" no_noop_4as """
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2-no_noop_4as --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-no_noop_4as/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
""" scrambled """
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2-scrambled --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-scrambled/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
""" onlyleft """
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2-onlyleft --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-onlyleft/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
""" slow """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id CarRacing-v2-slow --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1
""" slow: abs/transl/relative """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id CarRacing-v2-slow --env-seed 4 --background-color green --encoder-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2-slow --env-seed 4 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode relative --env-id CarRacing-v2-slow --env-seed 4 --background-color green --encoder-dir models/CarRacing-v2-slow/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/CarRacing-v2-slow/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --anchors-alpha None --anchors-method random --render-mode human
""" camera far"""
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id CarRacing-v2-camera_far --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2-camera_far --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2-camera_far/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
""" camera_far (policy no_noop_4as) """
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2-no_noop_4as --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-camera_far/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-no_noop_4as/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2-camera_far/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method random --zoom 2.7 --render-mode human
""" standard bus (absolute/translate/relative) """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id CarRacing-v2-bus --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2-bus/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2-bus/rgb/green/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id CarRacing-v2-bus --env-seed 1 --background-color red --encoder-dir models/CarRacing-v2-bus/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2-bus/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode relative --env-id CarRacing-v2-bus --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/CarRacing-v2-bus/rgb/green/ppo/relative/relu/alpha_0_999/seed_1 --anchors-alpha None --anchors-method random --render-mode human
""" ATARI Breakout """
""" abs/transl """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id BreakoutNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id BreakoutNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-file2 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
""" different cols """
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id BreakoutNoFrameskip-v4 --env-seed 1 --background-color green --encoder-dir models/BreakoutNoFrameskip-v4/rgb/green/ppo/absolute/relu/seed_0 --policy-dir models/BreakoutNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/BreakoutNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method random --render-mode human

""" ATARI boxing """
""" abs/transl """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id BoxingNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id BoxingNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-file2 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
""" transl different cols """
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id BoxingNoFrameskip-v4 --env-seed 1 --background-color green --encoder-dir models/BoxingNoFrameskip-v4/rgb/green/ppo/absolute/relu/seed_0 --policy-dir models/BoxingNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/BoxingNoFrameskip-v4/rgb_ppo_transitions_plain_obs.pkl --anchors-alpha None --anchors-method random --render-mode human

""" ATARI pong """
""" abs/transl """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id PongNoFrameskip-v4 --env-seed 1 --background-color plain --encoder-dir models/PongNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1 --policy-dir models/PongNoFrameskip-v4/rgb/plain/ppo/absolute/relu/seed_1

""" LunarLander """
""" gravity -10: abs/transl/relative (white/red) """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id LunarLanderRGB --env-seed 1 --background-color white --encoder-dir models/LunarLanderRGB/rgb/white/ppo/absolute/relu/seed_1 --policy-dir models/LunarLanderRGB/rgb/white/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id LunarLanderRGB --env-seed 1 --background-color white --encoder-dir models/LunarLanderRGB/rgb/white/ppo/absolute/relu/seed_1 --policy-dir models/LunarLanderRGB/rgb/red/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/LunarLanderRGB/rgb_ppo_transitions_white_obs.pkl --anchors-file2 data/anchors/LunarLanderRGB/rgb_ppo_transitions_red_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode relative --env-id LunarLanderRGB --env-seed 1 --background-color white --encoder-dir models/LunarLanderRGB/rgb/white/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/LunarLanderRGB/rgb/white/ppo/relative/relu/alpha_0_999/seed_1 --anchors-alpha None --anchors-method random --render-mode human

""" gravity -3: abs/transl/relative (white/red) """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id LunarLanderRGB-3 --env-seed 1 --background-color white --encoder-dir models/LunarLanderRGB-3/rgb/white/ppo/absolute/relu/seed_1 --policy-dir models/LunarLanderRGB-3/rgb/white/ppo/absolute/relu/seed_1
# python src/zeroshotrl/stitching_test.py --stitching-mode translate --env-id LunarLanderRGB-3 --env-seed 1 --background-color red --encoder-dir models/LunarLanderRGB/rgb/red/ppo/absolute/relu/seed_1 --policy-dir models/LunarLanderRGB-3/rgb/white/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/LunarLanderRGB/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/LunarLanderRGB/rgb_ppo_transitions_white_obs.pkl --anchors-alpha None --anchors-method random --render-mode human
# python src/zeroshotrl/stitching_test.py --stitching-mode relative --env-id LunarLanderRGB-3 --env-seed 1 --background-color white --encoder-dir models/LunarLanderRGB-3/rgb/white/ppo/relative/relu/alpha_0_999/seed_1 --policy-dir models/LunarLanderRGB-3/rgb/white/ppo/relative/relu/alpha_0_999/seed_1 --anchors-alpha None --anchors-method random --render-mode human

""" MiniworldOneRoom """
""" abs/transl/relative (standard/red) """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id MiniWorld-OneRoom-v0 --env-seed 1 --background-color standard --encoder-dir models/MiniWorld-OneRoom-v0/rgb/standard/ppo/absolute/relu/seed_1 --policy-dir models/MiniWorld-OneRoom-v0/rgb/standard/ppo/absolute/relu/seed_1

""" MiniWorldFourRooms """
# python src/zeroshotrl/stitching_test.py --stitching-mode absolute --env-id MiniWorld-FourRooms-v0 --env-seed 1 --background-color standard --encoder-dir models/MiniWorld-FourRooms-v0/rgb/standard/ppo/absolute/relu/seed_1 --policy-dir models/MiniWorld-FourRooms-v0/rgb/standard/ppo/absolute/relu/seed_1

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

# env_seeds_totest = [0]
# if env_id.startswith("CarRacing"):
#     env_seeds_totest = [40, 41, 42, 43] # 39
# torch.Size([4000, 3, 84, 84])

""" Parameters to change for single test """
model_color_1 = args.background_color
model_color_2 = "--" # args.policy_color

model_algo_1 = "ppo"
model_algo_2 = "ppo"

model_activation_1 = "relu"
model_activation_2 = "relu"

model_alpha_1 = None  # "0999"
model_alpha_2 = None  # "0999"
""" ----- """

video_path = "data/track_bg_videos/0.mp4"
image_path = "data/track_bg_images/0.jpg"

# imgsource = "plain" # "color" "plain"
render_md = args.render_mode

# env_pathname = f"{env_id}"
# env_pathname2 = f"{env_id2}"
num_envs = 1
envs = init_env(
    env_id,
    env_info,
    background_color=args.background_color,
    image_path=image_path,
    zoom=args.zoom,
    cust_seed=args.env_seed,
    render_md=render_md,
    num_envs=num_envs,
)
def init_stuff(envs, num_envs=1):
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
        print('enc1')
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
        print('pol2')
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
        from latentis.estimate.affine import SGDAffineTranslator


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
            "alignment_indices", str(args.encoder_dir).replace("models/", "")
        )
        if not os.path.exists(align_path):
            os.makedirs(align_path)
        translation_path = os.path.join(align_path, f"{args.anchors_method}_closest.pt")

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
    if args.use_resnet:
        from rl_agents.ppo.ppo_resnet import AgentResNet

        agent = AgentResNet(encoder1, policy2).to(device)
    else:
        agent = Agent(encoder1, policy2, translation=translation, num_envs=num_envs).to(device)

    return agent, encoder1, policy2
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

agent, encoder1, policy2 = init_stuff(envs)

finetuning = True
if finetuning:
    import gymnasium as gym

    gravity = -10
    if "-" in env_id:
        gravity = -int(env_id.split("-")[-1])
    from zeroshotrl.envs.lunarlander.lunar_lander_rgb import LunarLanderRGB
    print("Gravity:", gravity)
    env = LunarLanderRGB(render_mode="rgb_array", color=model_color_1, gravity=gravity)
    eval_env = LunarLanderRGB(render_mode="rgb_array", color=model_color_1, gravity=gravity)
    
    num_envs = 16
    num_eval_envs = 2

    # env setup
    from zeroshotrl.utils.env_initializer import make_env_atari

    finetune_envs = gym.vector.AsyncVectorEnv(
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
                #time_limit=1000,
                idx=i,
                capture_video=False,
                run_name="finetune",
            )
            for i in range(num_envs)
        ]
    )

    eval_envs = gym.vector.AsyncVectorEnv(
        [
            make_env_atari(
                eval_env,
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
                run_name="eval_finetune",
            )
            for i in range(num_eval_envs)
        ]
    )

    agent, encoder1, policy2 = init_stuff(finetune_envs, num_envs=num_envs)
    agent.encoder.eval()

    from zeroshotrl.finetune import PPOFinetune
    print("Starting finetuning...")
    finetuner = PPOFinetune(agent, finetune_envs, eval_envs, seed=1, total_timesteps=1000000, learning_rate=0.00005, device=device)
    finetuner.train()
    print("Finetuning done.")


forced_render = False
if env_id.startswith("MiniWorld") and render_md == "human":
    forced_render = True
test_rel_repr_vec(
    envs,
    agent,
    policy_algo=model_algo_2,
    limit_episode_length=4000,
    device=device,
    seed=args.env_seed,
    num_envs=num_envs,
    forced_render=forced_render,
)
# test_rel_repr(envs, agent)
