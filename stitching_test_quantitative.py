import os
import pickle
import random
import time
from pathlib import Path
import numpy as np
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
from utils.testing import test_rel_repr_vec

from utils.preprocess_env import (
    make_custom_env,
    make_custom_env_no_stack,
    # make_env,
)

# from ppo_naturalenv_discrete_rgb_nostack_relrepr_end_to_end import make_env

from utils.relative import init_anchors, init_anchors_from_obs, get_obs_anchors

# from rl_agents.ppo.ppo_resnet_fc import FeatureExtractorResNet, PolicyResNet, AgentResNet
# from rl_agents.ppo.ppo_end_to_end_relu import FeatureExtractor, Policy, Agent
# from rl_agents.ddqn.ddqn_end_to_end import FeatureExtractorDDQN, PolicyDDQN, AgentDDQN

from rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor, Policy, Agent

# from natural_rl_environment.natural_env import NaturalEnvWrapper

from utils.models import load_model, get_algo_instance, get_algo_instance_bw, load_model_from_path, load_encoder_from_path, load_policy_from_path

from utils.preprocess_env import PreprocessFrameRGB

from utils.env_initializer import instantiate_env, make_env_atari

from pytorch_lightning import seed_everything

seed_everything(42)

# parse args
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--stitching-mode", default="absolute", type=str, help="Stitching mode: absolute, relative, translate")
    parser.add_argument("--encoder-env-id", default="CarRacing-v2", type=str, help="Environment ID")
    parser.add_argument("--policy-env-id", default="CarRacing-v2", type=str, help="Environment ID")
    # make an optional argument for the background color, defaults to None
    parser.add_argument("--background", default=None, type=str, help="Background color")
    parser.add_argument("--encoder-colors", default=["green"], type=str, nargs="+", help="Encoder colors --> modify backgrounds")
    parser.add_argument("--policy-colors", default=["green"], type=str, nargs="+", help="Policy colors")
    parser.add_argument("--env-seeds", default=[0], type=int, nargs="+", help="Environment seeds")
    parser.add_argument("--encoder-anchors", default=None, type=str, help="Path to the encoder anchors")
    parser.add_argument("--controller-anchors", default=None, type=str, help="Path to the contrastive anchors")
    parser.add_argument("--encoder-seeds", default=[0], type=int, nargs="+", help="Encoder seeds")
    parser.add_argument("--policy-seeds", default=[0], type=int, nargs="+", help="Policy seeds")
    parser.add_argument("--encoder-algo", default="ppo", type=str, help="Encoder algorithm")
    parser.add_argument("--policy-algo", default="ppo", type=str, help="Policy algorithm")
    parser.add_argument("--zoom", default=2.7, type=float, help="Zoom factor")
    parser.add_argument("--encoder-activation-func", default="relu", type=str, help="Encoder activation function")
    parser.add_argument("--policy-activation-func", default="relu", type=str, help="Policy activation function")
    parser.add_argument("--playon", default="policy", type=str, help="Play on: policy, encoder. The env to play on")
    # parser.add_argument("--anchoring-method", default=None, type=str, help="Anchoring method: fps, kmeans, random")
    # mandatory args
    # parser.add_argument("--encoder-dir", default=None, type=str, help="Path to the encoder model to test", required=True)
    # parser.add_argument("--policy-dir", default=None, type=str, help="Path to the policy model to test", required=True)

    # args mandatory only if stitching mode is translate
    parser.add_argument("--anchors-alpha", default=None, type=str, help="Alpha value to use for anchors")
    parser.add_argument("--anchors-method", default="fps", type=str, help="Method to use for anchors: fps, kmeans, random")
    

    parser.add_argument("--render-mode", default="rgb_array", type=str, help="Render mode: human, rgb_array")
    
    args = parser.parse_args()
    return args

""" CARRACING """
""" standard: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2 --encoder-colors green red blue --policy-colors green red blue yellow --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2 --encoder-colors green red blue yellow --policy-colors green red blue yellow --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2 --encoder-colors green red blue --policy-colors green red blue --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array
""" multicolor: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2 --encoder-colors multicolor --policy-colors multicolor --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
""" no_noop_4as: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-no_noop_4as --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-no_noop_4as --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-no_noop_4as --encoder-colors green red blue --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array
""" slow: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-slow --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-slow --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-slow --encoder-colors green red blue --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array
""" noleft: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-noleft --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-noleft --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
""" scrambled: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-scrambled --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-scrambled --encoder-colors green red blue yellow --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id CarRacing-v2 --policy-env-id CarRacing-v2-scrambled --encoder-colors green red blue --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array
""" camera_far (policy standard): abs, transl, rel"""
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2 --encoder-colors green --policy-colors green red blue yellow --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --playon encoder
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2 --encoder-colors green --policy-colors green red blue yellow --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --playon encoder
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2 --encoder-colors green --policy-colors green red blue yellow --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array --playon encoder
""" camera_far (policy no_noop_4as) """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-no_noop_4as --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4  --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-no_noop_4as --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-anchors CarRacing-v2-camera_far --controller-anchors CarRacing-v2 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
"""camera_far (policy slow)"""
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-slow --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4  --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-slow --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4  --encoder-anchors CarRacing-v2-camera_far --controller-anchors CarRacing-v2 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
""" camera_far (policy noleft) """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-noleft --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4  --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-noleft --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4  --encoder-anchors CarRacing-v2-camera_far --controller-anchors CarRacing-v2 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
""" camera_far (policy heavy) """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-heavy --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4  --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-heavy --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4  --encoder-anchors CarRacing-v2-camera_far --controller-anchors CarRacing-v2 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
""" camera_far (policy scrambled) """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-scrambled --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4  --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-scrambled --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4  --encoder-anchors CarRacing-v2-camera_far --controller-anchors CarRacing-v2 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --zoom 1 --playon policy


""" CARRACING no stitching """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-no_noop_4as --policy-env-id CarRacing-v2-no_noop_4as --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-slow --policy-env-id CarRacing-v2-slow --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-heavy --policy-env-id CarRacing-v2-heavy --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-noleft --policy-env-id CarRacing-v2-noleft --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-scrambled --policy-env-id CarRacing-v2-scrambled --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id CarRacing-v2-camera_far --policy-env-id CarRacing-v2-camera_far --encoder-colors green --policy-colors green --env-seeds 1 2 3 4 --encoder-seeds 1 2 3 4 --policy-seeds 1 2 3 4 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array --playon policy --zoom 1







""" ATARI """
""" Breakout: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id BreakoutNoFrameskip-v4 --policy-env-id BreakoutNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id BreakoutNoFrameskip-v4 --policy-env-id BreakoutNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id BreakoutNoFrameskip-v4 --policy-env-id BreakoutNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array
""" Boxing: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id BoxingNoFrameskip-v4 --policy-env-id BoxingNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id BoxingNoFrameskip-v4 --policy-env-id BoxingNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3  --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id BoxingNoFrameskip-v4 --policy-env-id BoxingNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3  --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array
""" Pong: abs, transl, rel """
# python stitching_test_quantitative.py --stitching-mode absolute --encoder-env-id PongNoFrameskip-v4 --policy-env-id PongNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3 --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode translate --encoder-env-id PongNoFrameskip-v4 --policy-env-id PongNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3  --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --render-mode rgb_array
# python stitching_test_quantitative.py --stitching-mode relative --encoder-env-id PongNoFrameskip-v4 --policy-env-id PongNoFrameskip-v4 --encoder-colors plain green red --policy-colors plain green red --env-seeds 1 --encoder-seeds 0 1 2 3 --policy-seeds 0 1 2 3  --encoder-algo ppo --policy-algo ppo --encoder-activation-func relu --policy-activation-func relu --anchors-alpha 0.999 --render-mode rgb_array



args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_info = 'rgb'

from utils.testing import stitching_test_quantitative
results = stitching_test_quantitative(
        args.encoder_env_id, args.policy_env_id, env_info=env_info, playon=args.playon, background=args.background,
        encoder_backgrounds=args.encoder_colors, policy_backgrounds=args.policy_colors,
        env_seeds=args.env_seeds, # encoder_dir: str = None, policy_dir: str = None,
        encoder_seeds=args.encoder_seeds, policy_seeds=args.policy_seeds,
        encoder_anchors=args.encoder_anchors, controller_anchors=args.controller_anchors,
        encoder_algo=args.encoder_algo, policy_algo=args.policy_algo,
        encoder_activation_func=args.encoder_activation_func, policy_activation_func=args.policy_activation_func,
        anchors_alpha=args.anchors_alpha, zoom=args.zoom,
        stitching_mode=args.stitching_mode, anchoring_method=args.anchors_method, render_mode=args.render_mode, device=device
        )

if not os.path.exists(f"experiments/stitching_tests/{args.policy_env_id}/{env_info}/{args.stitching_mode}"):
    os.makedirs(f"experiments/stitching_tests/{args.policy_env_id}/{env_info}/{args.stitching_mode}")
    
stitch_filename = f"experiments/stitching_tests/{args.policy_env_id}/{env_info}/{args.stitching_mode}/"
if args.stitching_mode == "translate":
    stitch_filename += f"{args.anchors_method}/"
if args.stitching_mode == "relative":
    stitch_filename += f"a_{args.anchors_alpha}/"
if not os.path.exists(stitch_filename):
    os.makedirs(stitch_filename)
stitch_filename += f"{args.encoder_env_id}_{args.encoder_activation_func}_stitching_results_{args.policy_algo}.csv"
print(stitch_filename)

# save results to csv
results.to_csv(stitch_filename, index=False)
print(f"Saved stitching results to {stitch_filename}")

# results.to_csv(stitch_filename, sep="\t", index=False)