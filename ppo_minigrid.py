# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# from utils.preprocess_env import PreprocessFrameRGB, RepeatAction
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import pickle
from pathlib import Path
import torch.nn.functional as F
from stable_baselines3.common.evaluation import evaluate_policy
from init_training import init_stuff_ppo

from rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor, Policy, Agent
# from utils.relative import init_anchors, init_anchors_from_obs, get_obs_anchors, get_obs_anchors_totensor
from utils.helputils import save_model, upload_csv_wandb
from utils.evaluation import evaluate_vec_env

from logger import CustomLogger
import copy

from train_ppo import PPOTrainer_vec
from pytorch_lightning import seed_everything
from utils.argparser import *

class FilterFromDict(gym.ObservationWrapper):
    from typing import List
    # Minigrid: Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function EmptyEnv._gen_mission at 0x12dfeb4c0>, None))
    """
    Filter the observation from the dict to only return values for the specified key
    """
    def __init__(self, env=None, key:str=None):
        super(FilterFromDict, self).__init__(env)
        self.env = env
        if key is None:
            print('keys is None, trying to use \'image\' as default')
            key = 'image'
        self.key = key
        self.observation_space = env.observation_space.spaces[self.key]

    def observation(self, observation):
        observation = observation['image'] # print(observation['image'])
        return observation#[0][self.key]
    
# python ppo_minigrid_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_minigrid_discrete --exp-name empty_8x8 --env-id MiniGrid-Empty-8x8-v0 --seed 1 --num-envs 16 --grid-size 8 --goal-pos 6 6 --wall-color grey --total-timesteps 5000000
""" CARRACING """
""" standard green: abs, rel """
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000

""" standard red: abs, rel """
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background multicolor --car-mode standard --stack-n 4 --total-timesteps 5000000
seed_everything(42)

def parse_env_specific_args(parser):
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
    
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    parser = parse_relative_args(parser)
    parser = parse_env_specific_args(parser)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # 16 * 128 = 2048
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 2048 // 4 = 512

    if args.pretrained:
        assert args.model_path is not None, "--model-path must be specified if --pretrained is set"
    run_name = f"{args.env_id}__{args.exp_name}__a{str(args.anchors_alpha)}_{args.seed}__{int(time.time())}"
    eval_run_name = run_name + "_eval"
    wandb = None
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # paths for saving models and csv files, etc. # TODO: create a function for this
    if not args.track:
        log_path = os.path.join("runs", run_name) #f"runs/{run_name}/"
    else:
        log_path = f"{wandb.run.dir}"

    # create logger
    # logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
    # work_dir = Path.cwd() / "runs" / run_name
    custom_logger = CustomLogger(log_path, use_tb=False, use_wandb=False)#True if args.track else False)
    csv_file_name = "train"
    csv_file_path = os.path.join(log_path, f"{csv_file_name}.csv")
    eval_csv_file_name = "eval"
    eval_csv_file_path = os.path.join(log_path, f"{eval_csv_file_name}.csv")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    from minigrid.envs.empty_dual import EmptyDualEnv    
    from minigrid.envs import *
    from minigrid.wrappers import *

    from minigrid.envs.empty_dual import EmptyDualEnv
    env = EmptyEnv(size=args.grid_size, goal_pos=(4,4), wall_color=args.wall_color)
    # env = EmptyDualEnv(size=args.grid_size, goal_shape=args.goal_shape, goal_pos=args.goal_pos, goal_color=args.goal_color, item_color=args.item_color, wall_color=args.wall_color, render_mode="rgb_array")
    # env = EmptyEnv(size=8)
    env = RGBImgPartialObsWrapper(env)
    env = FilterFromDict(env, "image")
    eval_env = EmptyEnv(size=args.grid_size, goal_pos=(4,4), wall_color=args.wall_color)
    # eval_env = EmptyDualEnv(size=args.grid_size, goal_shape=args.goal_shape, goal_pos=args.goal_shape, goal_color=args.goal_color, item_color=args.item_color, wall_color=args.wall_color, render_mode="rgb_array")
    eval_env = RGBImgPartialObsWrapper(eval_env)
    eval_env = FilterFromDict(eval_env, "image")
    num_eval_envs = 5

    # env setup
    from utils.env_initializer import make_env_atari
    
    envs = gym.vector.SyncVectorEnv([ 
        make_env_atari(
            env, seed=args.seed, rgb=True, stack=args.stack_n, no_op=0, action_repeat=0,
            max_frames=False, episodic_life=False, clip_reward=False, check_fire=False, filter_dict=None,
            idx=i, capture_video=False, run_name=run_name
            )
        for i in range(args.num_envs)
    ])

    eval_envs = gym.vector.SyncVectorEnv([
        make_env_atari(
            eval_env, seed=args.seed, rgb=True, stack=args.stack_n, no_op=0, action_repeat=0,
            max_frames=False, episodic_life=False, clip_reward=False, check_fire=False, filter_dict=None,
            idx=i, capture_video=False, run_name=eval_run_name
            )
        for i in range(num_eval_envs)
    ])

    init_stuff_ppo(args=args, envs=envs, eval_envs=eval_envs, device=device, wandb=wandb, writer=writer, logger=custom_logger, log_path=log_path, csv_file_path=csv_file_path, eval_csv_file_path=eval_csv_file_path)