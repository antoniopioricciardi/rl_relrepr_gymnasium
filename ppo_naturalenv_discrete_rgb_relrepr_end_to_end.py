# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


from init_training import init_stuff_ppo

# from utils.relative import get_obs_anchors
from logger import Logger

from envs.natural_rl_environment.natural_env import NaturalEnvWrapper

from utils.argparser import *


def parse_env_specific_args(parser):
    # bg_colors_allowed = ["green", "red", "blue", "violet", "yellow"]
    parser.add_argument(
        "--background",
        type=str,
        default="green",
        help="the background of the car racing environment. Can be: plain, green, red, blue",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="",  # data/track_bg_images/0.jpg",
        help="the path of the image to use for the car racing environment background, if --background is set to 'image'",
    )

    return parser


# def make_env(env, seed, idx, capture_video, run_name):
#     def thunk(env=env):
#         # env = gym.make(env_id)
#         # env = CarRacing(continuous=False, background='red')
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env = NoopResetEnv(env, noop_max=30)
#         # env = MaxAndSkipEnv(env, skip=4)
#         env = RepeatAction(env, repeat=3)
#         env = EpisodicLifeEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = ClipRewardEnv(env)
#         # env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = PreprocessFrameRGB((84, 84, 3), env) # (3, 84, 84)
#         # env = gym.wrappers.GrayScaleObservation(env)
#         env = gym.wrappers.FrameStack(env, 3) #(4, 3, 84, 84)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    parser = parse_relative_args(parser)
    parser = parse_env_specific_args(parser)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)  # 16 * 128 = 2048
    args.minibatch_size = int(
        args.batch_size // args.num_minibatches
    )  # 2048 // 4 = 512

    if args.pretrained:
        assert (
            args.model_path is not None
        ), "--model-path must be specified if --pretrained is set"
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # paths for saving models and csv files, etc. # TODO: create a function for this
    if not args.track:
        log_path = os.path.join("runs", run_name)  # f"runs/{run_name}/"
    else:
        log_path = f"{wandb.run.dir}"

    # create logger
    # logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
    # work_dir = Path.cwd() / "runs" / run_name
    logger = Logger(
        log_path, use_tb=False, use_wandb=False
    )  # True if args.track else False)
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

    if args.background == "plain":
        env = gym.make(args.env_id, render_mode="rgb_array")
        eval_env = gym.make(args.env_id, render_mode="rgb_array")
    else:
        # assert args.background in bg_colors_allowed, f"background color not supported"\
        # f", if background-type is \"color\", only {bg_colors_allowed} allowed. Otherwise use \"plain\""
        env = NaturalEnvWrapper(
            args.env_id,
            imgsource="color",
            color=args.background,
            resource_files=None,
            render_mode="rgb_array",
            zoom=4,
        )
        eval_env = NaturalEnvWrapper(
            args.env_id,
            imgsource="color",
            color=args.background,
            resource_files=None,
            render_mode="rgb_array",
            zoom=4,
        )

    # env setup
    from utils.env_initializer import make_env_atari

    num_eval_envs = 5
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env_atari(
                env,
                seed=args.seed,
                stack=args.stack_n,
                no_op=30,
                action_repeat=args.stack_n,
                max_frames=False,
                episodic_life=True,
                clip_reward=True,
                check_fire=True,
                idx=i,
                capture_video=False,
                run_name=run_name,
            )
            for i in range(args.num_envs)
        ]
    )

    eval_envs = gym.vector.AsyncVectorEnv(
        [
            make_env_atari(
                eval_env,
                seed=args.seed,
                stack=args.stack_n,
                no_op=30,
                action_repeat=args.stack_n,
                max_frames=False,
                episodic_life=True,
                clip_reward=True,
                check_fire=True,
                idx=i,
                capture_video=False,
                run_name=eval_run_name,
            )
            for i in range(num_eval_envs)
        ]
    )

    init_stuff_ppo()
