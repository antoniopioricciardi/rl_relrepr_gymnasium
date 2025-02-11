# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from utils.preprocess_env import PreprocessFrameRGB, RepeatAction

from zeroshotrl.init_training import init_stuff_ppo

from zeroshotrl.logger import CustomLogger

from pytorch_lightning import seed_everything
from zeroshotrl.utils.argparser import *

seed_everything(42)

""" MINIWORLD """
"""  """
# python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name test --env-id MiniWorld-OneRoom-v0 --seed 1 --num-envs 8 --num-eval-envs 1 --background standard --stack-n 4 --total-timesteps 200000

# python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name OneRoom-v0_standard_rgb --env-id OneRoom-v0 --seed 1 --num-envs 8 --background standard --stack-n 4 --total-timesteps 200000 --num-eval-eps 30 --num-eval-envs 3


def parse_env_specific_args(parser):
    # env specific arguments
    parser.add_argument(
        "--background",
        type=str,
        default="standard",
        help="the color of the rocks at the base"
    )
    parser.add_argument(
        "--topdown",
        type=bool,
        default=False,
        help="whether to use topdown view"
        )
    return parser


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
    logger = CustomLogger(
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
    
    # import miniworld
    from zeroshotrl.envs.miniworld.oneroom import OneRoom
    env = OneRoom(render_mode="rgb_array", topdown=args.topdown)
    eval_env = OneRoom(render_mode="rgb_array", topdown=args.topdown)

    # env = gym.make("MiniWorld-OneRoom-v0", render_mode="rgb_array")
    env = gym.make(f"{args.env_id}", render_mode="rgb_array")
    eval_env = gym.make(f"{args.env_id}", render_mode="rgb_array")
    
    num_eval_envs = args.num_eval_envs

    # env setup
    from zeroshotrl.utils.env_initializer import make_env_atari

    envs = gym.vector.SyncVectorEnv(
        [
            make_env_atari(
                env,
                seed=args.seed,
                rgb=True,
                stack=args.stack_n,
                no_op=0,
                action_repeat=0,
                max_frames=False,
                episodic_life=False,
                clip_reward=False,
                check_fire=False,
                #time_limit=1000,
                idx=i,
                capture_video=False,
                run_name=run_name,
                color_transform=args.background,
            )
            for i in range(args.num_envs)
        ]
    )

    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env_atari(
                eval_env,
                seed=args.seed,
                rgb=True,
                stack=args.stack_n,
                no_op=0,
                action_repeat=4,
                max_frames=False,
                episodic_life=False,
                clip_reward=False,
                check_fire=False,
                idx=i,
                capture_video=False,
                run_name=eval_run_name,
            )
            for i in range(num_eval_envs)
        ]
    )

    init_stuff_ppo(
        args=args,
        envs=envs,
        eval_envs=eval_envs,
        device=device,
        wandb=wandb,
        writer=writer,
        logger=logger,
        log_path=log_path,
        csv_file_path=csv_file_path,
        eval_csv_file_path=eval_csv_file_path,
    )