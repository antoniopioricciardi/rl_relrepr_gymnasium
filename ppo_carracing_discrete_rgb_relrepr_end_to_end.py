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

from init_training import init_stuff_ppo

# from utils.relative import init_anchors, init_anchors_from_obs, get_obs_anchors, get_obs_anchors_totensor

from logger import CustomLogger

from pytorch_lightning import seed_everything
from utils.argparser import *

# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000
""" CARRACING """
""" standard green: abs, rel """
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000

""" standard red: abs, rel """
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background multicolor --car-mode standard --stack-n 4 --total-timesteps 5000000
seed_everything(42)


def parse_env_specific_args(parser):
    # env specific arguments
    parser.add_argument(
        "--background",
        type=str,
        default="green",
        help="the background of the car racing environment. Can be: green, red, blue, yellow",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="",  # data/track_bg_images/0.jpg",
        help="the path of the image to use for the car racing environment background, if --background is set to 'image'",
    )

    parser.add_argument(
        "--car-mode",
        type=str,
        default="standard",
        help="the model of the car. Can be: standard, fast, heavy, var1, var2",
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

    car_modes = [
        "standard",
        "slow",
        "no_noop",
        "no_noop_4as",
        "scrambled",
        "noleft",
        "heavy",
        "camera_far",
        "multicolor",
    ]
    assert args.car_mode in car_modes, f"car mode must be one of {car_modes}"
    zoom = 2.7
    if args.car_mode == "standard":
        from envs.carracing.car_racing import CarRacing
    # elif args.car_mode == "fast":
    #     from envs.carracing.car_racing_faster import CarRacing
    elif args.car_mode == "slow":
        # python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name green_rgb --env-id CarRacing-custom --seed 0 --num-envs 16 --background green --stack-n 4 --total-timesteps 5000000 --car-mode slow
        from envs.carracing.car_racing_slow import CarRacing
    # elif args.car_mode == "no_noop":
    #     from envs.carracing.car_racing_nonoop import CarRacing
    elif args.car_mode == "no_noop_4as":
        from envs.carracing.car_racing_nonoop_4as import CarRacing
    elif args.car_mode == "scrambled":
        from envs.carracing.car_racing_scrambled import CarRacing
    elif args.car_mode == "noleft":
        from envs.carracing.car_racing_noleft import CarRacing
    # elif args.car_mode == "heavy":
    #     from envs.carracing.car_racing_heavy import CarRacing
    elif args.car_mode == "camera_far":
        # from envs.carracing.car_racing_camera_far import CarRacing
        from envs.carracing.car_racing import CarRacing

        zoom = 1
        # python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name green_rgb --env-id CarRacing-custom --seed 0 --num-envs 16 --background green --stack-n 4 --total-timesteps 5000000 --car-mode no_noop
    env = CarRacing(
        continuous=False, background=args.background, zoom=zoom
    )  # , image_path=args.image_path)
    eval_env = CarRacing(
        continuous=False, background=args.background, zoom=zoom
    )  # , image_path=args.image_path)
    num_eval_envs = 5

    # env setup
    from utils.env_initializer import make_env_atari

    envs = gym.vector.AsyncVectorEnv(
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
                rgb=True,
                stack=args.stack_n,
                no_op=0,
                action_repeat=0,
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
