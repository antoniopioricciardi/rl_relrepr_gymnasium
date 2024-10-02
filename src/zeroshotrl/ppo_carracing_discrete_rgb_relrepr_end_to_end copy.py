# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
from pathlib import Path
import random
import time
import gin

import gin.config
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from utils.preprocess_env import PreprocessFrameRGB, RepeatAction

from zeroshotrl.init_training import init_stuff_ppo

# from utils.relative import init_anchors, init_anchors_from_obs, get_obs_anchors, get_obs_anchors_totensor

from zeroshotrl.logger import CustomLogger

from pytorch_lightning import seed_everything
from zeroshotrl.utils.argparser import *

# python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --cfg ppo.gin 

# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000
""" CARRACING """
""" standard green: abs, rel """
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000
# python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000
# python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 1 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-indices-path data/anchor_indices/CarRacing-v2_3136_anchor_indices_from_4000.txt
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


def resolve_value(value):
    # If the value starts with %, it's a macro, so resolve it
    if isinstance(value, gin.config.ConfigurableReference):
        return gin.query_parameter(str(value))
    return value


def get_operative_config(function_name):
    # Retrieve the list of parameters for the given function
    config_dict = {}
    # Query the operative config
    operative_config = gin.operative_config_str()

    # Look for parameters bound to the specific function
    for line in operative_config.splitlines():
        print(line)
        if line.startswith(function_name):
            param_assignment = line.split("=")
            if len(param_assignment) == 2:
                full_param_name = param_assignment[0].strip()
                param_name = full_param_name.split(".")[-1]
                # Use gin.query_parameter to get the actual value
                value = gin.query_parameter(full_param_name)
                # Resolve macros if present
                config_dict[param_name] = resolve_value(value)

    return config_dict


def get_config_dict():
    # Retrieve the list of parameters for the given function
    config_dict = {}
    # Query the operative config
    operative_config = gin.config_str()

    # Look for parameters bound to the specific function
    for line in operative_config.splitlines():
        print(line)
        param_assignment = line.split("=")
        if len(param_assignment) == 2:
            full_param_name = param_assignment[0].strip()
            param_name = full_param_name.split(".")[-1]
            # Use gin.query_parameter to get the actual value
            value = gin.query_parameter(full_param_name)
            # Resolve macros if present
            config_dict[param_name] = resolve_value(value)

    return config_dict


@gin.configurable
def run(
    env_id: str,
    exp_name: str,
    batch_size: int,
    num_minibatches: int,
    torch_deterministic: bool,
    num_steps: int,
    car_mode: str,
    background: str,
    seed: int,
    stack_n: int,
    num_envs: int,
    wandb_args: dict,
    pretrained: bool,
    model_path: str,
    anchors_alpha: float,
    **init_stuff_ppo_kwargs,
):
    batch_size = int(num_envs * num_steps)  # 16 * 128 = 2048
    minibatch_size = int(batch_size // num_minibatches)  # 2048 // 4 = 512

    if pretrained:
        assert (
            model_path is not None
        ), "--model-path must be specified if --pretrained is set"
    run_name = f"{env_id}__{exp_name}__a{str(anchors_alpha)}_{seed}__{int(time.time())}"
    eval_run_name = run_name + "_eval"
    wandb = None

    run_params = get_operative_config("run")
    track = wandb_args is not None and len(wandb_args) > 0
    if track:
        import wandb

        wandb.init(
            sync_tensorboard=True,
            config=run_params,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            **wandb_args,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in run_params.items()])),
    )

    # paths for saving models and csv files, etc. # TODO: create a function for this
    if not track:
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    cuda = True
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

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
    assert car_mode in car_modes, f"car mode must be one of {car_modes}"
    zoom = 2.7
    if car_mode == "standard":
        from zeroshotrl.envs.carracing.car_racing import CarRacing
    # elif car_mode == "fast":
    #     from zeroshotrl.envs.carracing.car_racing_faster import CarRacing
    elif car_mode == "slow":
        # python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name green_rgb --env-id CarRacing-custom --seed 0 --num-envs 16 --background green --stack-n 4 --total-timesteps 5000000 --car-mode slow
        from zeroshotrl.envs.carracing.car_racing_slow import CarRacing
    # elif car_mode == "no_noop":
    #     from zeroshotrl.envs.carracing.car_racing_nonoop import CarRacing
    elif car_mode == "no_noop_4as":
        from zeroshotrl.envs.carracing.car_racing_nonoop_4as import CarRacing
    elif car_mode == "scrambled":
        from zeroshotrl.envs.carracing.car_racing_scrambled import CarRacing
    elif car_mode == "noleft":
        from zeroshotrl.envs.carracing.car_racing_noleft import CarRacing
    # elif car_mode == "heavy":
    #     from zeroshotrl.envs.carracing.car_racing_heavy import CarRacing
    elif car_mode == "camera_far":
        # from zeroshotrl.envs.carracing.car_racing_camera_far import CarRacing
        from zeroshotrl.envs.carracing.car_racing import CarRacing

        zoom = 1
        # python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name green_rgb --env-id CarRacing-custom --seed 0 --num-envs 16 --background green --stack-n 4 --total-timesteps 5000000 --car-mode no_noop
    env = CarRacing(
        continuous=False, background=background, zoom=zoom
    )  # , image_path=args.image_path)
    eval_env = CarRacing(
        continuous=False, background=background, zoom=zoom
    )  # , image_path=args.image_path)
    num_eval_envs = 5

    # env setup
    from zeroshotrl.utils.env_initializer import make_env_atari

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env_atari(
                env,
                seed=seed,
                rgb=True,
                stack=stack_n,
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
            for i in range(num_envs)
        ]
    )

    eval_envs = gym.vector.AsyncVectorEnv(
        [
            make_env_atari(
                eval_env,
                seed=seed,
                rgb=True,
                stack=stack_n,
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
        envs=envs,
        eval_envs=eval_envs,
        device=device,
        wandb=wandb,
        writer=writer,
        logger=logger,
        log_path=log_path,
        csv_file_path=csv_file_path,
        eval_csv_file_path=eval_csv_file_path,
        minibatch_size=minibatch_size,
        anchors_alpha=anchors_alpha,
        model_path=model_path,
        num_minibatches=num_minibatches,
        num_steps=num_steps,
        pretrained=pretrained,
        seed=seed,
        **init_stuff_ppo_kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--param", action="append", help="Gin parameter overrides.", default=[]
    )

    args = parser.parse_args()
    print(args)
    config_file = Path(args.cfg)
    assert config_file.exists(), f"Config file {config_file} does not exist."

    from gin.config import ParsedConfigFileIncludesAndImports

    cfg: ParsedConfigFileIncludesAndImports = gin.parse_config_files_and_bindings(
        [config_file], bindings=args.param
    )

    run()
