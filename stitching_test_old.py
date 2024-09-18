import os
import pickle
from pathlib import Path
import numpy as np
import argparse

import gym
import torch
import pandas as pd


# from ppo_naturalenv_discrete_rgb_nostack_relrepr_end_to_end import make_env


# from rl_agents.ppo.ppo_resnet_fc import FeatureExtractorResNet, PolicyResNet, AgentResNet
# from rl_agents.ppo.ppo_end_to_end_relu import FeatureExtractor, Policy, Agent
# from rl_agents.ddqn.ddqn_end_to_end import FeatureExtractorDDQN, PolicyDDQN, AgentDDQN

from rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent


from utils.models import (
    get_algo_instance,
    get_algo_instance_bw,
    load_model_from_path,
    load_encoder_from_path,
)


from utils.env_initializer import make_env_atari

from pytorch_lightning import seed_everything

seed_everything(42)


""" CARRACING """
""" abs """
# python stitching_traces_carracing.py --single-test True --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1
""" transl (red green)"""
# python stitching_traces_carracing.py --single-test True --stitching-mode translate --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human


def init_carracing_env(
    car_mode="standard",
    background_color="green",
    image_path=None,
    cust_seed=0,
    render_md="rgb_array",
):
    if car_mode == "slow":
        from car_racing_slow import CarRacing
    elif car_mode == "no_noop":
        from car_racing_nonoop import CarRacing
    elif car_mode == "no_noop_4as":
        from car_racing_nonoop_4as import CarRacing
    elif car_mode == "scrambled":
        from car_racing_scrambled import CarRacing
    elif car_mode == "onlyleft":
        from car_racing_noleft import CarRacing
    elif car_mode == "heavy":
        from car_racing_heavy import CarRacing
    else:
        from car_racing import CarRacing
    env = CarRacing(
        continuous=False,
        background=background_color,
        image_path=image_path,
        render_mode="human",
    )
    nv = gym.vector.SyncVectorEnv(
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
                clip_reward=True,
                check_fire=False,
                idx=i,
                capture_video=False,
                run_name="test",
            )
            for i in range(1)
        ]
    )
    return nv


def init_env(
    env_id, background_color="green", image_path=None, cust_seed=0, render_md="human"
):
    if env_id.startswith("CarRacing-v2"):
        # separate car mode from env_id
        car_mode = env_id.split("-")[-1]
        nv = init_carracing_env(
            car_mode=car_mode,
            background_color=background_color,
            image_path=image_path,
            cust_seed=cust_seed,
            render_md=render_md,
        )
    elif env_id.startswith("Wolfenstein"):
        lvl = env_id.split("-")[-1]
        from wolfenstein_rl.wolfenstein_env import Wolfenstein

        use_rgb = True if env_info == "rgb" else False
        env = Wolfenstein(level=lvl, render_mode="human").env
        nv = gym.vector.SyncVectorEnv(
            [
                make_env_atari(
                    env,
                    seed=cust_seed,
                    rgb=use_rgb,
                    stack=4,
                    no_op=0,
                    action_repeat=0,
                    max_frames=False,
                    episodic_life=False,
                    clip_reward=False,
                    check_fire=False,
                    idx=i,
                    capture_video=False,
                    run_name="test",
                )
                for i in range(1)
            ]
        )
    else:
        from natural_rl_environment.natural_env import NaturalEnvWrapper

        if background_color == "plain":
            imgsource = "plain"
            env = gym.make(env_id, render_mode=render_md)
        else:
            imgsource = "color"
            env = NaturalEnvWrapper(
                env_id, imgsource, render_mode=render_md, color=background_color
            )
        nv = gym.vector.SyncVectorEnv(
            [
                make_env_atari(
                    env,
                    seed=cust_seed,
                    rgb=True,
                    stack=4,
                    no_op=0,
                    action_repeat=4,
                    max_frames=False,
                    episodic_life=True,
                    clip_reward=False,
                    idx=i,
                    capture_video=False,
                    run_name="test",
                )
                # make_env_atari(env, stack, env_seed, i, capture_video=False, run_name="test")
                for i in range(1)
            ]
        )
    return nv


from collections import namedtuple


def test_get_trajectory(env):
    policy_algo = "ppo"
    # env.seed(cust_seed)
    trajectory = dict()

    obs = env.reset()
    track = env.track
    trajectory[0] = (
        env.car.hull.position[0],
        env.car.hull.position[1],
        env.car.hull.angle,
    )

    score = 0
    max_ep_score = 0
    # run 1000 steps
    i = 0
    done_testing = False
    while not done_testing:
        i += 1
        obs = torch.as_tensor([obs], device=device)
        with torch.no_grad():
            if policy_algo == "ppo":
                action, logprob, _, value = agent.get_action_and_value_deterministic(
                    obs
                )
            elif policy_algo == "ddqn":
                action = agent(obs).argmax(dim=1, keepdim=True)[0]
            next_obs, reward, done, info = env.step(action[0].cpu().numpy())
        score += reward
        if score > max_ep_score:
            max_ep_score = score
        obs = next_obs
        trajectory[i] = (
            env.car.hull.position[0],
            env.car.hull.position[1],
            env.car.hull.angle,
        )
        done_testing = done
        # for it_idx, item in enumerate(info):
        #     if "episode" in item.keys():
        #         # print(score)
        #         score = item["episode"]["r"]
        #         done_testing = True
        #         break

    print(f"episode done, score: {score} - Episode Length: {i} steps")

    return score, max_ep_score, i, track, trajectory


# parse args
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-test", default=True, type=bool, help="Perform a single test or an extensive stitching test")
    parser.add_argument("--stitching-mode", default="absolute", type=str, help="Stitching mode: absolute, relative, translate")
    parser.add_argument("--env-id", default="CarRacing-v2", type=str, help="Environment ID")
    parser.add_argument("--env-seed", default=0, type=int, help="Environment seed")
    parser.add_argument("--background-color", default="green", type=str, help="Background color of the environment")
    # mandatory args
    parser.add_argument("--encoder-dir", default=None, type=str, help="Path to the encoder model to test", required=True)
    parser.add_argument("--policy-dir", default=None, type=str, help="Path to the policy model to test", required=True)

    # args mandatory only if stitching mode is translate
    parser.add_argument("--anchors-file1", default=None, type=str, help="Path to the anchors to use for stitching")
    parser.add_argument("--anchors-file2", default=None, type=str, help="Path to the anchors to use for stitching")
    parser.add_argument("--anchors-alpha", default=None, type=str, help="Alpha value to use for anchors")
    parser.add_argument("--anchors-method", default="fps", type=str, help="Method to use for anchors: fps, kmeans, random")


    parser.add_argument("--render-mode", default="human", type=str, help="Render mode: human, rgb_array")

    args = parser.parse_args()
    return args


""" CARRACING """
""" abs """
# python stitching_test_end_to_end.py --single-test True --stitching-mode absolute --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1
""" transl """
# python stitching_test_end_to_end.py --single-test True --stitching-mode translate --env-id CarRacing-v2 --env-seed 0 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_40 --policy-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_40 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" no_noop_4as """
# python stitching_test_end_to_end.py --single-test True --stitching-mode translate --env-id CarRacing-v2-no_noop_4as --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-no_noop_4as/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" scrambled """
# python stitching_test_end_to_end.py --single-test True --stitching-mode translate --env-id CarRacing-v2-scrambled --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-scrambled/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" onlyleft """
# python stitching_test_end_to_end.py --single-test True --stitching-mode translate --env-id CarRacing-v2-onlyleft --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-onlyleft/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human
""" heavy """
# python stitching_test_end_to_end.py --single-test True --stitching-mode translate --env-id CarRacing-v2-heavy --env-seed 0 --background-color red --encoder-dir models/CarRacing-v2/rgb/red/ppo/absolute/relu/seed_41 --policy-dir models/CarRacing-v2-heavy/rgb/green/ppo/absolute/relu/seed_0 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-alpha None --anchors-method fps --render-mode human


args = parse_args()
if args.stitching_mode == "translate":
    assert (
        args.anchors_file1 is not None and args.anchors_file2 is not None
    ), "Anchors file must be specified for stitching mode translate"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Single test allows to choose a single combination of encoder and policy to test.
If False, perform an extensive stitching test over all combinations of encoder and policy colors. """

single_test = args.single_test
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

if single_test:
    if env_info == "rgb":
        encoder_instance, policy_instance, agent_instance = get_algo_instance(
            model_algo_1, model_algo_2
        )
    else:
        encoder_instance, policy_instance, agent_instance = get_algo_instance_bw(
            model_algo_1, model_algo_2
        )
    # envs = init_env(env_id, background_color=args.background_color, image_path=image_path, cust_seed=args.env_seed, render_md=render_md)
    car_mode = env_id.split("-")[-1]
    if car_mode == "slow":
        from car_racing_slow import CarRacing
    elif car_mode == "no_noop":
        from car_racing_nonoop import CarRacing
    elif car_mode == "no_noop_4as":
        from car_racing_nonoop_4as import CarRacing
    elif car_mode == "scrambled":
        from car_racing_scrambled import CarRacing
    elif car_mode == "onlyleft":
        from car_racing_noleft import CarRacing
    elif car_mode == "heavy":
        from car_racing_heavy import CarRacing
    else:
        from car_racing import CarRacing
    env = CarRacing(
        continuous=False,
        background=args.background_color,
        image_path=image_path,
        render_mode="human",
    )
    env = make_env_atari(
        env,
        seed=cust_seed,
        rgb=True,
        stack=4,
        no_op=0,
        action_repeat=0,
        max_frames=False,
        episodic_life=False,
        clip_reward=True,
        check_fire=False,
        idx=0,
        capture_video=False,
        run_name="test",
    )()

    # envs = env
    path1_enc = os.path.join(args.encoder_dir, "encoder.pt")
    path1_pol = os.path.join(args.encoder_dir, "policy.pt")
    path2_enc = os.path.join(args.policy_dir, "encoder.pt")
    path2_pol = os.path.join(args.policy_dir, "policy.pt")

    encoder1 = load_encoder_from_path(
        path1_enc,
        encoder_instance,
        is_relative=False,
        is_pretrained=False,
        anchors_alpha=None,
        encoder_eval=True,
        device=device,
    )
    encoder2, policy2, agent2 = load_model_from_path(
        path2_enc,
        path2_pol,
        env.action_space.n,
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
        space1 = Space(name="encoder_model", vectors=space1)
        space2 = Space(name="policy_model", vectors=space2)

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

        align_path = os.path.join("alignment_indices", args.encoder_dir)
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
            "fitting translation layer between encoder and policy models spaces..."  # {model_color_1} and {model_color_2} spaces..."
        )
        translation.fit(source_data=space1, target_data=space2)
        print("done.\n\n")
        print(translation(space1))
    # agent = Agent(encoder1, policy2, translation=translation).to(device)
    agent = Agent(encoder1, policy2, translation=translation).to(device)


driving_model = str(args.policy_dir).replace("/", "_").replace("models_", "")
trajectories_path = f"experiments/trajectories/{env_id}/{stitching_md}/{driving_model}"
if not os.path.exists(trajectories_path):
    os.makedirs(trajectories_path)
if single_test:
    score, max_ep_score, i, track, trajectory = test_get_trajectory(env)
    print(
        f"Score: {score} - Max Episode Score: {max_ep_score} - Episode Length: {i} steps"
    )
    # save track to csv
    track_filename = f"{trajectories_path}/track_{cust_seed}.csv"
    # write heading on top of track_filename, those are: angle(rad), smoothing, x, y
    with open(track_filename, "w") as f:
        f.write("angle(rad),smoothing,x,y\n")
    # save list of tuples to csv
    with open(track_filename, "a") as f:
        for item in track:
            f.write(f"{item[0]},{item[1]},{item[2]},{item[3]}\n")
    print(f"Saved track to {track_filename}")
    enc_seed = args.encoder_dir.split("_")[-1]
    # save trajectory to csv
    trajectory_filename = f"{trajectories_path}/trajectory_encseed_{enc_seed}_track_{args.background_color}{cust_seed}.csv"
    trajectory_df = pd.DataFrame.from_dict(
        trajectory, orient="index", columns=["x", "y", "angle"]
    )
    trajectory_df.to_csv(trajectory_filename, index=False)
    # test_rel_repr_vec(envs, agent, policy_algo=model_algo_2, limit_episode_length=4000, device=device)
    exit(0)
# else:
#     from utils.testing import stitching_test_relative, stitching_test_alignment

#     if stitching_md == "relative":
#         use_relative = True
#         results = stitching_test_relative(
#             env_id,
#             env_info,
#             encoder_algo=model_algo_1,
#             policy_algo=model_algo_2,
#             encoder_model_type=model_activation_1,
#             policy_model_type=model_activation_2,
#             is_relative=use_relative,
#             is_pretrained=False,
#             swap_anchors=False,
#             anchors_path=None,
#             anchors_alpha=model_alpha_1,
#             render_mode="rgb_array",
#             device=device,
#             env_seeds_totest=env_seeds_totest,
#         )
#     elif stitching_md == "absolute":
#         use_relative = False
#         results = stitching_test_relative(
#             env_id,
#             env_info,
#             encoder_algo=model_algo_1,
#             policy_algo=model_algo_2,
#             encoder_model_type=model_activation_1,
#             policy_model_type=model_activation_2,
#             is_relative=use_relative,
#             is_pretrained=False,
#             swap_anchors=False,
#             anchors_path=None,
#             anchors_alpha=model_alpha_1,
#             render_mode="rgb_array",
#             device=device,
#             env_seeds_totest=env_seeds_totest,
#         )
#     elif stitching_md == "translate":
#         use_relative = False
#         results = stitching_test_alignment(
#             env_id,
#             env_info,
#             anchoring_method,
#             encoder_algo=model_algo_1,
#             policy_algo=model_algo_2,
#             encoder_model_type=model_activation_1,
#             policy_model_type=model_activation_2,
#             is_relative=use_relative,
#             is_pretrained=False,
#             swap_anchors=False,
#             anchors_path=None,
#             anchors_alpha=model_alpha_1,
#             render_mode="rgb_array",
#             device=device,
#             env_seeds_totest=env_seeds_totest,
#         )

#     # is_relative = "relative" if use_relative else "absolute"

#     # check if folder exists, if not create it, recursively
#     if not os.path.exists(f"experiments/stitching_tests/{env_id}/{env_info}/{stitching_md}"):
#         os.makedirs(f"experiments/stitching_tests/{env_id}/{env_info}/{stitching_md}")

#     stitch_filename = f"experiments/stitching_tests/{env_id}/{env_info}/{stitching_md}/"
#     if stitching_md == "translate":
#         stitch_filename += f"{anchoring_method}/"
#         if not os.path.exists(stitch_filename):
#             os.makedirs(stitch_filename)
#     stitch_filename += f"{model_activation_2}_stitching_results_{model_algo_2}.csv"


#     # # save results to csv
#     results.to_csv(stitch_filename, index=False)
#     print(f"Saved stitching results to {stitch_filename}")
