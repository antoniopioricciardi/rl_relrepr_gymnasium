import torch
import pandas as pd
import numpy as np

# from natural_rl_environment.natural_env import NaturalEnvWrapper
from zeroshotrl.utils.env_initializer import init_env
from zeroshotrl.utils.evaluation import evaluate_vec_env
# TODO: MOVE THEM TO A COMMON FILE

# from utils.preprocess_env import PreprocessFrameRGB

# def make_env_atari(env, seed, idx, capture_video, run_name):
#     def thunk(env=env):
#         # env = gym.make(env_id)
#         # env = CarRacing(continuous=False, background='red')
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         # env = NoopResetEnv(env, noop_max=30)
#         # env = MaxAndSkipEnv(env, skip=4)
#         # env = EpisodicLifeEnv(env)
#         # if "FIRE" in env.unwrapped.get_action_meanings():
#         #     env = FireResetEnv(env)
#         # env = ClipRewardEnv(env)
#         # env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = PreprocessFrameRGB((84, 84, 3), env) # (3, 84, 84)
#         # env = gym.wrappers.GrayScaleObservation(env)
#         # env = gym.wrappers.FrameStack(env, 4) #(4, 3, 84, 84)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


# def make_env_carracing(env, seed, idx, capture_video, run_name):
#     def thunk(env=env):
#         # env = gym.make(env_id)
#         # env = CarRacing(continuous=False, background='red')
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env = PreprocessFrameRGB((84, 84, 3), env) # (3, 84, 84)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


def test_rel_repr_vec(
    env, agent, policy_algo, limit_episode_length=-1, device="cpu", seed=0, num_envs=1, forced_render=False
):
    eval_rewards, eval_lengths, eval_avg_reward = evaluate_vec_env(
        agent=agent,
        num_envs=num_envs,
        env=env,
        global_step=0,
        device=device,
        episode_n=0,
        writer=None,
        logger=None,
        algorithm=policy_algo,
        episode_length_limit=limit_episode_length,
        seed=seed,
        forced_render=forced_render,
    )
    print(
        f"episode done, score: {eval_rewards}, avg: {eval_avg_reward} - Episode Length: {eval_lengths} steps"
    )

    return eval_avg_reward, eval_lengths


def stitching_test_quantitative(
    enc_env_id,
    playing_env_id,
    env_info="rgb",
    playon: str = "policy",
    background: str = None,
    encoder_backgrounds: list[str] = None,
    policy_backgrounds: list[str] = None,
    env_seeds: list[int] = [
        0,
        1,
        2,
        3,
        4,
    ],  # encoder_dir: str = None, policy_dir: str = None,
    encoder_seeds: list[int] = [0, 1, 2, 3, 4],
    policy_seeds: list[int] = [0, 1, 2, 3, 4],
    encoder_anchors: str = None,
    controller_anchors: str = None,
    encoder_algo="ppo",
    policy_algo="ppo",
    encoder_activation_func="relu",
    policy_activation_func="relu",
    anchors_alpha=0,
    zoom=2.7,
    stitching_mode="absolute",
    anchoring_method=None,
    render_mode="rgb_array",
    device="cpu",
):
    """
    backgrounds: Non mandatory background color to test the encoder on. If not provided, encoder_backgrounds is used as env background
    encoder_backgrounds: list of background colors to test the encoder on. Envs background are chosen from this list
    policy_backgrounds: list of background colors to test the policy on. Envs background are chosen from this list
    """
    assert encoder_backgrounds is not None, "encoder_backgrounds must be provided"
    assert policy_backgrounds is not None, "policy_backgrounds to test must be provided"
    # assert encoder_dir is not None, "encoder_dir must be provided"
    # assert policy_dir is not None, "policy_dir must be provided"
    assert stitching_mode in [
        "absolute",
        "translate",
        "relative",
    ], "stitching_mode must be one of ['absolute', 'translate', 'relative']"
    if stitching_mode == "translate":
        assert anchoring_method is not None, "anchoring_method must be provided"
        assert encoder_anchors is not None, "encoder_anchors must be provided"
        assert controller_anchors is not None, "controller_anchors must be provided"

    from latentis.space import LatentSpace
    from latentis.estimate.dim_matcher import ZeroPadding
    from latentis.estimate.orthogonal import SVDEstimator
    from latentis.translate.translator import LatentTranslator

    from pathlib import Path
    from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent
    import pickle
    import time
    import os

    from zeroshotrl.utils.models import (
        get_algo_instance,
        load_encoder_from_path,
        load_model_from_path,
    )

    # create a pandas dataframe to store the results
    df = pd.DataFrame(
        columns=[
            "env_seed",
            "encoder_background",
            "policy_background",
            "encoder_seed",
            "policy_seed",
            "encoder_env",
            "policy_env",
            "score",
            "episode_length",
            "algorithm",
            "clustering_time",
        ]
    )

    relative = stitching_mode == "relative"

    use_enc_bg = False
    if background is None:
        use_enc_bg = True

    for enc_bg in encoder_backgrounds:
        for pol_bg in policy_backgrounds:
            for enc_seed in encoder_seeds:
                for pol_seed in policy_seeds:
                    # print('----------------------------------')
                    # print('encoder seed:', enc_seed, 'policy seed:', pol_seed, 'encoder bg:', enc_bg, 'policy bg:', pol_bg)

                    clustering_time = 0
                    path1 = os.path.join(
                        enc_env_id,
                        env_info,
                        enc_bg,
                        encoder_algo,
                        "absolute",
                        encoder_activation_func,
                        f"seed_{enc_seed}",
                    )
                    path2 = os.path.join(
                        playing_env_id,
                        env_info,
                        pol_bg,
                        policy_algo,
                        "absolute",
                        policy_activation_func,
                        f"seed_{pol_seed}",
                    )
                    if stitching_mode == "relative":
                        a_alpha = str(anchors_alpha).replace(".", "_")
                        path1 = os.path.join(
                            enc_env_id,
                            env_info,
                            enc_bg,
                            encoder_algo,
                            "relative",
                            encoder_activation_func,
                            f"alpha_{a_alpha}",
                            f"seed_{enc_seed}",
                        )
                        path2 = os.path.join(
                            playing_env_id,
                            env_info,
                            pol_bg,
                            policy_algo,
                            "relative",
                            policy_activation_func,
                            f"alpha_{a_alpha}",
                            f"seed_{pol_seed}",
                        )

                    path1_enc = os.path.join("models", path1, "encoder.pt")
                    # path1_pol = os.path.join(path1, "policy.pt")
                    path2_enc = os.path.join("models", path2, "encoder.pt")
                    path2_pol = os.path.join("models", path2, "policy.pt")
                    if env_info == "rgb":
                        encoder_instance, policy_instance, agent_instance = (
                            get_algo_instance(
                                encoder_algo=encoder_algo,
                                policy_algo=policy_algo,
                                use_resnet=False,
                            )
                        )
                    # TODO: use bool "models_initialized" and move this part outside the loop to remove duplicate init_env
                    env_controller = init_env(
                        playing_env_id if playon == "policy" else enc_env_id,
                        env_info,
                        background_color=enc_bg,
                        image_path="",
                        zoom=zoom,
                        cust_seed=0,
                        render_md=render_mode,
                    )
                    encoder1 = load_encoder_from_path(
                        path1_enc,
                        encoder_instance,
                        is_relative=relative,
                        is_pretrained=False,
                        anchors_alpha=None,
                        encoder_eval=True,
                        device=device,
                    )
                    encoder2, policy2, agent2 = load_model_from_path(
                        path2_enc,
                        path2_pol,
                        env_controller.single_action_space.n,
                        encoder_instance,
                        policy_instance,
                        agent_instance,
                        is_relative=relative,
                        is_pretrained=False,
                        device=device,
                    )

                    translation = None
                    if stitching_mode == "translate":
                        anchors_dir = "data/anchors"
                        anchors_file1 = os.path.join(
                            anchors_dir,
                            encoder_anchors,
                            f"rgb_ppo_transitions_{enc_bg}_obs.pkl",
                        )
                        # TODO: anchors_file2 could use playing_env_id if visual variations are more than background colors
                        # however we need anchors
                        anchors_file2 = os.path.join(
                            anchors_dir,
                            controller_anchors,
                            f"rgb_ppo_transitions_{pol_bg}_obs.pkl",
                        )
                        print(
                            "anchor files:\n",
                            anchors_file1,
                            "\n",
                            anchors_file2,
                            "\n\n",
                        )
                        obs_set_1 = pickle.load(
                            Path(
                                # f"data/anchors/{env_id}/{env_info}_ppo_transitions_{model_color_1}_obs.pkl"
                                anchors_file1
                            ).open("rb")
                        )  # [30:2000]
                        obs_set_2 = pickle.load(
                            Path(
                                # f"data/anchors/{env_id}/{env_info}_ppo_transitions_{model_color_2}_obs.pkl"
                                anchors_file2
                            ).open("rb")
                        )  # [30:2000]
                        print("\n#####\nObs loaded\n#####\n")
                        # subset_indices = np.random.randint(0, len(obs_set_1), 5000)
                        obs_set_1 = obs_set_1  # [:4000]
                        obs_set_2 = obs_set_2  # [:4000]

                        print("Converting obs to torch tensor")
                        # convert the (4000, 3, 84, 84) numpy array to a torch tensor
                        obs_set_1 = torch.tensor(
                            np.array(obs_set_1), dtype=torch.float32
                        )
                        obs_set_2 = torch.tensor(
                            np.array(obs_set_2), dtype=torch.float32
                        )
                        print("Done converting obs to torch tensor\n#####\n")

                        # obs_set_1 = torch.cat([obs_set_1, obs_set_2], dim=0)  # [anch_indices
                        # obs_set_2 = obs_set_1

                        subset_indices = np.arange(len(obs_set_1))  # [:4000]

                        space1 = (
                            encoder1.forward_single(obs_set_1.to(device)).detach().cpu()
                        )
                        space2 = (
                            encoder2.forward_single(obs_set_2.to(device)).detach().cpu()
                        )

                        # print('AAAAA', obs_set_1.shape, obs_set_2.shape, space1.shape, space2.shape)

                        from collections import namedtuple

                        Space = namedtuple("Space", ["name", "vectors"])
                        space1 = Space(name=enc_bg, vectors=space1)
                        space2 = Space(name=pol_bg, vectors=space2)

                        space1_vectors = space1.vectors
                        space2_vectors = space2.vectors

                        space1_anchors = space1_vectors[:]
                        space2_anchors = space2_vectors[:]

                        from utils.anchoring_methods import get_anchors

                        """ CHANGE ANCHOR SAMPLING METHOD HERE """
                        # if not os.path.exists(f"alignment_indices/{env_id}/{env_info}"):
                        #     os.makedirs(f"alignment_indices/{env_id}/{env_info}")
                        # translation_path = f'alignment_indices/{env_id}/{env_info}/{anchoring_method}_{model_color_1}_{model_seed_1}_closest.pt'#{model_color_2}_closest.pt'

                        # align_path = os.path.join('alignment_indices', str(encoder_dir).replace('models/', ''))
                        align_path = os.path.join(
                            "alignment_indices", str(path1).replace("models/", "")
                        )
                        if not os.path.exists(align_path):
                            os.makedirs(align_path)
                        translation_path = os.path.join(
                            align_path, f"{anchoring_method}_closest.pt"
                        )

                        num_anchors = 3136  # len(space1_anchors) # 3136
                        start = time.time()
                        torch.manual_seed(42)
                        space1_anchors, space2_anchors = get_anchors(
                            space1_vectors,
                            space2_vectors,
                            num_anchors,
                            subset_indices,
                            anchoring_method,
                            translation_path,
                            device,
                        )
                        clustering_time = round(time.time() - start, 3)
                        print(
                            f"{anchoring_method} done in {clustering_time} seconds.\n\n"
                        )

                        translation = LatentTranslator(
                            random_seed=42,
                            estimator=SVDEstimator(
                                dim_matcher=ZeroPadding()
                            ),  # SGDAffineTranslator(),#SVDEstimator(dim_matcher=ZeroPadding()),
                            source_transforms=None,  # [transforms.StandardScaling()],
                            target_transforms=None,  # [transforms.StandardScaling()],
                        )
                        space1_anchors = space1_anchors.to(device)  # [:3136]
                        space2_anchors = space2_anchors.to(device)  # [:3136]
                        space1 = LatentSpace(vectors=space1_anchors, name="space1")
                        space2 = LatentSpace(vectors=space2_anchors, name="space2")
                        print("\n##############################################\n")
                        print(
                            f"fitting translation layer between {enc_bg} and {pol_bg} spaces..."
                        )
                        translation.fit(source_data=space1, target_data=space2)
                        print("done.\n\n")
                        print(translation(space1))
                    # agent = Agent(encoder1, policy2, translation=translation).to(device)
                    agent = Agent(encoder1, policy2, translation=translation).to(device)

                    for i in env_seeds:
                        if use_enc_bg:
                            background = enc_bg
                        models_initialized = False
                        cust_seed = i
                        print("----------------------------------")
                        print(
                            f"Testing on {background} background with {enc_bg} encoder (seed {enc_seed}) and {pol_bg} policy (seed {pol_seed}). Environment seed {cust_seed}"
                        )

                        limit_episode_length = -1
                        if playing_env_id.startswith("Breakout"):
                            limit_episode_length = 4000
                        env_controller = init_env(
                            playing_env_id if playon == "policy" else enc_env_id,
                            env_info,
                            background_color=background,
                            image_path="",
                            zoom=zoom,
                            cust_seed=cust_seed,
                            render_md=render_mode,
                        )

                        score, ep_length = test_rel_repr_vec(
                            env_controller,
                            agent,
                            policy_algo,
                            limit_episode_length,
                            device=device,
                        )  # , cust_seed=1)

                        # score, max_ep_score, ep_length = test_rel_repr_vec(env_controller, agent, policy_algo, limit_episode_length, device=device)#, cust_seed=1)
                        print(f"Episode finished: {score} points, {ep_length} steps")
                        print(clustering_time)
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        "env_seed": i,
                                        "encoder_background": enc_bg,
                                        "policy_background": pol_bg,
                                        "encoder_seed": enc_seed,
                                        "policy_seed": pol_seed,
                                        "encoder_env": enc_env_id,
                                        "policy_env": playing_env_id,
                                        "score": score,
                                        "episode_length": ep_length,
                                        "algorithm": "ppo",
                                        "clustering_time": clustering_time,
                                    },
                                    index=[0],
                                ),
                            ]
                        )
    # print a recap of the results, along with the max score and max episode length, average score and average episode length over all seeds
    print(df)
    return df