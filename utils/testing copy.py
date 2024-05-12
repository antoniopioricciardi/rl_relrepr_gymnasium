import gym
import torch
import pandas as pd
import numpy as np
# from natural_rl_environment.natural_env import NaturalEnvWrapper
from envs.carracing.car_racing import CarRacing
# from utils.alignment import LatentTranslation, IdentityTranslator, AffineTranslator, LSTSQTranslator, LSTSQOrthoTranslator, SVDTranslator
# from latentis import transforms
from utils.env_initializer import instantiate_env, init_env
from pytorch_lightning import seed_everything

# TODO: MOVE THEM TO A COMMON FILE

from utils.preprocess_env import PreprocessFrameRGB
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env_atari(env, seed, idx, capture_video, run_name):
    def thunk(env=env):
        # env = gym.make(env_id)
        # env = CarRacing(continuous=False, background='red')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = PreprocessFrameRGB((84, 84, 3), env) # (3, 84, 84)
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4) #(4, 3, 84, 84)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_env_carracing(env, seed, idx, capture_video, run_name):
    def thunk(env=env):
        # env = gym.make(env_id)
        # env = CarRacing(continuous=False, background='red')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = PreprocessFrameRGB((84, 84, 3), env) # (3, 84, 84)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk



def test_rel_repr_vec(env, agent, policy_algo, limit_episode_length=-1, device='cpu'):
    envs = env

    # env.seed(cust_seed)
    obs, _ = envs.reset()
    score = 0
    max_ep_score = 0
    # run 1000 steps
    i = 0
    done_testing = False
    while not done_testing:
        i += 1
        obs = torch.as_tensor(obs, device=device)
        with torch.no_grad():
            if policy_algo == "ppo":
                action, logprob, _, value = agent.get_action_and_value_deterministic(obs)
            elif policy_algo == "ddqn":
                action = agent(obs).argmax(dim=1, keepdim=True)[0]
            # next_obs, reward, done, info = envs.step(action.cpu().numpy())
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())#.cpu().numpy())
            done = np.logical_or(terminated, truncated)
        score += reward[0]
        if score > max_ep_score:
            max_ep_score = score
        obs = next_obs

        # if episodic_life:    
        for it_idx, item in enumerate(info):
            if "episode" in item.keys():
                # print(score)
                score = item["episode"]["r"]
                done_testing = True
                break
        
        if limit_episode_length > 0:
            if i > limit_episode_length:
                done_testing = True
                break
    
    print(f'episode done, score: {score} - Episode Length: {i} steps' )

    return score, max_ep_score, i



def stitching_test_quantitative(
        enc_env_id, playing_env_id, env_info='rgb', playon: str = 'policy',
        backgrounds: list[str] = None, policy_backgrounds: list[str] = None,
        env_seeds: list[int] = [0, 1, 2, 3, 4], #encoder_dir: str = None, policy_dir: str = None,
        encoder_seeds: list[int] = [0, 1, 2, 3, 4], policy_seeds: list[int] = [0, 1, 2, 3, 4],
        encoder_anchors: str = None, controller_anchors: str = None,
        encoder_algo="ppo", policy_algo="ppo", encoder_activation_func="relu", policy_activation_func="relu", anchors_alpha=0,
        zoom=2.7, stitching_mode='absolute', anchoring_method=None, render_mode="rgb_array", device='cpu' 
        ):
    assert backgrounds is not None, "backgrounds must be provided"
    assert policy_backgrounds is not None, "policy_backgrounds to test must be provided"
    # assert encoder_dir is not None, "encoder_dir must be provided"
    # assert policy_dir is not None, "policy_dir must be provided"
    assert stitching_mode in ['absolute', 'translate', 'relative'], "stitching_mode must be one of ['absolute', 'translate', 'relative']"
    if stitching_mode == 'translate':
        assert anchoring_method is not None, "anchoring_method must be provided"
        assert encoder_anchors is not None, "encoder_anchors must be provided"
        assert controller_anchors is not None, "controller_anchors must be provided"

    from latentis.space import LatentSpace
    from latentis.utils import seed_everything
    from latentis.estimate.dim_matcher import ZeroPadding
    from latentis.estimate.orthogonal import SVDEstimator
    from latentis.translate.translator import LatentTranslator
    
    from pathlib import Path
    from rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent
    import pickle
    import time
    import numpy as np
    import os

    from utils.models import load_model, get_algo_instance, load_encoder_from_path, load_model_from_path
    # create a pandas dataframe to store the results
    df = pd.DataFrame(columns=["env_seed", "encoder_background", "policy_background",
                               "encoder_seed", "policy_seed", "encoder_env", "policy_env",
                               "score", "max_score_reached", "episode_length", "algorithm", "clustering_time"])
    
    relative = (stitching_mode=='relative')
    for enc_bg in backgrounds:
        for pol_bg in policy_backgrounds:
            for enc_seed in encoder_seeds:
                for pol_seed in policy_seeds:
                    # print('----------------------------------')
                    # print('encoder seed:', enc_seed, 'policy seed:', pol_seed, 'encoder bg:', enc_bg, 'policy bg:', pol_bg)

                    clustering_time = 0
                    path1 = os.path.join(enc_env_id, env_info, enc_bg, encoder_algo, 'absolute', encoder_activation_func, f'seed_{enc_seed}')
                    path2 = os.path.join(playing_env_id, env_info, pol_bg, policy_algo, 'absolute', policy_activation_func, f'seed_{pol_seed}')
                    if stitching_mode == "relative":
                        a_alpha = str(anchors_alpha).replace('.', '_')
                        path1 = os.path.join(enc_env_id, env_info, enc_bg, encoder_algo, 'relative', encoder_activation_func, f'alpha_{a_alpha}', f'seed_{enc_seed}')
                        path2 = os.path.join(playing_env_id, env_info, pol_bg, policy_algo, 'relative', policy_activation_func, f'alpha_{a_alpha}', f'seed_{pol_seed}')

                    path1_enc = os.path.join('models', path1, "encoder.pt")
                    # path1_pol = os.path.join(path1, "policy.pt")
                    path2_enc = os.path.join('models', path2, "encoder.pt")
                    path2_pol = os.path.join('models', path2, "policy.pt")
                    if env_info == "rgb":
                        encoder_instance, policy_instance, agent_instance = get_algo_instance(
                            encoder_algo=encoder_algo, policy_algo=policy_algo
                        )
                    # TODO: use bool "models_initialized" and move this part outside the loop to remove duplicate init_env
                    env_controller = init_env(playing_env_id if playon=='policy' else enc_env_id, env_info, background_color=enc_bg, image_path='', zoom=zoom, cust_seed=0, render_md=render_mode)
                    encoder1 = load_encoder_from_path(
                        path1_enc, encoder_instance, is_relative=relative, is_pretrained=False, anchors_alpha=None, encoder_eval=True, device=device
                        )    
                    encoder2, policy2, agent2 = load_model_from_path(
                        path2_enc, path2_pol, env_controller.single_action_space.n,
                        encoder_instance, policy_instance, agent_instance, is_relative=relative, is_pretrained=False, device=device
                        )

                    translation = None
                    if stitching_mode == "translate":
                        anchors_dir = "data/anchors"
                        anchors_file1 = os.path.join(anchors_dir, encoder_anchors, f"rgb_ppo_transitions_{enc_bg}_obs.pkl")
                        # TODO: anchors_file2 could use playing_env_id if visual variations are more than background colors
                        # however we need anchors
                        anchors_file2 = os.path.join(anchors_dir, controller_anchors, f"rgb_ppo_transitions_{pol_bg}_obs.pkl")
                        print('anchor files:\n', anchors_file1, '\n', anchors_file2, '\n\n')
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
                        print('\n#####\nObs loaded\n#####\n')
                        # subset_indices = np.random.randint(0, len(obs_set_1), 5000)
                        obs_set_1 = obs_set_1#[:4000]
                        obs_set_2 = obs_set_2#[:4000]

                        print('Converting obs to torch tensor')
                        # convert the (4000, 3, 84, 84) numpy array to a torch tensor
                        obs_set_1 = torch.tensor(np.array(obs_set_1), dtype=torch.float32)
                        obs_set_2 = torch.tensor(np.array(obs_set_2), dtype=torch.float32)
                        print('Done converting obs to torch tensor\n#####\n')

                        # obs_set_1 = torch.cat([obs_set_1, obs_set_2], dim=0)  # [anch_indices
                        # obs_set_2 = obs_set_1

                        subset_indices = np.arange(len(obs_set_1))#[:4000]

                        space1 = encoder1.forward_single(obs_set_1.to(device)).detach().cpu()
                        space2 = encoder2.forward_single(obs_set_2.to(device)).detach().cpu()

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
                        align_path = os.path.join('alignment_indices', str(path1).replace('models/', ''))
                        if not os.path.exists(align_path):
                            os.makedirs(align_path)
                        translation_path = os.path.join(align_path, f'{anchoring_method}_closest.pt')

                        num_anchors = 3136 # len(space1_anchors) # 3136
                        start = time.time()
                        torch.manual_seed(42)
                        space1_anchors, space2_anchors = get_anchors(space1_vectors, space2_vectors, num_anchors, subset_indices, anchoring_method, translation_path, device)
                        clustering_time = round(time.time() - start, 3)
                        print(f"{anchoring_method} done in {clustering_time} seconds.\n\n")

                        translation = LatentTranslator(
                            random_seed=42,
                            estimator=SVDEstimator(dim_matcher=ZeroPadding()),# SGDAffineTranslator(),#SVDEstimator(dim_matcher=ZeroPadding()),
                            source_transforms=None, #[transforms.StandardScaling()],
                            target_transforms=None, #[transforms.StandardScaling()],
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
                        models_initialized = False
                        cust_seed = i
                        print('----------------------------------')
                        print(f"Testing on {enc_bg} background with {enc_bg} encoder (seed {enc_seed}) and {pol_bg} policy (seed {pol_seed}). Environment seed {cust_seed}")
                        
                        limit_episode_length = -1
                        if playing_env_id.startswith("Breakout"):
                            limit_episode_length = 4000      
                        env_controller = init_env(playing_env_id if playon=='policy' else enc_env_id, env_info, background_color=enc_bg, image_path='', zoom=zoom, cust_seed=cust_seed, render_md=render_mode)                        
                        
                        score, max_ep_score, ep_length = test_rel_repr_vec(env_controller, agent, policy_algo, limit_episode_length, device=device)#, cust_seed=1)

                        # score, max_ep_score, ep_length = test_rel_repr_vec(env_controller, agent, policy_algo, limit_episode_length, device=device)#, cust_seed=1)
                        print(f"Episode finished: {score} points, {ep_length} steps")
                        print(clustering_time)
                        
                        df = df.append({"env_seed": i, "encoder_background": enc_bg, "policy_background": pol_bg,
                                        "encoder_seed": enc_seed, "policy_seed": pol_seed,
                                        "encoder_env": enc_env_id, "policy_env": playing_env_id,
                                        "score": score, "max_score_reached": max_ep_score, "episode_length": ep_length,
                                        "algorithm": "ppo", "clustering_time": clustering_time}, ignore_index=True)
    # print a recap of the results, along with the max score and max episode length, average score and average episode length over all seeds
    print(df)
    return df





# def stitching_test_relative(
#         enc_env_id, playing_env_id, env_info='rgb', playon: str = 'policy',
#         backgrounds: list[str] = None, policy_backgrounds: list[str] = None,
#         env_seeds: list[int] = [0, 1, 2, 3, 4], #encoder_dir: str = None, policy_dir: str = None,
#         encoder_seeds: list[int] = [0, 1, 2, 3, 4], policy_seeds: list[int] = [0, 1, 2, 3, 4],
#         encoder_anchors: str = None, controller_anchors: str = None,
#         encoder_algo="ppo", policy_algo="ppo", encoder_activation_func="relu", policy_activation_func="relu", enc_alpha=0, pol_alpha=0,
#         zoom=2.7, stitching_mode='absolute', anchoring_method=None, render_mode="rgb_array", device='cpu' 
#         ):
#     assert backgrounds is not None, "backgrounds must be provided"
#     assert policy_backgrounds is not None, "policy_backgrounds to test must be provided"
#     # assert encoder_dir is not None, "encoder_dir must be provided"
#     # assert policy_dir is not None, "policy_dir must be provided"
#     assert stitching_mode in ['absolute', 'translate', 'relative'], "stitching_mode must be one of ['absolute', 'translate', 'relative']"
#     if stitching_mode == 'translate':
#         assert anchoring_method is not None, "anchoring_method must be provided"
#         assert encoder_anchors is not None, "encoder_anchors must be provided"
#         assert controller_anchors is not None, "controller_anchors must be provided"

#     from latentis.space import LatentSpace
#     from latentis.utils import seed_everything
#     from latentis.estimate.dim_matcher import ZeroPadding
#     from latentis.estimate.orthogonal import SVDEstimator
#     from latentis.translate.translator import LatentTranslator
    
#     from pathlib import Path
#     from rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent
#     import pickle
#     import time
#     import numpy as np
#     import os

#     from utils.models import load_model, get_algo_instance, load_encoder_from_path, load_model_from_path
#     # create a pandas dataframe to store the results
#     df = pd.DataFrame(columns=["env_seed", "encoder_background", "policy_background",
#                                "encoder_seed", "policy_seed", "encoder_env", "policy_env",
#                                "score", "max_score_reached", "episode_length", "algorithm", "clustering_time"])
    
#     for enc_bg in backgrounds:
#         for pol_bg in policy_backgrounds:
#             for enc_seed in encoder_seeds:
#                 for pol_seed in policy_seeds:
#                     # print('----------------------------------')
#                     # print('encoder seed:', enc_seed, 'policy seed:', pol_seed, 'encoder bg:', enc_bg, 'policy bg:', pol_bg)

#                     clustering_time = 0
#                     # convert enc_alpha and pol_alpha to string, remove the dot
#                     e_a = str(enc_alpha).replace('.', '')
#                     p_a = str(pol_alpha).replace('.', '')
#                     path1 = os.path.join(enc_env_id, env_info, enc_bg, encoder_algo, 'relative', encoder_activation_func, f'alpha_{e_a}', f'seed_{enc_seed}')
#                     path2 = os.path.join(playing_env_id, env_info, pol_bg, policy_algo, 'relative', policy_activation_func, f'alpha_{p_a}', f'seed_{pol_seed}')
#                     # path1 = os.path.join(encoder_dir, env_info, enc_bg, encoder_algo, 'absolute', encoder_model_type, f'seed_{enc_seed}')
#                     # path2 = os.path.join(policy_dir, env_info, pol_bg, encoder_algo, 'absolute', encoder_model_type, f'seed_{enc_seed}')
#                     path1_enc = os.path.join('models', path1, "encoder.pt")
#                     # path1_pol = os.path.join(path1, "policy.pt")
#                     path2_enc = os.path.join('models', path2, "encoder.pt")
#                     path2_pol = os.path.join('models', path2, "policy.pt")
#                     if env_info == "rgb":
#                         encoder_instance, policy_instance, agent_instance = get_algo_instance(
#                             encoder_algo=encoder_algo, policy_algo=policy_algo
#                         )
#                     # TODO: use bool "models_initialized" and move this part outside the loop to remove duplicate init_env
#                     env_controller = init_env(playing_env_id if playon=='policy' else enc_env_id, env_info, background_color=enc_bg, image_path='', zoom=zoom, cust_seed=0, render_md=render_mode)
#                     encoder1 = load_encoder_from_path(
#                         path1_enc, encoder_instance, is_relative=is_relative, is_pretrained=False, anchors_alpha=None, encoder_eval=True, device=device
#                         )    
#                     encoder2, policy2, agent2 = load_model_from_path(
#                         path2_enc, path2_pol, env_controller.single_action_space.n,
#                         encoder_instance, policy_instance, agent_instance, is_relative=is_relative, is_pretrained=False, device=device
#                         )

#                     translation = None
#                     if stitching_mode == "translate":
#                         anchors_dir = "data/anchors"
#                         anchors_file1 = os.path.join(anchors_dir, encoder_anchors, f"rgb_ppo_transitions_{enc_bg}_obs.pkl")
#                         # TODO: anchors_file2 could use playing_env_id if visual variations are more than background colors
#                         # however we need anchors
#                         anchors_file2 = os.path.join(anchors_dir, controller_anchors, f"rgb_ppo_transitions_{pol_bg}_obs.pkl")
#                         print('anchor files:\n', anchors_file1, '\n', anchors_file2, '\n\n')
#                         obs_set_1 = pickle.load(
#                             Path(
#                                 # f"data/anchors/{env_id}/{env_info}_ppo_transitions_{model_color_1}_obs.pkl"
#                                 anchors_file1
#                             ).open("rb")
#                         )  # [30:2000]
#                         obs_set_2 = pickle.load(
#                             Path(
#                                 # f"data/anchors/{env_id}/{env_info}_ppo_transitions_{model_color_2}_obs.pkl"
#                                 anchors_file2
#                             ).open("rb")
#                         )  # [30:2000]
#                         print('\n#####\nObs loaded\n#####\n')
#                         # subset_indices = np.random.randint(0, len(obs_set_1), 5000)
#                         obs_set_1 = obs_set_1#[:4000]
#                         obs_set_2 = obs_set_2#[:4000]

#                         print('Converting obs to torch tensor')
#                         # convert the (4000, 3, 84, 84) numpy array to a torch tensor
#                         obs_set_1 = torch.tensor(np.array(obs_set_1), dtype=torch.float32)
#                         obs_set_2 = torch.tensor(np.array(obs_set_2), dtype=torch.float32)
#                         print('Done converting obs to torch tensor\n#####\n')

#                         # obs_set_1 = torch.cat([obs_set_1, obs_set_2], dim=0)  # [anch_indices
#                         # obs_set_2 = obs_set_1

#                         subset_indices = np.arange(len(obs_set_1))#[:4000]

#                         space1 = encoder1.forward_single(obs_set_1.to(device)).detach().cpu()
#                         space2 = encoder2.forward_single(obs_set_2.to(device)).detach().cpu()

#                         # print('AAAAA', obs_set_1.shape, obs_set_2.shape, space1.shape, space2.shape)

#                         from collections import namedtuple

#                         Space = namedtuple("Space", ["name", "vectors"])
#                         space1 = Space(name=enc_bg, vectors=space1)
#                         space2 = Space(name=pol_bg, vectors=space2)

#                         space1_vectors = space1.vectors
#                         space2_vectors = space2.vectors

#                         space1_anchors = space1_vectors[:]
#                         space2_anchors = space2_vectors[:]

#                         from utils.anchoring_methods import get_anchors
#                         """ CHANGE ANCHOR SAMPLING METHOD HERE """
#                         # if not os.path.exists(f"alignment_indices/{env_id}/{env_info}"):
#                         #     os.makedirs(f"alignment_indices/{env_id}/{env_info}")
#                         # translation_path = f'alignment_indices/{env_id}/{env_info}/{anchoring_method}_{model_color_1}_{model_seed_1}_closest.pt'#{model_color_2}_closest.pt'
                        
#                         # align_path = os.path.join('alignment_indices', str(encoder_dir).replace('models/', ''))
#                         align_path = os.path.join('alignment_indices', str(path1).replace('models/', ''))
#                         if not os.path.exists(align_path):
#                             os.makedirs(align_path)
#                         translation_path = os.path.join(align_path, f'{anchoring_method}_closest.pt')

#                         num_anchors = 3136 # len(space1_anchors) # 3136
#                         start = time.time()
#                         torch.manual_seed(42)
#                         space1_anchors, space2_anchors = get_anchors(space1_vectors, space2_vectors, num_anchors, subset_indices, anchoring_method, translation_path, device)
#                         clustering_time = round(time.time() - start, 3)
#                         print(f"{anchoring_method} done in {clustering_time} seconds.\n\n")

#                         translation = LatentTranslator(
#                             random_seed=42,
#                             estimator=SVDEstimator(dim_matcher=ZeroPadding()),# SGDAffineTranslator(),#SVDEstimator(dim_matcher=ZeroPadding()),
#                             source_transforms=None, #[transforms.StandardScaling()],
#                             target_transforms=None, #[transforms.StandardScaling()],
#                         )
#                         space1_anchors = space1_anchors.to(device)  # [:3136]
#                         space2_anchors = space2_anchors.to(device)  # [:3136]
#                         space1 = LatentSpace(vectors=space1_anchors, name="space1")
#                         space2 = LatentSpace(vectors=space2_anchors, name="space2")
#                         print("\n##############################################\n")
#                         print(
#                             f"fitting translation layer between {enc_bg} and {pol_bg} spaces..."
#                         )
#                         translation.fit(source_data=space1, target_data=space2)
#                         print("done.\n\n")
#                         print(translation(space1))
#                     # agent = Agent(encoder1, policy2, translation=translation).to(device)
#                     agent = Agent(encoder1, policy2, translation=translation).to(device)

#                     for i in env_seeds:
#                         models_initialized = False
#                         cust_seed = i
#                         print('----------------------------------')
#                         print(f"Testing on {enc_bg} background with {enc_bg} encoder (seed {enc_seed}) and {pol_bg} policy (seed {pol_seed}). Environment seed {cust_seed}")
                        
#                         limit_episode_length = -1
#                         if playing_env_id.startswith("Breakout"):
#                             limit_episode_length = 4000      
#                         env_controller = init_env(playing_env_id if playon=='policy' else enc_env_id, env_info, background_color=enc_bg, image_path='', zoom=zoom, cust_seed=cust_seed, render_md=render_mode)                        
                        
#                         score, max_ep_score, ep_length = test_rel_repr_vec(env_controller, agent, policy_algo, limit_episode_length, device=device)#, cust_seed=1)

#                         # score, max_ep_score, ep_length = test_rel_repr_vec(env_controller, agent, policy_algo, limit_episode_length, device=device)#, cust_seed=1)
#                         print(f"Episode finished: {score} points, {ep_length} steps")
#                         print(clustering_time)
                        
#                         df = df.append({"env_seed": i, "encoder_background": enc_bg, "policy_background": pol_bg,
#                                         "encoder_seed": enc_seed, "policy_seed": pol_seed,
#                                         "encoder_env": enc_env_id, "policy_env": playing_env_id,
#                                         "score": score, "max_score_reached": max_ep_score, "episode_length": ep_length,
#                                         "algorithm": "ppo", "clustering_time": clustering_time}, ignore_index=True)
#     # print a recap of the results, along with the max score and max episode length, average score and average episode length over all seeds
#     print(df)
#     return df
















# def stitching_test_relativeOLD(
#         env_id, env_info, encoder_algo="ppo",
#         policy_algo="ppo", encoder_model_type="tanh", policy_model_type="tanh",
#         is_relative=False, is_pretrained=False, swap_anchors=False, anchors_path=None,
#         anchors_alpha="0999", render_mode="rgb_array", device='cpu', env_seeds_totest=[0]
#         ):
    
#     from utils.models import load_model, get_algo_instance

#     # env_pathname = f"{env_id}_{env_info}"
#     if env_id.startswith("CarRacing"): # if env_id == "CarRacing-v2":
#         background_to_test = ["green", "red", "blue", "yellow"]
#         seed_lst_enc = [40, 41, 42, 43]
#         seed_lst_pol = [40, 41, 42, 43]
#     else:
#         background_to_test = ["plain", "green", "red", "blue"] # ["plain", "green", "red", "blue", "yellow"]
#         seed_lst_enc = [9, 0, 1, 2] # [39, 40, 41, 42, 43]
#         seed_lst_pol = [9, 0, 1, 2] # [39, 40, 41, 42, 43]
#     # create a pandas dataframe to store the results
#     df = pd.DataFrame(columns=["env_seed", "encoder_background", "policy_background", "encoder_seed", "policy_seed",
#                                "score", "max_score_reached", "episode_length", "algorithm"])
#     # test all combinations of backgrounds, averaged over 5 seeds. Store avg and max score
#     for _, enc_bg in enumerate(background_to_test):
#         for _, pol_bg in enumerate(background_to_test):
#             for enc_seed in seed_lst_enc:
#                 # enc_seed = seed_lst_enc[enc_idx]
#                 for pol_seed in seed_lst_pol:
#                     # pol_seed = seed_lst_pol[pol_idx]
#                     for i in env_seeds_totest:
#                         cust_seed = i #+40
#                         print('----------------------------------')
#                         print(f"Testing on {enc_bg} background with {enc_bg} encoder (seed {enc_seed}) and {pol_bg} policy (seed {pol_seed}). Seed {cust_seed}")
#                         # print(f"Seed: {i}")
                        
#                         if env_id == "CarRacing-v2":
#                             envs = instantiate_env(
#                                 env_id, num_envs=1, env_variation=enc_bg, env_seed=cust_seed, num_stack=4,
#                                 num_no_op=0, action_repeat=0, max_frames=False, episodic_life=False, render_mode=render_mode, image_path=None
#                                 )
#                         else:
#                             envs = instantiate_env(
#                                 env_id, num_envs=1, env_variation=enc_bg, env_seed=cust_seed, num_stack=4,
#                                 num_no_op=0, action_repeat=4, max_frames=False, episodic_life=True, render_mode=render_mode, image_path=None
#                                 )
#                         # envs = instantiate_env(env_id, stack=True, render_mode=render_mode, env_variation=enc_bg, env_seed=cust_seed, image_path=None)
                            
#                         encoder_instance, policy_instance, agent_instance = get_algo_instance(encoder_algo, policy_algo)
#                         encoder, policy, agent = load_model(env_id=env_id, env_info=env_info, is_relative=is_relative, encoder_model_color=enc_bg, encoder_algo=encoder_algo,
#                                             encoder_model_type=encoder_model_type, policy_model_color=pol_bg, policy_algo=policy_algo,
#                                             policy_model_type=policy_model_type, action_space=envs.single_action_space.n, is_pretrained=is_pretrained,
#                                             FeatureExtractor=encoder_instance, Policy=policy_instance, Agent=agent_instance, anchors_alpha=anchors_alpha,
#                                             encoder_seed=enc_seed, policy_seed=pol_seed, encoder_eval=True, policy_eval=True, device=device
#                                             )
#                         # score, max_ep_score, ep_length = test_rel_repr_vec(envs, agent)
#                         limit_episode_length = -1
#                         if env_id.startswith("Breakout"):
#                             limit_episode_length = 4000
#                         score, max_ep_score, ep_length = test_rel_repr_vec(envs, agent, policy_algo, limit_episode_length, device=device)#, cust_seed=1)
#                         # score, max_ep_score, ep_length = test_rel_repr(env_id, cust_seed, enc_bg, policy_model_type, encoder_model_type, pol_bg, enc_bg, swap_policy=True)
#                         print(f"Episode finished: {score} points, {ep_length} steps")
                        
#                         df = df.append({"env_seed": i, "encoder_background": enc_bg, "policy_background": pol_bg,
#                                         "encoder_seed": enc_seed, "policy_seed": pol_seed,
#                                         "score": score, "max_score_reached": max_ep_score, "episode_length": ep_length,
#                                         "algorithm": "ppo"}, ignore_index=True)
#     # print a recap of the results, along with the max score and max episode length, average score and average episode length over all seeds
#     print(df)
#     return df







# def stitching_test_alignment_old(
#         env_id, env_info, anchoring_method='fps', encoder_algo="ppo",
#         policy_algo="ppo", encoder_model_type="tanh", policy_model_type="tanh",
#         is_relative=False, is_pretrained=False, swap_anchors=False, anchors_path=None,
#         anchors_alpha="0999", render_mode="rgb_array", device='cpu', env_seeds_totest=[0]
#         ):    


#     from latentis.space import LatentSpace
#     from latentis.utils import seed_everything
#     from latentis.estimate.dim_matcher import ZeroPadding
#     from latentis.estimate.orthogonal import SVDEstimator
#     from latentis.translate.translator import LatentTranslator
    

    
#     from pathlib import Path
#     from rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent
# #     from rl_agents.ddqn.ddqn_end_to_end_relu_align import AgentDDQN
#     import pickle
#     import time
#     import numpy as np
#     import os

#     from utils.models import load_model, get_algo_instance

#     if not os.path.exists(f"alignment_indices/{env_id}/{env_info}"):
#         os.makedirs(f"alignment_indices/{env_id}/{env_info}")
#     # env_pathname = f"{env_id}_{env_info}"
#     if env_id.startswith("CarRacing"): # if env_id == "CarRacing-v2":
#         background_to_test = ["green", "red", "blue", "yellow"]
#         seed_lst_enc = [1, 2, 3, 4]
#         seed_lst_pol = [1, 2, 3, 4]
#     else:
#         background_to_test = ["plain", "green", "red", "blue"]
#         seed_lst_enc = [0, 1, 2, 3] # [39, 40, 41, 42, 43]
#         seed_lst_pol = [0, 1, 2, 3] # [39, 40, 41, 42, 43]
#     # create a pandas dataframe to store the results
#     df = pd.DataFrame(columns=["env_seed", "encoder_background", "policy_background", "encoder_seed", "policy_seed",
#                                "score", "max_score_reached", "episode_length", "algorithm", "clustering_time"])
#     # test all combinations of backgrounds, averaged over 5 seeds. Store avg and max score
#     for _, enc_bg in enumerate(background_to_test):
#         for _, pol_bg in enumerate(background_to_test):
#             for enc_seed in seed_lst_enc:
#                 # enc_seed = seed_lst_enc[enc_idx]
#                 for pol_seed in seed_lst_pol:
#                     # pol_seed = seed_lst_pol[pol_idx]
#                     models_initialized = False
#                     for i in env_seeds_totest:
#                         cust_seed = i #+ 40
#                         time_to_translate = 0
#                         clustering_time = 0
#                         print('----------------------------------')
#                         print(f"Testing on {enc_bg} background with {enc_bg} encoder (seed {enc_seed}) and {pol_bg} policy (seed {pol_seed}). Seed {cust_seed}")
#                         # print(f"Seed: {i}")


#                         # envs = instantiate_env(env_id, stack=True, render_mode=render_mode, env_variation=enc_bg, env_seed=cust_seed, image_path=None)   
#                         if env_id == "CarRacing-v2":
#                             envs = instantiate_env(
#                                 env_id, num_envs=1, env_variation=enc_bg, env_seed=cust_seed, num_stack=4,
#                                 num_no_op=0, action_repeat=0, max_frames=False, episodic_life=False, render_mode=render_mode, image_path=None
#                                 )
#                         else:
#                             envs = instantiate_env(
#                                 env_id, num_envs=1, env_variation=enc_bg, env_seed=cust_seed, num_stack=4,
#                                 num_no_op=0, action_repeat=4, max_frames=False, episodic_life=True, render_mode=render_mode, image_path=None
#                                 )
#                         if not models_initialized:
#                             models_initialized = True
#                             encoder_instance_1, policy_instance_1, agent_instance_1 = get_algo_instance(encoder_algo, encoder_algo)
#                             encoder_instance_2, policy_instance_2, agent_instance_2 = get_algo_instance(encoder_algo, encoder_algo)

#                             encoder_1, policy_1, agent_1 = load_model(
#                                 env_id=env_id, env_info=env_info, is_relative=is_relative, encoder_model_color=enc_bg, encoder_algo=encoder_algo,
#                                 encoder_model_type=encoder_model_type, policy_model_color=enc_bg, policy_algo=encoder_algo,
#                                 policy_model_type=encoder_model_type, action_space=envs.single_action_space.n, is_pretrained=is_pretrained,
#                                 FeatureExtractor=encoder_instance_1, Policy=policy_instance_1, Agent=agent_instance_1, anchors_alpha=anchors_alpha,
#                                 encoder_seed=enc_seed, policy_seed=enc_seed, encoder_eval=True, policy_eval=True, device=device
#                                 )
                            
#                             encoder_2, policy_2, agent_2 = load_model(
#                                 env_id=env_id, env_info=env_info, is_relative=is_relative, encoder_model_color=pol_bg, encoder_algo=policy_algo,
#                                 encoder_model_type=policy_model_type, policy_model_color=pol_bg, policy_algo=policy_algo,
#                                 policy_model_type=policy_model_type, action_space=envs.single_action_space.n, is_pretrained=is_pretrained,
#                                 FeatureExtractor=encoder_instance_2, Policy=policy_instance_2, Agent=agent_instance_2, anchors_alpha=anchors_alpha,
#                                 encoder_seed=pol_seed, policy_seed=pol_seed, encoder_eval=True, policy_eval=True, device=device
#                                 )

#                             obs_set_1 = pickle.load(
#                                 Path(
#                                     f"data/anchors/{env_id}/rgb_ppo_transitions_{enc_bg}_obs.pkl"
#                                     # f"anchors/{env_id}_rgb_nostack_ppo_rgb_transitions_{enc_bg}_obs.pkl"
#                                 ).open("rb")
#                             )  # [30:2000]
#                             obs_set_2 = pickle.load(
#                                 Path(
#                                     f"data/anchors/{env_id}/rgb_ppo_transitions_{pol_bg}_obs.pkl"
#                                     # f"anchors/{env_id}_rgb_nostack_ppo_rgb_transitions_{pol_bg}_obs.pkl"
#                                 ).open("rb")
#                             )  # [30:2000]
#                             # subset_indices = np.random.randint(0, len(obs_set_1), 8000)
#                             subset_indices = np.arange(len(obs_set_1))

#                             obs_set_1 = torch.tensor(np.array(obs_set_1), dtype=torch.float32)
#                             obs_set_2 = torch.tensor(np.array(obs_set_2), dtype=torch.float32)

#                             space1 = encoder_1.forward_single(obs_set_1.to(device)).detach().cpu()
#                             space2 = encoder_2.forward_single(obs_set_2.to(device)).detach().cpu()

#                             from collections import namedtuple

#                             Space = namedtuple("Space", ["name", "vectors"])
#                             space1 = Space(name=enc_bg, vectors=space1)
#                             space2 = Space(name=pol_bg, vectors=space2)

#                             space1_vectors = space1.vectors
#                             space2_vectors = space2.vectors

#                             space1_anchors = space1_vectors[:]
#                             space2_anchors = space2_vectors[:]

#                             from utils.anchoring_methods import get_anchors
#                             """ CHANGE ANCHOR SAMPLING METHOD HERE """
#                             """ only provide filename without directory nor sampling method. That part is handled by get_anchors """
#                             translation_path = f'alignment_indices/{env_id}/{env_info}/{anchoring_method}_{enc_bg}_{enc_seed}_closest.pt'#{pol_bg}_closest.pt'
#                             # filename = f'{env_id}_rgb_nostack_ppo_rgb_transitions_{enc_bg}_{pol_bg}_closest.pt'
#                             num_anchors = 3136
#                             start = time.time()
#                             torch.manual_seed(42)
#                             space1_anchors, space2_anchors = get_anchors(space1_vectors, space2_vectors, num_anchors, subset_indices, anchoring_method, translation_path, device)
#                             clustering_time = round(time.time() - start, 3)
#                             print(f"{anchoring_method} done in {clustering_time} seconds.\n\n")
#                             translation = LatentTranslator(
#                                 random_seed=42,
#                                 estimator=SVDEstimator(dim_matcher=ZeroPadding()),
#                                 source_transforms=None, #[transforms.StandardScaling()],
#                                 target_transforms=None, #[transforms.StandardScaling()],
#                             )


#                             # translation = LatentTranslation(
#                             #     seed=42,
#                             #     translator=SVDTranslator(),
#                             #     source_transforms=None, #[Transforms.StandardScaling()], # None,
#                             #     target_transforms=None, #[Transforms.StandardScaling()] # None,
#                             # )
#                             space1_anchors = space1_anchors.to(device)  # [:3136]
#                             space2_anchors = space2_anchors.to(device)  # [:3136]
#                             space1 = LatentSpace(vectors=space1_anchors, name="space1")
#                             space2 = LatentSpace(vectors=space2_anchors, name="space2")

#                             print("\n##############################################\n")
#                             print(
#                                 f"fitting translation layer between {enc_bg} and {pol_bg} spaces using {anchoring_method}..."
#                             )
#                             start = time.time()
#                             translation.fit(source_data=space1, target_data=space2)
#                             # get elapsed time, rounded to 3 decimals
#                             time_to_translate = round(time.time() - start, 3)
#                             print(f"done in {time_to_translate} seconds.\n\n")

#                             agent = Agent(encoder_1, policy_2, translation=translation).to(device)
                        
#                         limit_episode_length = -1
#                         if env_id.startswith("Breakout"):
#                             limit_episode_length = 4000
#                         score, max_ep_score, ep_length = test_rel_repr_vec(envs, agent, policy_algo, limit_episode_length, device=device)#, cust_seed=1)
#                         print(f"Episode finished: {score} points, {ep_length} steps")
                        
#                         df = df.append({"env_seed": i, "encoder_background": enc_bg, "policy_background": pol_bg,
#                                         "encoder_seed": enc_seed, "policy_seed": pol_seed,
#                                         "score": score, "max_score_reached": max_ep_score, "episode_length": ep_length,
#                                         "algorithm": "ppo", "clustering_time": clustering_time}, ignore_index=True)
#     # print a recap of the results, along with the max score and max episode length, average score and average episode length over all seeds
#     print(df)
#     return df








# def stitching_test_alignment(
#         env_id, env_info, anchoring_method='fps', encoder_algo="ppo",
#         policy_algo="ppo", encoder_model_type="tanh", policy_model_type="tanh",
#         is_relative=False, is_pretrained=False, swap_anchors=False, anchors_path=None,
#         anchors_alpha="0999", render_mode="rgb_array", device='cpu', env_seeds_totest=[0]
#         ):    
#     # from utils.alignment import (
#     #     LatentTranslation,
#     #     IdentityTranslator,
#     #     AffineTranslator,
#     #     LSTSQTranslator,
#     #     LSTSQOrthoTranslator,
#     #     SVDTranslator,
#     # )
#     # from latentis.transforms import Transforms


#     from latentis.space import LatentSpace
#     from latentis.utils import seed_everything
#     from latentis import transforms
#     from latentis.estimate.dim_matcher import ZeroPadding
#     from latentis.estimate.orthogonal import SVDEstimator
#     from latentis.translate.translator import LatentTranslator
    

    
#     from pathlib import Path
#     from rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent
#     from rl_agents.ddqn.ddqn_end_to_end_relu_align import AgentDDQN
#     import pickle
#     import time
#     import numpy as np
#     import os

#     from utils.models import load_model_carracing, get_algo_instance

#     # translators = [
#     #     IdentityTranslator(),
#     #     AffineTranslator(),
#     #     LSTSQTranslator(),
#     #     LSTSQOrthoTranslator(),
#     #     SVDTranslator(),
#     # ]
#     # transforms_options = [
#     #     [],
#     #     [Transforms.Centering()],
#     #     [Transforms.StandardScaling()],
#     #     [Transforms.Centering(), Transforms.StandardScaling()],
#     #     [Transforms.Centering(), Transforms.L2()],
#     # ]

#     if not os.path.exists(f"alignment_indices/{env_id}/{env_info}"):
#         os.makedirs(f"alignment_indices/{env_id}/{env_info}")
#     # env_pathname = f"{env_id}_{env_info}"
#     if env_id.startswith("CarRacing"): # if env_id == "CarRacing-v2":
#         background_to_test = ["green", "red", "blue", "yellow"]
#         seed_lst_enc = [40, 41, 42, 43]
#         seed_lst_pol = [40, 41, 42, 43]
#     else:
#         background_to_test = ["plain", "green", "red", "blue"]
#         seed_lst_enc = [9, 0, 1, 2] # [39, 40, 41, 42, 43]
#         seed_lst_pol = [9, 0, 1, 2] # [39, 40, 41, 42, 43]
#     # create a pandas dataframe to store the results
#     df = pd.DataFrame(columns=["env_seed", "encoder_background", "policy_background", "score", "max_score_reached", "episode_length", "algorithm", "time_to_translate"])
#     # test all combinations of backgrounds, averaged over 5 seeds. Store avg and max score
#     for enc_idx, enc_bg in enumerate(background_to_test):
#         for pol_idx, pol_bg in enumerate(background_to_test):
#             enc_seed = seed_lst_enc[enc_idx]
#             pol_seed = seed_lst_pol[pol_idx]
#             models_initialized = False
#             for i in env_seeds_totest:
#                 cust_seed = i #+ 40
#                 time_to_translate = 0
#                 clustering_time = 0
#                 print('----------------------------------')
#                 print(f"Testing {enc_bg} encoder with {pol_bg} policy on {enc_bg} background. Seed {cust_seed}")
#                 # print(f"Seed: {i}")


#                 # envs = instantiate_env(env_id, stack=True, render_mode=render_mode, env_variation=enc_bg, env_seed=cust_seed, image_path=None)   
#                 if env_id == "CarRacing-v2":
#                     envs = instantiate_env(
#                         env_id, num_envs=1, env_variation=enc_bg, env_seed=cust_seed, num_stack=4,
#                         num_no_op=0, action_repeat=0, max_frames=False, episodic_life=False, render_mode=render_mode, image_path=None
#                         )
#                 else:
#                     envs = instantiate_env(
#                         env_id, num_envs=1, env_variation=enc_bg, env_seed=cust_seed, num_stack=4,
#                         num_no_op=0, action_repeat=4, max_frames=False, episodic_life=True, render_mode=render_mode, image_path=None
#                         )
#                 if not models_initialized:
#                     models_initialized = True
#                     encoder_instance_1, policy_instance_1, agent_instance_1 = get_algo_instance(encoder_algo, encoder_algo)
#                     encoder_instance_2, policy_instance_2, agent_instance_2 = get_algo_instance(encoder_algo, encoder_algo)

#                     encoder_1, policy_1, agent_1 = load_model_carracing(
#                         env_id=env_id, env_info=env_info, is_relative=is_relative, encoder_model_color=enc_bg, encoder_algo=encoder_algo,
#                         encoder_model_type=encoder_model_type, policy_model_color=enc_bg, policy_algo=encoder_algo,
#                         policy_model_type=encoder_model_type, action_space=envs.single_action_space.n, is_pretrained=is_pretrained,
#                         FeatureExtractor=encoder_instance_1, Policy=policy_instance_1, Agent=agent_instance_1, anchors_alpha=anchors_alpha,
#                         encoder_seed=enc_seed, policy_seed=enc_seed, encoder_eval=True, policy_eval=True, device=device
#                         )
                    
#                     encoder_2, policy_2, agent_2 = load_model_carracing(
#                         env_id=env_id, env_info=env_info, is_relative=is_relative, encoder_model_color=pol_bg, encoder_algo=policy_algo,
#                         encoder_model_type=policy_model_type, policy_model_color=pol_bg, policy_algo=policy_algo,
#                         policy_model_type=policy_model_type, action_space=envs.single_action_space.n, is_pretrained=is_pretrained,
#                         FeatureExtractor=encoder_instance_2, Policy=policy_instance_2, Agent=agent_instance_2, anchors_alpha=anchors_alpha,
#                         encoder_seed=pol_seed, policy_seed=pol_seed, encoder_eval=True, policy_eval=True, device=device
#                         )

#                     obs_set_1 = pickle.load(
#                         Path(
#                             f"data/anchors/{env_id}/rgb_ppo_transitions_{enc_bg}_obs.pkl"
#                             # f"anchors/{env_id}_rgb_nostack_ppo_rgb_transitions_{enc_bg}_obs.pkl"
#                         ).open("rb")
#                     )  # [30:2000]
#                     obs_set_2 = pickle.load(
#                         Path(
#                             f"data/anchors/{env_id}/rgb_ppo_transitions_{pol_bg}_obs.pkl"
#                             # f"anchors/{env_id}_rgb_nostack_ppo_rgb_transitions_{pol_bg}_obs.pkl"
#                         ).open("rb")
#                     )  # [30:2000]
#                     # subset_indices = np.random.randint(0, len(obs_set_1), 8000)
#                     subset_indices = np.arange(len(obs_set_1))

#                     obs_set_1 = torch.tensor(np.array(obs_set_1), dtype=torch.float32)
#                     obs_set_2 = torch.tensor(np.array(obs_set_2), dtype=torch.float32)

#                     space1 = encoder_1.forward_single(obs_set_1.to(device)).detach().cpu()
#                     space2 = encoder_2.forward_single(obs_set_2.to(device)).detach().cpu()

#                     from collections import namedtuple

#                     Space = namedtuple("Space", ["name", "vectors"])
#                     space1 = Space(name=enc_bg, vectors=space1)
#                     space2 = Space(name=pol_bg, vectors=space2)

#                     space1_vectors = space1.vectors
#                     space2_vectors = space2.vectors

#                     space1_anchors = space1_vectors[:]
#                     space2_anchors = space2_vectors[:]

#                     from utils.anchoring_methods import get_anchors
#                     """ CHANGE ANCHOR SAMPLING METHOD HERE """
#                     """ only provide filename without directory nor sampling method. That part is handled by get_anchors """
#                     translation_path = f'alignment_indices/{env_id}/{env_info}/{anchoring_method}_{enc_bg}_{pol_bg}_closest.pt'
#                     # filename = f'{env_id}_rgb_nostack_ppo_rgb_transitions_{enc_bg}_{pol_bg}_closest.pt'
#                     num_anchors = 3136
#                     start = time.time()
#                     torch.manual_seed(42)
#                     space1_anchors, space2_anchors = get_anchors(space1_vectors, space2_vectors, num_anchors, subset_indices, anchoring_method, translation_path, device)
#                     clustering_time = round(time.time() - start, 3)
#                     print(f"{anchoring_method} done in {clustering_time} seconds.\n\n")
#                     translation = LatentTranslation(
#                         seed=42,
#                         translator=SVDTranslator(),
#                         source_transforms=None, #[Transforms.StandardScaling()], # None,
#                         target_transforms=None, #[Transforms.StandardScaling()] # None,
#                     )
#                     space1_anchors = space1_anchors.to(device)  # [:3136]
#                     space2_anchors = space2_anchors.to(device)  # [:3136]

#                     print("\n##############################################\n")
#                     print(
#                         f"fitting translation layer between {enc_bg} and {pol_bg} spaces using {anchoring_method}..."
#                     )
#                     start = time.time()
#                     translation.fit(source_anchors=space1_anchors, target_anchors=space2_anchors)
#                     # get elapsed time, rounded to 3 decimals
#                     time_to_translate = round(time.time() - start, 3)
#                     print(f"done in {time_to_translate} seconds.\n\n")

#                     agent = Agent(encoder_1, policy_2, translation=translation).to(device)
                
#                 score, max_ep_score, ep_length = test_rel_repr_vec(envs, agent, policy_algo=policy_algo, device=device)
#                 print(f"Episode finished: {score} points, {ep_length} steps")
                
#                 df = df.append({"env_seed": i, "encoder_background": enc_bg, "policy_background": pol_bg,
#                                 "score": score, "max_score_reached": max_ep_score, "episode_length": ep_length,
#                                 "algorithm": "ppo", "clustering_time": clustering_time}, ignore_index=True)
#     # print a recap of the results, along with the max score and max episode length, average score and average episode length over all seeds
#     print(df)
#     return df
