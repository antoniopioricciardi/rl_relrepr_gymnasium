import numpy as np
import torch
import pickle

import argparse

device = "cpu"

# set command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-id", type=str, default="BreakoutNoFrameskip-v4", help="Environment id"
)
parser.add_argument("--seed", type=int, default=40, help="Random seed")

# parse arguments
bg_colors_allowed = ["plain", "green", "red", "blue", "violet", "yellow", "white"]
# parser = argparse.ArgumentParser()
# parser.add_argument("--background-type", type=str, default="plain", choices=["plain", "color", "image", "video"],
#                     help="background type: plain, color, image, video")
parser.add_argument(
    "--background",
    type=str,
    default="green",
    help=f"background color {bg_colors_allowed}, image or video paths",
)
parser.add_argument(
    "--render-mode", type=str, default="human", help="render mode: human, rgb_array"
)
parser.add_argument(
    "--algo",
    type=str,
    default="ppo",
    required=True,
    help="algorithm used to generate actions: ppo, ddqn...",
)
parser.add_argument(
    "--actions-path", type=str, default="", required=True, help="path to actions list"
)
# parse arguments
args = parser.parse_args()

# if args.env_id.startswith("CarRacing-v2"):
#     instantiate_env(env_id=args.env_id, num_envs=1, env_variation=args.background, env_seed=args.seed, num_stack=0, num_no_op=0, action_repeat=0, max_frames=False, episodic_life=False, render_mode=args.render_mode, image_path=None)
# else:
#     instantiate_env(env_id=args.env_id, num_envs=1, env_variation=args.background, env_seed=args.seed, num_stack=3, num_no_op=30, action_repeat=3, max_frames=False, episodic_life=True, render_mode=args.render_mode, image_path=None)

from zeroshotrl.utils.env_initializer import init_env

env = init_env(
    args.env_id,
    "rgb",
    background_color=args.background,
    image_path="",
    cust_seed=args.seed,
    render_md=args.render_mode,
)

# if args.env_id.startswith("CarRacing-v2"):
#     env = instantiate_env(env_id=args.env_id, num_envs=1, env_variation=args.background, env_seed=args.seed, num_stack=4, num_no_op=0, action_repeat=0, max_frames=False, episodic_life=False, clip_reward=False, render_mode=args.render_mode, image_path=None)
# elif args.env_id.startswith("Wolfenstein"):
#     lvl = args.env_id.split('-')[1]
#     from wolfenstein_rl.wolfenstein_env import Wolfenstein
#     env = Wolfenstein(level=lvl, render_mode=args.render_mode).env
#     env = gym.vector.SyncVectorEnv([
#         make_env_atari(
#         env, seed=args.seed, rgb=True, stack=4, no_op=0, action_repeat=0,
#         max_frames=False, episodic_life=False, clip_reward=True, check_fire=False, idx=i, capture_video=False, run_name='test'
#         )
#         for i in range(1)
#         ])

# else:
#     env = instantiate_env(env_id=args.env_id, num_envs=1, env_variation=args.background, env_seed=args.seed, num_stack=4, num_no_op=0, action_repeat=4, max_frames=False, episodic_life=True, clip_reward=False, render_mode=args.render_mode, image_path=None)

# track_background = args.background if args.background_type == "color" else args.background_type


actions_path = args.actions_path  # "ppo_rgb_transitions_green_a.pkl" # TODO
print(actions_path)
# relative = True
# pretrained = False
# model_type = 'ppo'
env_id = args.env_id  # "BreakoutNoFrameskip-v4"
env_info = "rgb"
seed = args.seed

""" Parameters to change for single test """
track_background = args.background  # "plain"
# encoder_model_color = "plain"
# policy_model_color = "plain"
# encoder_seed = "39"
# policy_seed = "39"

encoder_algo = args.algo
# policy_algo = "ppo"

# encoder_activation = "relu"
# policy_activation = "relu"

# encoder_alpha="0999"
# policy_alpha="0999"
""" ----- """

# env_pathname = f"{env_id}_{env_info}"


obs_list = []

obs, _ = env.reset(seed=args.seed)
score = 0
scores = []
# open file containing actions list. Filename is ppo_rgb_transitions_green_a.pkl
# open actions file using numpy
# actions_list = np.load(actions_path)
with open(actions_path, "rb") as f:
    actions_list = pickle.load(f)  # ["actions"]

# obs_list.append(obs)
# run 1000 steps


for i in range(len(actions_list)):
    # obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    obs = torch.Tensor(obs).to(device)
    action = actions_list[i]
    next_obs, reward, terminated, truncated, info = env.step([action])
    done = np.logical_or(terminated, truncated)
    score += reward
    obs_list.append(
        obs[0][-1].detach().numpy()
    )  # -1 because we take last frame of the stack
    obs = next_obs
    if done:
        if env_id.startswith("CarRacing"):
            print("episode finished, score: ", score)
            scores.append(score[0])
            score = 0
            seed += 1
            obs, _ = (
                env.reset(seed=seed)
            )  # shouldn't call this. Vectorized env handles this. If I remove this, need to collect ALL data again
        # elif info["lives"] == 0:
        #     print('episode finished, score: ', score)
        #     scores.append(score)
        #     score = 0
        #     obs = env.reset()
        else:
            # print number of lives using vectorized env
            print("episode finished, score: ", score)
            scores.append(score[0])
            score = 0
# obs_list.append(torch.tensor(obs, dtype=torch.float32)[-1]) # we have no unsqueeze here, so we directly take the last frame
# convert list to numpy array
if len(scores) == 0:
    scores.append(score)
print(f"Finished collecting observations, avg score: {sum(scores)/len(scores)}")
# obs_list = np.array(obs_list, dtype=np.float32)
# print("Storing frames as a numpy array of shape: ", obs_list.shape)


# # show every saved image
# for i, image in enumerate(images):
#     plt.imshow(image[3])
#     plt.title(f"step {i}")
#     plt.show()

import os

if not os.path.exists(f"data/anchors/{env_id}"):  # /{env_info}'):
    os.makedirs(f"data/anchors/{env_id}")  # /{env_info}')

with open(
    f"data/anchors/{env_id}/{env_info}_{encoder_algo}_transitions_{track_background}_obs.pkl",
    "wb",
) as f:
    pickle.dump(obs_list, f)

# save file as a numpy array
# np.save(f"data/anchors/{env_pathname}_{encoder_algo}_rgb_transitions_{track_background}_obs.npy", obs_list)
