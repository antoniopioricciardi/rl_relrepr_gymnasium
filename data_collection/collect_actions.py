import os
import torch
import numpy as np

# set command line arguments
import argparse
# parse arguments


import tqdm

# from rl_agents.ddqn import FeatureExtractorDDQN, PolicyDDQN, AgentDDQN
# from rl_agents.ppo import FeatureExtractor, Policy, Agent
# from rl_agents.ppo.ppo_end_to_end_relu_stack import FeatureExtractor, Policy, Agent
from zeroshotrl.utils.models import load_model, get_algo_instance, get_algo_instance_bw, load_model_from_path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-id", type=str, default="BreakoutNoFrameskip-v4", help="Environment id"
)
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument(
    "--model-seed",
    type=int,
    default=40,
    help="training seed for the model to be loaded",
)
parser.add_argument(
    "--num-steps", type=int, default=20000, help="Number of steps to collect"
)
parser.add_argument(
    "--background",
    type=str,
    default="plain",
    help="background type: plain, green, red, blue, yellow",
)
parser.add_argument(
    "--model-dir",
    type=str,
    default="data/ppo_models",
    help="path to the directory containing the encoder and policy models",
)
parser.add_argument(
    "--render-mode", type=str, default="rgb_array", help="render mode: human, rgb_array"
)
# TODO: add arguments for encoder and policy model paths

args = parser.parse_args()

pretrained = False
relative = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

relative = False
pretrained = False
# model_type = 'ppo'
env_id = args.env_id
env_info = "rgb"
# cust_seed = 1

""" Parameters to change for single test """
background_color = args.background  # "plain"
# encoder_model_color = "plain"
# policy_model_color = "plain"
encoder_seed = args.model_seed  # "39"
policy_seed = args.model_seed  # "39"

encoder_algo = "ppo"
policy_algo = "ppo"

encoder_activation = "relu"
policy_activation = "relu"

encoder_alpha = None  # "0999"
# policy_alpha="0999"
# """ ----- """

# video_path = "data/track_bg_videos/0.mp4"
# image_path = "data/track_bg_images/0.jpg"

imgsource = "plain" if background_color == "plain" else "color"

# env_pathname = f"{env_id}_{env_info}"

if env_info == "bw":
    encoder_instance, policy_instance, agent_instance = get_algo_instance_bw(
        encoder_algo, policy_algo
    )
else:
    encoder_instance, policy_instance, agent_instance = get_algo_instance(
        encoder_algo, policy_algo, use_resnet=False
    )
from zeroshotrl.utils.env_initializer import init_env

env = init_env(
    args.env_id,
    "rgb",
    background_color=args.background,
    image_path="",
    cust_seed=args.seed,
    render_md=args.render_mode,
)

encoder_path = os.path.join(
    args.model_dir, "encoder.pt"
)
policy_path = os.path.join(
    args.model_dir, "policy.pt"
)

encoder, policy, agent = load_model_from_path(
    encoder_path,
    policy_path,
    env.single_action_space.n,
    FeatureExtractor=encoder_instance,
    Policy=policy_instance,
    Agent=agent_instance,
    encoder_eval=True,
    policy_eval=True,
    is_relative=False,
    is_pretrained=False,
    anchors_alpha=None,
    device=device
    )
    

# encoder, policy, agent = load_model_from_path(
#     path_enc, path_pol, envs.single_action_space.n,
#     encoder_instance, policy_instance, agent_instance, is_relative=False, is_pretrained=False, device=device
#     )

# encoder, policy, agent = load_model(
#     env_id,
#     env_info,
#     relative,
#     background_color,
#     encoder_algo,
#     encoder_activation,
#     background_color,
#     policy_algo,
#     policy_activation,
#     env.single_action_space.n,
#     pretrained,
#     FeatureExtractor=encoder_instance,
#     Policy=policy_instance,
#     Agent=agent_instance,
#     anchors_alpha=encoder_alpha,
#     encoder_seed=encoder_seed,
#     policy_seed=policy_seed,
#     encoder_eval=True,
#     policy_eval=True,
#     device=device,
# )

# from natural_rl_environment.natural_env import NaturalEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)

# env = CarRacing(continuous=False, render_mode="rgb_array", background="green")
num_steps = args.num_steps
actions = []

scores = []
score = 0
# write reinforcement learning loop
seed = args.seed
obs, _ = env.reset(seed=seed)
random_every = 0  # pong:20
k = 0
for i in tqdm.tqdm(range(num_steps)):
    # action = env.action_space.sample()
    with torch.no_grad():
        # action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(device))
        action, logprob, _, value = agent.get_action_and_value(
            torch.Tensor(obs).to(device)
        )
        if (random_every > 0) and (i % random_every == 0):
            if k < 20:
                action = torch.tensor([env.single_action_space.sample()])
                k += 1
            else:
                k = 0
    actions.append(action.cpu().numpy().item())
    next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
    done = np.logical_or(terminated, truncated)
    obs = next_obs
    score += reward
    # if i == 100:
    #     obs = env.reset()
    if done:
        if env_id == "CarRacing-v2":
            print("episode finished, score: ", score)
            scores.append(score)
            score = 0
            seed += 1
            obs, _ = env.reset(seed=seed)
        else:
            # print number of lives using vectorized env
            print("episode finished, score: ", score)
            scores.append(score[0])
            score = 0

    # # Only print when at least 1 env is done
    # if "final_info" not in info:
    #     continue

    # for info_idx, info in enumerate(info["final_info"]):
    #     # Skip the envs that are not done
    #     if info is None:
    #         continue
    #     score = info["episode"]["r"]

if len(scores) == 0:
    scores.append(score)
print("Finished collecting actions, average score: ", sum(scores) / len(scores))


# save actions
import pickle

with open(f"data/actions_lists/{args.env_id}_actions_{num_steps}.pkl", "wb") as f:
    pickle.dump(actions, f)

# save actions using numpy
# import numpy as np
# np.save(f"data/actions_lists/{args.env_id}_actions_{num_steps}.npy", actions)
