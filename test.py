from zeroshotrl.envs.lunarlander.lunar_lander_orig import LunarLander
import numpy as np
import torch

env = LunarLander(render_mode="rgb_array", gravity=-10)

# load thet agent from models/LunarLander/states/standard/ppo/relu/seed_1

encoder_weights = torch.load("models/LunarLander/states/standard/ppo/relu/seed_1/encoder.pt")
controller_weights = torch.load("models/LunarLander/states/standard/ppo/relu/seed_1/controller.pt")




# env = ScreenObsWrapper(env)
# env.reset()

# # obs = env.render()

# # print("Observation shape:", obs.shape)
# # print("Observation dtype:", obs.dtype)

# score = 0
# # training loop sampling random actions. Render it
# for i in range(100):
#     action = env.action_space.sample()
#     obs, reward, truncated, terminated, info = env.step(action)
#     score += reward
#     done = np.logical_or(truncated, terminated)
#     if done:
#         env.reset()

# print(score, i)

