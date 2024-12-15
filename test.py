from zeroshotrl.envs.lunarlander.lunar_lander_rgb import LunarLanderRGB
import numpy as np

env = LunarLanderRGB(render_mode="rgb_array", color="red")
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