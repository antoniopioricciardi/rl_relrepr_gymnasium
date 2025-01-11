import gymnasium as gym
import miniworld
from zeroshotrl.utils.preprocess_env import ColorTransformObservation


env = gym.make("MiniWorld-OneRoom-v0", render_mode="human")
env = ColorTransformObservation(env, color="standard")
observation, info = env.reset(seed=0)
score = 0   
for _ in range(1000):
    # action = policy(observation)  # User-defined policy function
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    td_v = env.render_top_view()

    # plot both views
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(observation)
    plt.subplot(1, 2, 2)
    plt.imshow(td_v)
    plt.show()

    score += reward
    if terminated or truncated:
        observation, info = env.reset()
env.close()
print("Score:", score)