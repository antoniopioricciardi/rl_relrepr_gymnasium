import gymnasium as gym
import miniworld
from zeroshotrl.utils.preprocess_env import ColorTransformObservation

"""
WARN: env.render_top_view to get variables from other wrappers is deprecated and will be removed in v1.0,
to get this variable you can do `env.unwrapped.render_top_view` for environment variables or `env.get_wrapper_attr('render_top_view')`
that will search the reminding wrappers.
"""

# env = gym.make("MiniWorld-OneRoom-v0", render_mode="human")
env = gym.make("MiniWorld-OneRoom-v0", render_mode="human")
env = ColorTransformObservation(env, color="standard")
observation, info = env.reset(seed=0)
score = 0   
for _ in range(1000):
    # action = policy(observation)  # User-defined policy function
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
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