import gymnasium as gym
import miniworld

from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor, Policy, Agent

env = gym.make("MiniWorld-OneRoom-v0", render_mode="rgb_array")
eval_env = gym.make("MiniWorld-OneRoom-v0", render_mode="rgb_array")

encoder = FeatureExtractor(env.observation_space.shape[0])
controller = Policy(env.action_space.n)
agent = Agent(encoder, controller)

done = False
obs, _ = env.reset()

while not done:
    action = agent(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if done:
        obs = env.reset()