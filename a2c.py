import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import panda_gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(obs).squeeze(1)
        return action, log_prob, entropy, value

def make_env(env_id):
    env = gym.make(env_id)
    # env = RecordEpisodeStatistics(env)
    return env

def main():
    env_id = "PandaReachDense-v3"
    num_envs = 8
    env_fns = [lambda: make_env(env_id) for _ in range(num_envs)]
    env = gym.vector.AsyncVectorEnv(env_fns)

    obs_dim = np.array(env.observation_space["observation"].shape).prod()
    act_dim = np.array(env.action_space.shape).prod()

    agent = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

    # Hyperparameters
    total_timesteps = 200_000
    num_steps = 2048
    gamma = 0.99
    gae_lambda = 0.95
    update_epochs = 10
    minibatch_size = 64
    clip_grad_norm = 0.5

    obs, info = env.reset()
    print(obs)
    exit(2)
    obs = torch.tensor(obs["observation"], dtype=torch.float32, device=device)
    global_step = 0

    while global_step < total_timesteps:
        # Rollout storage
        obs_batch = []
        actions_batch = []
        logprobs_batch = []
        rewards_batch = []
        dones_batch = []
        values_batch = []

        # Collect experiences
        for step in range(num_steps):
            with torch.no_grad():
                # Get actions for all environments
                action, logprob, _, value = agent.get_action_and_value(obs)

            # Convert actions to numpy and clip them
            action_np = action.cpu().numpy().clip(-1, 1)

            # Step all environments
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            # Compute done flags for each environment
            done = np.logical_or(terminated, truncated)

            # Append current batch data (each tensor here has shape (num_envs, ...))
            obs_batch.append(obs)
            actions_batch.append(action)
            logprobs_batch.append(logprob)
            rewards_batch.append(torch.tensor(reward, device=device))
            dones_batch.append(torch.tensor(done, device=device))
            values_batch.append(value)

            # For environments that are done, reset them and update next_obs accordingly
            if done.any():
                reset_indices = np.where(done)[0]
                reset_obs, reset_info = env.reset(indices=reset_indices)
                next_obs["observation"][reset_indices] = reset_obs["observation"]

            # Prepare observations for the next step
            obs = torch.tensor(next_obs["observation"], dtype=torch.float32, device=device)

            # Increase global step by the number of environments
            global_step += num_envs

        # Compute advantage estimates using GAE
        advantages = [None] * num_steps
        last_advantage = torch.zeros(num_envs, device=device)
        with torch.no_grad():
            next_value = agent.get_action_and_value(obs)[3]  # shape: (num_envs,)
        for t in reversed(range(num_steps)):
            mask = 1.0 - dones_batch[t].float()
            delta = rewards_batch[t] + gamma * next_value * mask - values_batch[t]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[t] = last_advantage
            next_value = values_batch[t]
        
        # Flatten batches
        b_obs = torch.stack(obs_batch).reshape(-1, obs.shape[-1])
        b_actions = torch.stack(actions_batch).reshape(-1, actions_batch[0].shape[-1])
        b_logprobs = torch.stack(logprobs_batch).reshape(-1)
        b_advantages = torch.stack(advantages).reshape(-1).detach()
        b_values = torch.stack(values_batch).reshape(-1)
        b_returns = (b_advantages + b_values).detach()

        # Optimize policy for multiple epochs
        for epoch in range(update_epochs):
            indices = np.arange(num_steps)
            np.random.shuffle(indices)
            for start in range(0, num_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = new_logprob - b_logprobs[mb_inds]

                ratio = logratio.exp()
                policy_loss = -(b_advantages[mb_inds] * ratio).mean()
                value_loss = ((new_value - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), clip_grad_norm)
                optimizer.step()

        # Logging
        episode_returns = [d['episode']['r'] for d in info if 'episode' in d]
        if episode_returns:
            print(f"Step: {global_step}, Average Episode Return: {np.mean(episode_returns)}")

if __name__ == "__main__":
    main()