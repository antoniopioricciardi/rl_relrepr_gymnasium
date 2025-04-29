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
    env = make_env(env_id)

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

    obs, _ = env.reset()
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
        for _ in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs.unsqueeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().clip(-1, 1)[0])
            done = terminated or truncated
            obs_batch.append(obs)
            actions_batch.append(action)
            logprobs_batch.append(logprob)
            rewards_batch.append(torch.tensor(reward, device=device))
            dones_batch.append(torch.tensor(done, device=device))
            values_batch.append(value)

            obs = torch.tensor(next_obs["observation"], dtype=torch.float32, device=device)
            global_step += 1
            if done:
                obs, _ = env.reset()
                obs = torch.tensor(obs["observation"], dtype=torch.float32, device=device)

        # Compute advantage estimates using GAE
        advantages = []
        last_advantage = 0
        with torch.no_grad():
            next_value = agent.get_action_and_value(obs.unsqueeze(0))[3]
        for t in reversed(range(num_steps)):
            mask = 1.0 - dones_batch[t].float()
            delta = rewards_batch[t] + gamma * next_value * mask - values_batch[t]
            advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages.insert(0, advantage)
            last_advantage = advantage
            next_value = values_batch[t]
        
        # Flatten batches
        b_obs = torch.stack(obs_batch)
        b_actions = torch.stack(actions_batch)
        b_logprobs = torch.stack(logprobs_batch)
        b_advantages = torch.stack(advantages).detach()
        b_returns = (b_advantages + torch.stack(values_batch)).detach()

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
        if "episode" in info:
            print(f"Step: {global_step}, Episode Return: {info['episode']['r']}")

if __name__ == "__main__":
    main()