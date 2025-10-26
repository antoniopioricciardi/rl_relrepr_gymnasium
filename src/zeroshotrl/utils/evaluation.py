import torch
import gymnasium as gym
import numpy as np


def evaluate_vec_env_ddqn(
    agent, num_envs, env, global_step, device, episode_n, writer=None, logger=None
):
    # eval and save model
    # TODO: set eval_freq as a hyperparameter
    # envs = gym.vector.AsyncVectorEnv(
    #     [make_env(env, seed + i, i, capture_video, f"{run_name}_eval") for i in range(num_envs)]
    # )
    # check whether envs is instance of AsyncVectorEnv or SyncVectorEnv
    if not isinstance(env, gym.vector.AsyncVectorEnv):
        raise ValueError("envs must be an instance of AsyncVectorEnv")
    eval_dones = np.zeros(num_envs)
    eval_rewards = np.zeros(num_envs)
    eval_lengths = np.zeros(num_envs)
    # perform evaluation
    obs = torch.Tensor(env.reset()).to(device)
    # ALGO LOGIC: action logic
    # loop until all eval_dones are True
    while not np.all(eval_dones):
        with torch.no_grad():
            q_values = agent(obs)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            # action = envs.action_space.sample()
        next_obs, rewards, dones, infos = env.step(actions)
        # increment scores for envs that are not done
        eval_rewards[eval_dones == 0] += rewards[eval_dones == 0]
        # increment lengths for envs that are not done
        eval_lengths[eval_dones == 0] += 1
        # get indices of envs that are done
        eval_done_idxs = np.argwhere(dones).flatten()
        # set eval_dones to 1 for envs that are done
        eval_dones[eval_done_idxs] = 1
        obs = torch.Tensor(next_obs).to(device)

    # log evaluation data
    eval_avg_reward = np.mean(eval_rewards)
    eval_avg_length = np.mean(eval_lengths)
    print(
        f"Eval episode: {episode_n}, Eval avg reward: {eval_avg_reward}, Eval avg length: {eval_avg_length}"
    )
    if writer is not None:
        writer.add_scalar("charts/eval_score", eval_avg_reward, global_step)
    if logger is not None:
        # logger from meta
        with logger.log_and_dump_ctx(global_step, ty="eval") as log:
            log("step", global_step)
            log("episode", episode_n)
            log("episode_reward", eval_avg_reward)
            log("episode_length", eval_avg_length)

    return eval_rewards, eval_lengths, eval_avg_reward


def evaluate_vec_env(
    agent,
    num_envs,
    env,
    global_step,
    device,
    episode_n,
    writer=None,
    logger=None,
    algorithm="ppo",
    episode_length_limit=-1,
    seed=0,
    forced_render=False,
):
    if (not isinstance(env, gym.vector.AsyncVectorEnv)) and (
        not isinstance(env, gym.vector.SyncVectorEnv)
    ):
        raise ValueError("envs must be an instance of AsyncVectorEnv or SyncVectorEnv")
    # Track per-env completion and stats
    eval_dones = np.zeros(num_envs)
    eval_rewards = np.zeros(num_envs)
    eval_lengths = np.zeros(num_envs)
    dds = np.zeros(num_envs)
    # perform evaluation
    e_obs, _ = env.reset(seed=seed)
    e_obs = torch.Tensor(e_obs).to(device)

    # ALGO LOGIC: action logic
    # loop until all eval_dones are True
    score = 0
    not_d = True
    # while not all envs are done
    while not_d:  # not np.all(eval_dones):
        if forced_render:
            env.envs[0].render()
        with torch.no_grad():
            if algorithm == "ppo":
                action, logprob, _, value = agent.get_action_and_value_deterministic(e_obs)
            elif algorithm == "ddqn":
                q_values = agent(e_obs)
                action = torch.argmax(q_values, dim=1)  # .cpu().numpy()
            else:
                raise ValueError(f"algorithm {algorithm} not supported")
            # action, logprob, _, value = agent.get_action_and_value(e_obs)
            # action = envs.action_space.sample()
        e_next_obs, e_reward, terminated, truncated, e_info = env.step(
            action.cpu().numpy()
        )
        # accumulate rewards/lengths for envs that are not yet marked done
        e_dones = np.logical_or(terminated, truncated)
        # Add step rewards only for envs not already finalized
        if isinstance(e_reward, np.ndarray):
            eval_rewards[dds == 0] += e_reward[dds == 0]
        else:
            # single env fallback
            if not dds[0]:
                eval_rewards[0] += e_reward
        eval_lengths[dds == 0] += 1
        e_obs = torch.Tensor(e_next_obs).to(device)
        score += e_reward
        # print("r:", e_reward)
        # Only print when at least 1 env is done
        if "final_info" not in e_info:
            continue

        for info_idx, info in enumerate(e_info["final_info"]):
            # Skip the envs that are not done
            if info is None:
                continue
            if not dds[info_idx]:
                # Mark env as done
                dds[info_idx] = 1
                # If RecordEpisodeStatistics is present, prefer its totals
                if isinstance(info, dict) and "episode" in info:
                    ep = info["episode"]
                    # Use provided totals if available
                    if isinstance(ep, dict):
                        if "r" in ep:
                            eval_rewards[info_idx] = ep["r"]
                        if "l" in ep:
                            eval_lengths[info_idx] = ep["l"]
        # check for the rare case an agent is stuck in an infinite loop
        if episode_length_limit > 0:
            dds[eval_lengths >= episode_length_limit] = 1
        # if all envs are done, break the loop
        if np.all(dds):
            not_d = False

    # log evaluation data
    eval_avg_reward = np.mean(eval_rewards)
    eval_avg_length = np.mean(eval_lengths)
    print(
        f"Eval episode: {episode_n}, Eval avg reward: {eval_avg_reward}, Eval avg length: {eval_avg_length}"
    )
    # print("score:", score)
    if writer is not None:
        writer.add_scalar("charts/eval_score", eval_avg_reward, global_step)
    if logger is not None:
        # logger from meta
        with logger.log_and_dump_ctx(global_step, ty="eval") as log:
            log("step", global_step)
            log("episode", episode_n)
            log("episode_reward", eval_avg_reward)
            log("episode_length", eval_avg_length)

    return eval_rewards, eval_lengths, eval_avg_reward
