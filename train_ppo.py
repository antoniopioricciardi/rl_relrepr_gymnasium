# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import time

import numpy as np
import torch
import torch.nn as nn


# from utils.relative import init_anchors, init_anchors_from_obs, get_obs_anchors, get_obs_anchors_totensor
from utils.helputils import save_model, upload_csv_wandb
from utils.evaluation import evaluate_vec_env


# """ RESUME TRAINING """
# experiment_name = args.exp_name
# if args.track and wandb.run.resumed:
#     starting_update = wandb.run.summary.get("charts/update") + 1
#     global_step = starting_update * args.self.batch_size
#     api = wandb.Api()
#     run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
#     model = run.file("agent.pt")
#     model.download(f"models/{experiment_name}/")
#     agent.load_state_dict(torch.load(
#         f"models/{experiment_name}/agent.pt", map_location=device))
#     agent.eval()
#     print(f"resumed at update {starting_update}")


class PPOTrainer_vec:
    def __init__(
        self,
        seed,
        total_timesteps,
        num_steps,
        num_eval_eps,
        num_minibatches,
        minibatch_size,
        update_epochs,
        envs,
        eval_envs,
        encoder,
        policy,
        agent,
        use_relative,
        pretrained,
        optimizer,
        checkpoint_frequency,
        learning_rate,
        gamma,
        norm_adv,
        gae_lambda,
        clip_coef,
        ent_coef,
        vf_coef,
        clip_vloss,
        max_grad_norm,
        target_kl,
        anneal_lr,
        track,
        wandb,
        writer,
        logger,
        log_path,
        csv_file_path,
        eval_csv_file_path,
        device,
        num_updates=0,
    ):
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.num_steps = num_steps
        self.num_eval_eps = num_eval_eps

        self.num_envs = envs.num_envs
        self.num_eval_envs = eval_envs.num_envs

        self.batch_size = int(self.num_envs * self.num_steps)  # 16 * 128 = 2048
        self.minibatch_size = int(self.batch_size // num_minibatches)  # 2048 // 4 = 512

        self.envs = envs
        self.eval_envs = eval_envs
        self.encoder = encoder
        self.policy = policy
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        self.logger = logger
        self.CHECKPOINT_FREQUENCY = checkpoint_frequency

        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.use_relative = use_relative
        self.pretrained = pretrained
        self.norm_adv = norm_adv
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_vloss = clip_vloss
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.anneal_lr = anneal_lr

        self.track = track
        self.wandb = wandb
        self.log_path = log_path

        self.csv_file_path = csv_file_path
        self.eval_csv_file_path = eval_csv_file_path

        self.device = device

        self.num_updates = num_updates

    def train(self):
        # ALGO Logic: Storage setup
        obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        episode_n = 0
        # elapsed_time = start_time
        steps_per_second = 0
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        # if self.num_updates > 0:
        #     num_updates = self.num_updates
        # else:
        num_updates = self.total_timesteps // self.batch_size  # 5000000 // 2048 = 2441
        # compute eval_freq so that we perform a total of num_eval_steps evaluations
        eval_freq = (
            self.total_timesteps // (self.num_envs * self.num_eval_eps)
        ) * self.num_envs  # (5000000 // (16 * 1000)) * 16 = 4992

        # score = 0
        # scores_list = []
        # best_score = -np.inf
        eval_best_score = -np.inf
        best_model_params = {}
        # if we are using a pretrained model
        if self.pretrained:
            # and using anchors
            if self.use_relative:
                # only store anchors obs once
                best_model_params["encoder"] = self.agent.encoder.obs_anchors
        else:
            # otherwise store both the encoder (containing anchor obs, too) and policy params
            best_model_params["encoder"] = self.agent.encoder.state_dict()
        best_model_params["policy"] = self.agent.policy.state_dict()

        """ NOTE: wen updating best_model_params, we only check if pretrained is True.
        No need to check if using anchors here, since if we do not use pretrained we always store encoder params
        regardles of using anchors or not. And if we are, they are already stored the state dict"""

        starting_update = 1

        start_time = time.time()

        for update in range(starting_update, num_updates + 1):
            # for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            for step in range(0, self.num_steps):
                """ EVALUATION """
                # eval and save model
                if global_step % eval_freq == 0:
                    print("### EVALUATION ###")
                    self.agent.eval()
                    eval_rewards, eval_lengths, eval_avg_reward = evaluate_vec_env(
                        agent=self.agent,
                        num_envs=self.num_eval_envs,
                        env=self.eval_envs,
                        global_step=global_step,
                        device=self.device,
                        episode_n=episode_n,
                        writer=self.writer,
                        logger=self.logger,
                    )
                    self.agent.train()
                    # decay best_score to ensure saving a model with converging anchors
                    eval_best_score *= 0.99
                    # save model if it's the best so far
                    if eval_avg_reward >= eval_best_score:
                        eval_best_score = eval_avg_reward
                        # best_score = eval_avg_reward
                        if not self.pretrained:
                            best_model_params["encoder"] = (
                                self.agent.encoder.state_dict()
                            )
                        best_model_params["policy"] = self.agent.policy.state_dict()
                    print(
                        "Current eval_avg_reward: ",
                        eval_avg_reward,
                        "======= eval_best_score: ",
                        eval_best_score,
                    )

                    """ HANDLE MODELS SAVING """
                    if self.track:
                        # make sure to tune `CHECKPOINT_FREQUENCY`
                        # so models are not saved too frequently
                        # if update % CHECKPOINT_FREQUENCY == 0:
                        save_model(
                            wandb=self.wandb,
                            log_path=self.log_path,
                            model_params_dict=best_model_params,
                            save_wandb=self.track,
                        )
                        upload_csv_wandb(self.wandb, self.eval_csv_file_path)
                        upload_csv_wandb(self.wandb, self.csv_file_path)
                """ END OF EVALUATION """

                steps_per_second = int(global_step / (time.time() - start_time))
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(self.device),
                    torch.Tensor(next_done).to(self.device),
                )

                if next_done[0]:
                    episode_n += 1
                    """ LOG DATA TO CSV """
                    # we are logging upon the termination of the first env so that it stays consistent.
                    # using "for item in info" could get data from any env as they finish at different times
                    for info in infos["final_info"]:
                        # Skip the envs that are not done
                        if info is None:
                            continue
                        # logger from meta
                        with self.logger.log_and_dump_ctx(
                            global_step, ty="train"
                        ) as log:
                            log("frames", global_step * self.num_envs)
                            log("step", global_step)
                            log("episode", episode_n)
                            log("episode_reward", info["episode"]["r"])
                            log("episode_length", info["episode"]["l"])
                            log("total_time", info["episode"]["t"])
                            log("fps", steps_per_second)
                            # log('episode', global_episode)
                            # log('buffer_size', len(replay_storage))

                        # Skip the envs that are not done
                        # if info is None:
                        #     continue
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

                        if self.use_relative:
                            # curr_anchors = np.array([encoder.anchors[i].cpu().detach()  for i in range(10)])
                            self.wandb.log(
                                {
                                    f"anchors_embeddings/{i}": self.wandb.Histogram(
                                        self.encoder.anchors[i].cpu().detach()
                                    )
                                    for i in range(10)
                                },
                                step=global_step,
                            )
                            # compute mse between anchors and previous anchors
                            # mse = np.mean(np.square(curr_anchors - previous_anchors))
                            # wandb.log({"anchors_mse": mse}, step=global_step)
                            # previous_anchors = curr_anchors
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    if self.use_relative and not self.pretrained:
                        self.agent.encoder.update_anchors()
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar(
                "losses/explained_variance", explained_var, global_step
            )
            # print("SPS:", int(global_step / (time.time() - start_time)))
            print("SPS:", steps_per_second)
            self.writer.add_scalar("charts/SPS", steps_per_second, global_step)

        """ EVALUATION """
        # eval and save model
        # if global_step % eval_freq == 0:
        print("### EVALUATION ###")
        self.agent.eval()
        eval_rewards, eval_lengths, eval_avg_reward = evaluate_vec_env(
            agent=self.agent,
            num_envs=self.num_eval_envs,
            env=self.eval_envs,
            global_step=global_step,
            device=self.device,
            episode_n=episode_n,
            writer=self.writer,
            logger=self.logger,
        )
        self.agent.train()
        # decay best_score to ensure saving a model with converging anchors
        eval_best_score *= 0.99
        # save model if it's the best so far
        if eval_avg_reward >= eval_best_score:
            eval_best_score = eval_avg_reward
            # best_score = eval_avg_reward
            if not self.pretrained:
                best_model_params["encoder"] = self.agent.encoder.state_dict()
            best_model_params["policy"] = self.agent.policy.state_dict()
        print(
            "Current eval_avg_reward: ",
            eval_avg_reward,
            "======= eval_best_score: ",
            eval_best_score,
        )

        """ HANDLE MODELS SAVING """
        if self.track:
            save_model(
                wandb=self.wandb,
                log_path=self.log_path,
                model_params_dict=best_model_params,
                save_wandb=self.track,
            )
            upload_csv_wandb(self.wandb, self.eval_csv_file_path)
            upload_csv_wandb(self.wandb, self.csv_file_path)

            self.wandb.finish()
        """ END OF EVALUATION """

        self.envs.close()
        self.eval_envs.close()
        self.writer.close()
