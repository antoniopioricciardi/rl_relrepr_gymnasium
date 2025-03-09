# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import time

import numpy as np
import torch
import torch.nn as nn
import os

# from utils.relative import init_anchors, init_anchors_from_obs, get_obs_anchors, get_obs_anchors_totensor
from zeroshotrl.utils.helputils import save_model, upload_csv_wandb
from zeroshotrl.utils.evaluation import evaluate_vec_env

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from zeroshotrl.logger import CustomLogger
from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent

from zeroshotrl.utils.env_initializer import init_env


class PPOFinetune:
    def __init__(self, agent, eval_agent, env_id, envs, eval_envs, seed, total_timesteps, learning_rate,
                 num_eval_eps, track, exp_name, wandb_project_name, wandb_entity,
                 device, args):
        self.agent = agent
        self.envs = envs
        self.eval_envs = eval_envs
        self.learning_rate = learning_rate
        self.device = device

        self.seed = seed
        self.total_timesteps = total_timesteps
        self.num_steps = 128
        self.num_eval_eps = num_eval_eps

        self.num_envs = envs.num_envs
        self.num_eval_envs = eval_envs.num_envs

        self.batch_size = int(self.num_envs * self.num_steps)  # 16 * 128 = 2048
        num_minibatches = 4
        self.minibatch_size = int(self.batch_size // num_minibatches)  # 2048 // 4 = 512

        self.envs = envs
        self.eval_envs = eval_envs
        # self.encoder = encoder
        # self.policy = policy
        self.agent = agent
        self.eval_agent = eval_agent
        # writer = SummaryWriter("finetuning/custom")
        # logger = CustomLogger(
        # "finetuning", use_tb=False, use_wandb=False
        # )  # True if args.track else False)
        # self.writer = writer
        # self.logger = logger
        self.CHECKPOINT_FREQUENCY = 50

        self.update_epochs = 4
        self.learning_rate = learning_rate
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
        self.optimizer = optimizer
        self.gamma = 0.99
        self.use_relative = False
        self.pretrained = True
        self.norm_adv = True
        self.gae_lambda = 0.95
        self.clip_coef = 0.1
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.clip_vloss = True
        self.max_grad_norm = 0.5
        self.target_kl = None

        self.anneal_lr = True

        self.track = args.track
        # self.log_path = log_path

        run_name = f"{exp_name}_{seed}__{int(time.time())}"# f"{env_id}__{exp_name}_{seed}__{int(time.time())}"
        eval_run_name = run_name + "_eval"
        self.wandb = None
        if self.track:
            import wandb
            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.wandb = wandb
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # paths for saving models and csv files, etc. # TODO: create a function for this
        if not args.track:
            log_path = os.path.join("runs", run_name)  # f"runs/{run_name}/"
        else:
            log_path = f"{self.wandb.run.dir}"

        # create logger
        # logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # work_dir = Path.cwd() / "runs" / run_name
        self.logger = CustomLogger(
            log_path, use_tb=False, use_wandb=False
        )  # True if args.track else False)
        csv_file_name = "train"
        self.csv_file_path = os.path.join(log_path, f"{csv_file_name}.csv")
        eval_csv_file_name = "eval"
        self.eval_csv_file_path = os.path.join(log_path, f"{eval_csv_file_name}.csv")

        # self.csv_file_path = csv_file_path
        # self.eval_csv_file_path = eval_csv_file_path

        self.device = device

    def train(self):
        anchors_upd_step = 0
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
                    # self.agent.eval()
                    # eval_agent.eval()
                    self.eval_agent.load_state_dict(self.agent.state_dict())
                    self.eval_agent.eval()
                    eval_rewards, eval_lengths, eval_avg_reward = evaluate_vec_env(
                        agent=self.eval_agent,
                        num_envs=self.num_eval_envs,
                        env=self.eval_envs,
                        global_step=global_step,
                        device=self.device,
                        episode_n=episode_n,
                        writer=self.writer,
                        logger=self.logger,
                    )
                    # self.agent.train()
                    # decay best_score to ensure saving a model with converging anchors
                    eval_best_score *= 0.99
                    # save model if it's the best so far
                    if eval_avg_reward >= eval_best_score:
                        eval_best_score = eval_avg_reward
                        # best_score = eval_avg_reward
                        if not self.pretrained:
                            best_model_params["encoder"] = self.agent.encoder.state_dict()
                            if self.use_relative:
                                # save anchors buffer and then upload the model to wandb.
                                # this is to ensure that the model is saved with the updated anchors just once
                                # to avoid saving heavy models with anchors at each evaluation
                                self.agent.encoder.save_anchors_buffer()

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
                        # save_model(
                        #     wandb=self.wandb,
                        #     log_path=self.log_path,
                        #     model_params_dict=best_model_params,
                        #     save_wandb=self.track,
                        # )
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
                if self.use_relative and not self.pretrained:
                    # self.agent.encoder.update_anchors()
                    self.agent.encoder.update_anchors(anchors_upd_step, num_updates*self.update_epochs)
                    anchors_upd_step += 1
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    # if self.use_relative and not self.pretrained:
                    #     self.agent.encoder.update_anchors()
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

            # # log the dynamic alpha and feature variance
            # self.writer.add_scalar(
            #     "anchors/dynamic_alpha", self.agent.encoder.dynamic_alpha, global_step
            # )
            # self.writer.add_scalar(
            #     "anchors/feature_variance", self.agent.encoder.feature_variance, global_step
            # )

        """ EVALUATION """
        # eval and save model
        # if global_step % eval_freq == 0:
        print("### EVALUATION ###")
        # self.agent.eval()

        self.eval_agent.load_state_dict(self.agent.state_dict())
        self.eval_agent.eval()
        eval_rewards, eval_lengths, eval_avg_reward = evaluate_vec_env(
            agent=self.eval_agent,# self.agent,
            num_envs=self.num_eval_envs,
            env=self.eval_envs,
            global_step=global_step,
            device=self.device,
            episode_n=episode_n,
            writer=self.writer,
            logger=self.logger,
        )
        # self.agent.train()
        # decay best_score to ensure saving a model with converging anchors
        eval_best_score *= 0.99
        # save model if it's the best so far
        if eval_avg_reward >= eval_best_score:
            eval_best_score = eval_avg_reward
            # best_score = eval_avg_reward
            if not self.pretrained:
                best_model_params["encoder"] = self.agent.encoder.state_dict()
                if self.use_relative:
                    # save anchors buffer and then upload the model to wandb.
                    # this is to ensure that the model is saved with the updated anchors just once
                    # to avoid saving heavy models with anchors at each evaluation
                    self.agent.encoder.save_anchors_buffer()
            best_model_params["policy"] = self.agent.policy.state_dict()
        print(
            "Current eval_avg_reward: ",
            eval_avg_reward,
            "======= eval_best_score: ",
            eval_best_score,
        )

        """ HANDLE MODELS SAVING """

        if self.track:
            # save_model(
            #     wandb=self.wandb,
            #     log_path=self.log_path,
            #     model_params_dict=best_model_params,
            #     save_wandb=self.track,
            # )
            upload_csv_wandb(self.wandb, self.eval_csv_file_path)
            upload_csv_wandb(self.wandb, self.csv_file_path)

            self.wandb.finish()
        """ END OF EVALUATION """

        self.envs.close()
        self.eval_envs.close()
        self.writer.close()



if __name__ == "__main__":
    # finetuning = False
    # if finetuning:
    import gymnasium as gym
    # env setup
    from zeroshotrl.utils.env_initializer import make_env_atari
    from zeroshotrl.utils.models import init_stuff

    # parse argsuments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CarRacing-v2")
    parser.add_argument("--background-color", type=str, default="green")
    parser.add_argument("--encoder-dir", type=str, default="models/encoder")
    parser.add_argument("--policy-dir", type=str, default="models/policy")
    parser.add_argument("--anchors-file1", type=str, default="data/obs_set_1.pt")
    parser.add_argument("--anchors-file2", type=str, default="data/obs_set_2.pt")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--num-eval-eps", type=int, default=20)
    parser.add_argument("--use-resnet", type=bool, default=False)
    parser.add_argument("--anchors-method", type=str, default="fps")
    parser.add_argument("--stitching-mode", type=str, default="relative")
    parser.add_argument("--zoom", type=bool, default=2.7)
    parser.add_argument("--env-seed", type=int, default=1)
    parser.add_argument("--track", type=bool, default=False)
    # parser.add_argument("--exp_name", type=str, default="finetune")
    parser.add_argument("--wandb-project-name", type=str, default="finetune")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ Parameters to change for single test """
    env_id = args.env_id 
    model_color_1 = args.background_color
    encoder_dir = args.encoder_dir
    policy_dir = args.policy_dir
    anchors_file1 = args.anchors_file1
    anchors_file2 = args.anchors_file2
    use_resnet = args.use_resnet
    anchoring_method = args.anchors_method  # "fps"  # "fps", "kmeans", "random"
    num_eval_eps = args.num_eval_eps
    total_timesteps = args.total_timesteps
    learning_rate = args.learning_rate

    exp_name = args.env_id + "-" + args.background_color + "-" + args.stitching_mode 
    model_color_2 = "--" # args.policy_color

    model_algo_1 = "ppo"
    model_algo_2 = "ppo"
    env_info = "rgb"

    stitching_md = args.stitching_mode
    image_path = ""
    relative = False
    if stitching_md == "relative":
        relative = True
    
    num_finetune_envs = 16
    num_eval_envs = 2

    finetune_envs = init_env(
        env_id,
        env_info,
        background_color=args.background_color,
        image_path=image_path,
        zoom=args.zoom,
        cust_seed=args.env_seed,
        render_md="rgb_array",
        num_envs=num_finetune_envs,
        sync_async="async"
    )
    eval_envs = init_env(
        env_id,
        env_info,
        background_color=args.background_color,
        image_path=image_path,
        zoom=args.zoom,
        cust_seed=args.env_seed,
        render_md="rgb_array",
        num_envs=num_eval_envs,
        sync_async="async"
    )


    agent, encoder1, policy2 = init_stuff(finetune_envs, env_info, model_algo_1, model_algo_2,
               model_color_1, model_color_2, encoder_dir, policy_dir, anchors_file1, anchors_file2, use_resnet,
               device, relative, anchoring_method, stitching_md, num_envs=num_finetune_envs, set_eval=False)
    
    agent.policy.train()
    # agent.encoder.train()
    agent.translation.train()
    agent.encoder.eval()
    # agent.translation.eval()
    for param in agent.encoder.parameters():
        param.requires_grad = False # False
    for param in agent.policy.parameters():
        param.requires_grad = True
    # for param in agent.translation.parameters():
    #     param.requires_grad = False

    from zeroshotrl.finetune import PPOFinetune
    print("Starting finetuning...")
    # finetuner = PPOFinetune(agent, finetune_envs, eval_envs, seed=1, total_timesteps=200000, learning_rate=0.00005, device=device)

    import copy
    eval_agent = copy.deepcopy(agent)
    eval_agent.eval()

    print(agent.policy.training)

    finetuner = PPOFinetune(agent, eval_agent, env_id, finetune_envs, eval_envs, seed=1, total_timesteps=args.total_timesteps, learning_rate=args.learning_rate,
                            num_eval_eps=args.num_eval_eps, track=args.track,
                            exp_name=exp_name, wandb_project_name=args.wandb_project_name,
                            wandb_entity=None, device=device, args=args)
    finetuner.train()
    print("Finetuning done.")


# exmaple usage:
# python src/zeroshotrl/finetune.py --track True --wandb-project-name finetuning --stitching-mode translate --env-id CarRacing-v2 --env-seed 1 --background-color green --encoder-dir models/CarRacing-v2/rgb/green/ppo/absolute/relu/seed_1 --policy-dir models/CarRacing-v2/rgb/blue/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/CarRacing-v2/rgb_ppo_transitions_green_obs.pkl --anchors-file2 data/anchors/CarRacing-v2/rgb_ppo_transitions_blue_obs.pkl --total-timesteps 200000 --learning-rate 0.00005 --num-eval-eps 20  --anchors-method random

""" LUNARLANDER """
# python src/zeroshotrl/finetune.py --track True --wandb-project-name finetuning --stitching-mode translate --env-id LunarLanderRGB --env-seed 1 --background-color white --encoder-dir models/LunarLanderRGB/rgb/white/ppo/absolute/relu/seed_1 --policy-dir models/LunarLanderRGB/rgb/red/ppo/absolute/relu/seed_2 --anchors-file1 data/anchors/LunarLanderRGB/rgb_ppo_transitions_white_obs.pkl --anchors-file2 data/anchors/LunarLanderRGB/rgb_ppo_transitions_red_obs.pkl --total-timesteps 2500000 --learning-rate 0.00005 --num-eval-eps 250  --anchors-method random

""" LUNARLANDER -3 """
# python src/zeroshotrl/finetune.py --track True --wandb-project-name finetuning --stitching-mode translate --env-id LunarLanderRGB-3 --env-seed 1 --background-color red --encoder-dir models/LunarLanderRGB/rgb/red/ppo/absolute/relu/seed_1 --policy-dir models/LunarLanderRGB-3/rgb/white/ppo/absolute/relu/seed_1 --anchors-file1 data/anchors/LunarLanderRGB/rgb_ppo_transitions_red_obs.pkl --anchors-file2 data/anchors/LunarLanderRGB/rgb_ppo_transitions_white_obs.pkl --total-timesteps 2500000 --learning-rate 0.00005 --num-eval-eps 250  --anchors-method random