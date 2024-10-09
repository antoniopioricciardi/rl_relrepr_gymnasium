# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os

import torch
import torch.optim as optim

# from utils.preprocess_env import PreprocessFrameRGB, RepeatAction

import gin
from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align_not_working import (
    FeatureExtractor,
    Policy,
    Agent,
)
from zeroshotrl.utils.relative import (
    get_obs_anchors,
)  # , init_anchors, init_anchors_from_obs, get_obs_anchors_totensor
# from utils.helputils import save_model, upload_csv_wandb
# from utils.evaluation import evaluate_vec_env


from zeroshotrl.train_ppo import PPOTrainer_vec
# from pytorch_lightning import seed_everything
# from utils.env_initializer import make_env_atari


# env setup
@gin.configurable
def init_stuff_ppo(
    use_relative: bool,
    anchors_path: str,
    anchors_indices_path: str,
    use_resnet: bool,
    pretrained: bool,
    anchors_alpha: float,
    model_path: str,
    learning_rate: float,
    seed,
    total_timesteps,
    num_steps,
    num_eval_eps,
    num_minibatches,
    minibatch_size,
    update_epochs,
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
    envs,
    eval_envs,
    device,
    wandb,
    writer,
    logger,
    log_path,
    csv_file_path,
    eval_csv_file_path,
):
    obs_set = None
    # if we're not using anchors need to obtain observations to use as anchors
    if use_relative:
        # anchor_obs = get_obs_anchors(args.anchors_path) # [:128]
        obs_set = get_obs_anchors(anchors_path)  # , args.anchors_indices_path)

        # open anchor_indices.txt and read the indices
        with open(anchors_indices_path, "r") as f:
            anchor_indices = f.readlines()
        anchor_indices = [int(item.strip()) for item in anchor_indices]
        obs_set = obs_set[anchor_indices, :]

    if use_resnet:
        from zeroshotrl.rl_agents.ppo.ppo_resnet import (
            FeatureExtractorResNet,
            PolicyResNet,
            AgentResNet,
        )

        encoder = FeatureExtractorResNet(
            use_relative=use_relative,
            obs_anchors=obs_set,
            obs_anchors_filename=anchors_path,
        ).to(device)
    else:
        encoder = FeatureExtractor(
            use_relative=use_relative,
            pretrained=pretrained,
            obs_anchors=obs_set,
            anchors_alpha=anchors_alpha,
        ).to(device)

    if use_relative:
        encoder.fit(obs_anchors=obs_set)

    # if we are using a pretrained encoder, load its params into our encoder
    if pretrained:
        model_path = os.path.join(
            "models",
            model_path,
            "encoder.pt",
        )
        encoder_params = torch.load(
            model_path,
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        print(encoder_params.keys())
        encoder.load_state_dict(encoder_params, strict=False)
        encoder.eval()
        encoder.requires_grad_(False)

    previous_anchors = []

    if use_resnet:
        policy = PolicyResNet(
            envs.single_action_space.n,
            use_fc=True,
            encoder_out_dim=encoder.out_dim,
            repr_dim=3136,
        ).to(device)
    else:
        policy = Policy(envs.single_action_space.n).to(device)

    if use_resnet:
        agent = AgentResNet(encoder, policy).to(device)
    else:
        agent = Agent(encoder, policy).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    CHECKPOINT_FREQUENCY = 50

    trainer = PPOTrainer_vec(
        seed=seed,
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_eval_eps=num_eval_eps,
        num_minibatches=num_minibatches,
        minibatch_size=minibatch_size,
        update_epochs=update_epochs,
        envs=envs,
        eval_envs=eval_envs,
        encoder=encoder,
        policy=policy,
        agent=agent,
        use_relative=use_relative,
        pretrained=pretrained or use_resnet,
        optimizer=optimizer,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        learning_rate=learning_rate,
        gamma=gamma,
        norm_adv=norm_adv,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_vloss=clip_vloss,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        anneal_lr=anneal_lr,
        track=track,
        wandb=wandb,
        writer=writer,
        logger=logger,
        log_path=log_path,
        csv_file_path=csv_file_path,
        eval_csv_file_path=eval_csv_file_path,
        device=device,
    )

    trainer.train()
