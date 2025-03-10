# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os

import torch
import torch.optim as optim

# from utils.preprocess_env import PreprocessFrameRGB, RepeatAction


from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import (
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
def init_stuff_ppo(
    args,
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
    if args.use_relative:
        # anchor_obs = get_obs_anchors(args.anchors_path) # [:128]
        obs_set = get_obs_anchors(args.anchors_path)  # , args.anchors_indices_path)

        # open anchor_indices.txt and read the indices
        with open(args.anchors_indices_path, "r") as f:
            anchor_indices = f.readlines()
        anchor_indices = [int(item.strip()) for item in anchor_indices]
        obs_set = obs_set[anchor_indices, :]
        obs_set.to(device)

    if args.use_resnet:
        from zeroshotrl.rl_agents.ppo.ppo_resnet import (
            FeatureExtractorResNet,
            PolicyResNet,
            AgentResNet,
        )

        encoder = FeatureExtractorResNet(
            use_relative=args.use_relative,
            obs_anchors=obs_set,
            obs_anchors_filename=args.anchors_path,
        ).to(device)
    else:
        encoder = FeatureExtractor(
            use_relative=args.use_relative,
            pretrained=args.pretrained,
            # obs_anchors_filename=args.anchors_path,
            # obs_anchors=obs_set,
            anchors_alpha=args.anchors_alpha,
        ).to(device)
        # encoder.fit(obs_anchors=obs_set)

    if args.use_relative:
        encoder.set_obs_anchors(obs_anchors=obs_set)

    # if we are using a pretrained encoder, load its params into our encoder
    if args.pretrained:
        model_path = os.path.join(
            "models",
            args.model_path,
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

    # previous_anchors = []

    if args.use_resnet:
        policy = PolicyResNet(
            envs.single_action_space.n,
            use_fc=True,
            encoder_out_dim=encoder.out_dim,
            repr_dim=3136,
        ).to(device)
    else:
        policy = Policy(envs.single_action_space.n).to(device)

    if args.use_resnet:
        agent = AgentResNet(encoder, policy).to(device)
    else:
        agent = Agent(encoder, policy).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    CHECKPOINT_FREQUENCY = 50

    trainer = PPOTrainer_vec(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        num_eval_eps=args.num_eval_eps,
        num_minibatches=args.num_minibatches,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        envs=envs,
        eval_envs=eval_envs,
        encoder=encoder,
        policy=policy,
        agent=agent,
        use_relative=args.use_relative,
        pretrained=args.pretrained or args.use_resnet,
        optimizer=optimizer,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        norm_adv=args.norm_adv,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_vloss=args.clip_vloss,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        anneal_lr=args.anneal_lr,
        track=args.track,
        wandb=wandb,
        writer=writer,
        logger=logger,
        log_path=log_path,
        csv_file_path=csv_file_path,
        eval_csv_file_path=eval_csv_file_path,
        device=device,
    )

    trainer.train()