import os
import torch
import torch.nn as nn



def _build_path(
    env_id,
    env_info,
    model_color,
    algo,
    is_relative,
    model_type,
    anchors_alpha,
    seed=None,
    name="encoder",
):
    path = f"models/{env_id}/{env_info}/{model_color}/{algo}/"  # _{model_color}/
    path += "relative/" if is_relative else "absolute/"
    path += f"{model_type}/"
    path += f"a_{anchors_alpha}/" if anchors_alpha is not None else ""
    path += f"seed_{seed}/" if seed is not None else ""
    path += f"{name}.pt"
    return path


def load_encoder(
    env_id,
    env_info,
    is_relative,
    encoder_model_color,
    encoder_algo,
    encoder_model_type,
    is_pretrained,
    FeatureExtractor: nn.Module,
    anchors_alpha,
    encoder_seed=None,
    encoder_eval=True,
    device="cpu",
):
    encoder_path = _build_path(
        env_id,
        env_info,
        encoder_model_color,
        encoder_algo,
        is_relative,
        encoder_model_type,
        anchors_alpha,
        encoder_seed,
        name="encoder",
    )

    encoder_params = torch.load(
        encoder_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    obs_anchors = None
    # if is_relative:
    #     obs_anchors = encoder_params["obs_anchors"]

    encoder = FeatureExtractor(
        use_relative=is_relative,
        pretrained=is_pretrained,
        obs_anchors=obs_anchors,
        anchors_alpha=anchors_alpha,
    ).to(device)

    # print(encoder_params.keys())
    encoder.load_state_dict(encoder_params, strict=False)

    if is_relative:
        encoder.set_anchors()  # update_anchors()

    if encoder_eval:
        encoder.eval()
        encoder.requires_grad_(False)
    return encoder


def load_model(
    env_id,
    env_info,
    is_relative,
    encoder_model_color,
    encoder_algo,
    encoder_model_type,
    policy_model_color,
    policy_algo,
    policy_model_type,
    action_space,
    is_pretrained,
    FeatureExtractor: nn.Module,
    Policy: nn.Module,
    Agent: nn.Module,
    anchors_alpha,
    encoder_seed=None,
    policy_seed=None,
    encoder_eval=True,
    policy_eval=True,
    device="cpu",
):
    encoder = load_encoder(
        env_id,
        env_info,
        is_relative,
        encoder_model_color,
        encoder_algo,
        encoder_model_type,
        is_pretrained,
        FeatureExtractor,
        anchors_alpha,
        encoder_seed,
        encoder_eval,
        device,
    )

    policy_path = _build_path(
        env_id,
        env_info,
        policy_model_color,
        policy_algo,
        is_relative,
        policy_model_type,
        anchors_alpha,
        policy_seed,
        name="policy",
    )
    # anchors = None
    policy_params = torch.load(policy_path, map_location="cpu")
    policy = Policy(num_actions=action_space).to(device)
    # print shape of last layer of policy
    policy.load_state_dict(policy_params)

    # if encoder_eval:
    #     encoder.eval()
    #     encoder.requires_grad_(False)
    if policy_eval:
        policy.eval()
        policy.requires_grad_(False)

    agent = Agent(encoder, policy).to(device)

    return encoder, policy, agent


def load_model_from_path(
    encoder_path,
    policy_path,
    action_space,
    FeatureExtractor: nn.Module,
    Policy: nn.Module,
    Agent: nn.Module,
    encoder_eval=True,
    policy_eval=True,
    is_relative=False,
    is_pretrained=False,
    anchors_alpha=None,
    device="cpu",
):
    # encoder_params = torch.load(encoder_path, map_location="cpu")
    # policy_params = torch.load(policy_path, map_location="cpu")
    
    encoder = load_encoder_from_path(
        encoder_path,
        FeatureExtractor,
        is_relative,
        is_pretrained,
        anchors_alpha=anchors_alpha,
        encoder_eval=encoder_eval,
        device=device,
    )
    
    # anchors = None
    policy = load_policy_from_path(
        policy_path, action_space, Policy, policy_eval, device=device
    )

    agent = Agent(encoder, policy).to(device)
    return encoder, policy, agent


def load_encoder_from_path(
    env_info,
    encoder_path,
    FeatureExtractor: nn.Module,
    is_relative=False,
    is_pretrained=False,
    anchors_alpha=None,
    encoder_eval=True,
    device="cpu",
    state_dim=None,
):
    encoder_params = torch.load(
        encoder_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # obs_anchors = None
    # anchors_filename = None
    # if is_relative:
    #     # obs_anchors = encoder_params["obs_anchors"]

    #     # anchors_filename = encoder_params["obs_anchors_filename"]
    #     obs_anchors = torch.load(
    #         anchors_filename,
    #         map_location="cuda:0" if torch.cuda.is_available() else "cpu",
    #     )
    #     obs_anchors = torch.tensor(obs_anchors).to(device)
    if env_info is None:
        env_info = "rgb"

    if env_info == "rgb":
        encoder = FeatureExtractor(
            use_relative=is_relative,
            pretrained=is_pretrained,
            # obs_anchors=obs_anchors,
            # obs_anchors_filename=anchors_filename,
            anchors_alpha=anchors_alpha,
        )
    elif env_info == "states":
        encoder = FeatureExtractor(
            state_dim=state_dim,
            use_relative=is_relative,
            pretrained=is_pretrained,
            # obs_anchors=obs_anchors,
            # obs_anchors_filename=anchors_filename,
            anchors_alpha=anchors_alpha,
        )
    # if is_relative:
    #     encoder.set_anchors(obs_anchors)

    encoder.load_state_dict(encoder_params, strict=False)
    encoder.to(device)

    if is_relative:
        anchors = encoder_params["saved_anchors"]
        encoder.set_anchors(anchors)  # update_anchors()

    if encoder_eval:
        encoder.eval()
        encoder.requires_grad_(False)
    return encoder


def load_policy_from_path(
    policy_path,
    action_space,
    Policy: nn.Module,
    policy_eval=True,
    encoder_out_dim=None,
    repr_dim=None,
    device="cpu",
):
    policy_params = torch.load(
        policy_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if encoder_out_dim is None:
        policy = Policy(num_actions=action_space).to(device)
    else:
        # we are using resnet model
        policy = Policy(
            num_actions=action_space, encoder_out_dim=encoder_out_dim, repr_dim=repr_dim
        ).to(device)
    # print shape of last layer of policy
    policy.load_state_dict(policy_params)

    if policy_eval:
        policy.eval()
        policy.requires_grad_(False)

    policy = policy.to(device)
    return policy


def get_algo_instance(encoder_algo, policy_algo, use_resnet):
    if encoder_algo.startswith("ppo"):
        if encoder_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu import FeatureExtractor
        else:
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor
        encoder_instance = FeatureExtractor

    elif encoder_algo.startswith("ddqn"):
        if encoder_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end import FeatureExtractorDDQN
        # else:
        #     from src.zeroshotrl.rl_agents.ddqn.ddqn_end_to_end_stack import FeatureExtractorDDQN
        encoder_instance = FeatureExtractorDDQN

    if policy_algo.startswith("ppo"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu import Policy
        else:
            if use_resnet:
                from zeroshotrl.rl_agents.ppo.ppo_resnet import PolicyResNet

                policy_instance = PolicyResNet
            else:
                from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import Policy

                policy_instance = Policy

    elif policy_algo.startswith("ddqn"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end import PolicyDDQN
        # else:
        #     from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end_stack import PolicyDDQN
        policy_instance = PolicyDDQN
    # if encoder_model=='resnet':
    #     agent = AgentResNet

    if policy_algo.startswith("ppo"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu import Agent
        else:
            if use_resnet:
                from zeroshotrl.rl_agents.ppo.ppo_resnet import AgentResNet

                agent_instance = AgentResNet
            else:
                from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import Agent

                agent_instance = Agent
    elif policy_algo.startswith("ddqn"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end import AgentDDQN
        # else:
        #     from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end_stack import AgentDDQN
        agent_instance = AgentDDQN
    return encoder_instance, policy_instance, agent_instance

def get_algo_instance_states(encoder_algo, policy_algo):
    if encoder_algo.startswith("ppo"):
        from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import StateExtractor
        encoder_instance = StateExtractor
    if policy_algo.startswith("ppo"):
        from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import Policy, Agent
        policy_instance = Policy
        agent_instance = Agent

    return encoder_instance, policy_instance, agent_instance


def get_algo_instance_bw(encoder_algo, policy_algo):
    if encoder_algo.startswith("ppo"):
        if encoder_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu import FeatureExtractor
        else:
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align_bw import (
                FeatureExtractor,
            )
        encoder_instance = FeatureExtractor

    elif encoder_algo.startswith("ddqn"):
        if encoder_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end import FeatureExtractorDDQN
        # else:
        #     from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end_stack import FeatureExtractorDDQN
        encoder_instance = FeatureExtractorDDQN

    if policy_algo.startswith("ppo"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu import Policy
        else:
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align_bw import Policy
        policy_instance = Policy

    elif policy_algo.startswith("ddqn"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end import PolicyDDQN
        # else:
        #     from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end_stack import PolicyDDQN
        policy_instance = PolicyDDQN
    # if encoder_model=='resnet':
    #     agent = AgentResNet

    if policy_algo.startswith("ppo"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu import Agent
        else:
            from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align_bw import Agent
        agent_instance = Agent
    elif policy_algo.startswith("ddqn"):
        if policy_algo.endswith("nostack"):
            from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end import AgentDDQN
        # else:
        #     from zeroshotrl.rl_agents.ddqn.ddqn_end_to_end_stack import AgentDDQN
        agent_instance = AgentDDQN
    return encoder_instance, policy_instance, agent_instance




def init_stuff(envs, env_info, model_algo_1, model_algo_2,
               model_color_1, model_color_2, encoder_dir, policy_dir, anchors_file1, anchors_file2, use_resnet,
               device, relative, anchoring_method, stitching_md,
               num_envs=1, set_eval=True, state_dim=None):
    if stitching_md == "translate":
        assert anchors_file1 is not None and anchors_file2 is not None, "Anchors file is required for translation"
    from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor
    if env_info == "rgb":
        encoder_instance, policy_instance, agent_instance = get_algo_instance(
            model_algo_1, model_algo_2, use_resnet=use_resnet
        )
    elif env_info == "states":
        from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import StateExtractor
        encoder_instance, policy_instance, agent_instance = get_algo_instance_states(
            model_algo_1, model_algo_2
        )
    else:
        encoder_instance, policy_instance, agent_instance = get_algo_instance_bw(
            model_algo_1, model_algo_2
        )

    if not use_resnet:
        path1_enc = os.path.join(encoder_dir, "encoder.pt")
        path2_enc = os.path.join(policy_dir, "encoder.pt")
    # path1_pol = os.path.join(encoder_dir, "policy.pt")
    path2_pol = os.path.join(policy_dir, "policy.pt")

    random_encoder = False
    if random_encoder:
        obs_anchors = None
        # if is_relative:
        #     obs_anchors = encoder_params["obs_anchors"]
        encoder1 = FeatureExtractor(
            use_relative=relative,
            pretrained=False,
            obs_anchors=obs_anchors,
            anchors_alpha=None,
        ).to(device)
    elif use_resnet:
        from rl_agents.ppo.ppo_resnet import FeatureExtractorResNet

        encoder1 = FeatureExtractorResNet().to(device)
    else:
        if env_info == "rgb":
            print('enc1')
            encoder1 = load_encoder_from_path(
                path1_enc,
                encoder_instance,
                is_relative=relative,
                is_pretrained=False,
                anchors_alpha=None,
                encoder_eval=set_eval,
                device=device,
            )
        elif env_info == "states":
            print('enc1 states')
            encoder1 = load_encoder_from_path(
                env_info,
                path1_enc,
                encoder_instance,
                is_relative=relative,
                is_pretrained=False,
                anchors_alpha=None,
                encoder_eval=set_eval,
                device=device,
                state_dim=state_dim,
            )
    if use_resnet:
        policy2 = load_policy_from_path(
            path2_pol,
            envs.single_action_space.n,
            policy_instance,
            policy_eval=set_eval,
            encoder_out_dim=encoder1.out_dim,
            repr_dim=3136,
            device=device,
        )
    else:
        print('pol2')
        encoder2, policy2, agent2 = load_model_from_path(
            path2_enc,
            path2_pol,
            envs.single_action_space.n,
            encoder_instance,
            policy_instance,
            agent_instance,
            encoder_eval=set_eval,
            policy_eval=set_eval,
            is_relative=False,
            is_pretrained=False,
            device=device,
        )

    translation_layer = None
    if stitching_md == "translate":
        from zeroshotrl.utils.translation import translate
        agent, encoder1, policy2, translation_layer = translate(anchors_file1, anchors_file2, encoder_dir, encoder1, encoder2, policy2, model_color_1, model_color_2, anchoring_method, use_resnet, num_envs, device)
    else:
        from zeroshotrl.rl_agents.ppo.ppo_end_to_end_relu_stack_align import FeatureExtractor, Agent
        agent = Agent(encoder1, policy2).to(device)
    return agent, encoder1, policy2
