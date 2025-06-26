import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# from timm.data import resolve_data_config, create_transform
from torchvision.models import resnet18  # , resnet34, resnet50
from torchvision import transforms

# Relative stuff
from latentis.project import RelativeProjector
from latentis.project import relative
from latentis.transform import Centering, StandardScaling


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureExtractorResNet(nn.Module):
    def __init__(
        self,
        use_relative=False,
        obs_anchors=None,
        obs_anchors_filename=None,
        device="cpu",
    ):
        super().__init__()
        self.use_relative = use_relative

        if self.use_relative:
            # obs_anchors_filename is used to recover the obs_anchors when loading the model
            self.register_buffer("obs_anchors_filename", obs_anchors_filename)
            self.obs_anchors = obs_anchors
            self.projector = RelativeProjector(
                projection_fn=relative.cosine_proj,
                abs_transforms=[Centering(), StandardScaling()],
            )

        # resnet part
        self.network = resnet18(pretrained=True).to(device)

        # this transform should be unused
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

        # could be done with a simple self.network.requires_grad_(False)
        # TODO: check which one is better
        for param in self.network.parameters():
            param.requires_grad = False

        self.num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Identity()  # should be unused
        self.repr_dim = 3136  # 1024
        self.image_channel = 3

        # x = torch.randn([32] + [9, 84, 84])
        self.num_frames_stack = 4
        x = torch.randn([16, self.num_frames_stack, 3, 84, 84])
        with torch.no_grad():
            out_shape = self.forward(x).shape
        self.out_dim = out_shape[1]

        self.network.eval()
        self.network.requires_grad_(False)

    # @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        with torch.no_grad():
            # obs = obs / 255.0 - 0.5
            # values in the range from [0, 1] -> [-0.5, 0.5]
            obs = obs - 0.5  # we already divide by 255 in the preprocessing

            # forward_conv is now receiving a (batch_size, 4, 3, 84, 84) tensor instead of (batch_size, 3, 84, 84).
            # We are stacking 4 frames.
            # reshape to (batch_size * 4, 3, 84, 84) -> (16 * 4, 3, 84, 84) -> (64, 3, 84, 84)
            obs = obs.view(
                obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4]
            )
            for name, module in self.network._modules.items():
                obs = module(obs)
                if name == "layer2":
                    break
            # final obs size: (batch_size * 3, 128, 11, 11) -> (16 * 3, 128, 11, 11) -> (64, 128, 11, 11)
            # reshape to (batch_size, 3, 128, 11, 11)
            conv = obs.view(
                obs.shape[0] // self.num_frames_stack,
                self.num_frames_stack,
                obs.shape[1],
                obs.shape[2],
                obs.shape[3],
            )
            if flatten:
                # conv = conv.view(conv.size(0), -1)
                conv = torch.flatten(conv, start_dim=1)  # 46464

        """ code for working with frames stacks """
        # obs = obs / 255.0 - 0.5
        # # time_step = obs.shape[1] // self.image_channel
        # # obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        # # obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        # # obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        # for name, module in self.model._modules.items():
        #     obs = module(obs)
        #     if name == 'layer2':
        #         break

        # # conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
        # # conv_current = conv[:, 1:, :, :, :]
        # # conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        # # conv = torch.cat([conv_current, conv_prev], axis=1)
        # # conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
        # if flatten:
        #     conv = conv.view(conv.size(0), -1)

        return conv

    def set_anchors(self):
        self.anchors = self.network(self.obs_anchors)
        # self.register_buffer("anchors", anchors)

    def _compute_relative_representation(self, hidden):
        return self.projector(x=hidden, anchors=self.anchors)  # .vectors

    def forward(self, obs):
        if self.use_relative:
            with torch.no_grad():
                hidden = self.forward_conv(obs)
                hidden = self._compute_relative_representation(hidden)

                # TODO: hypeperam per cenetering and scaling
                # hidden = (hidden - self.anchors_mean)# / self.anchors_std
                # hidden: torch.Tensor = F.normalize(hidden, p=2, dim=-1)
                # hidden = hidden @ self.anchors.T  # relative representations
        else:
            hidden = self.forward_conv(obs)
        return hidden


class PolicyResNet(nn.Module):
    def __init__(
        self,
        num_actions,
        use_fc=True,
        encoder_out_dim: int = 3136,
        repr_dim: int = 3136,
    ) -> None:
        super().__init__()
        self.encoder_out_dim = encoder_out_dim  # 15488
        self.repr_dim = repr_dim  # 3136
        if use_fc:
            self.fc = nn.Linear(self.encoder_out_dim, self.repr_dim)
        else:
            self.fc = nn.Identity()

        self.network_linear = nn.Sequential(
            self.fc,
            nn.LayerNorm(64 * 7 * 7),
            nn.ReLU(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        # x = self.fc(x)
        # x = self.ln(x)
        x = self.network_linear(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # x = self.fc(x)
        # x = self.ln(x)
        x = self.network_linear(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action_and_value_deterministic(self, x, action=None):
        x = self.network_linear(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            # action = probs.sample()
            # take maximum likelihood action
            action = probs.probs.argmax(dim=1, keepdim=True)[0]
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class AgentResNet(nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractorResNet,
        policy: PolicyResNet,
        translation=None,
    ):
        super().__init__()

        self.encoder = feature_extractor
        self.policy = policy

        self.translation = translation

    def get_value(self, x):
        if self.translation is None:
            return self.policy.get_value(self.encoder(x))
        else:
            return self.policy.get_value(self.translation(self.encoder(x)))

    def get_action_and_value(self, x, action=None):
        if self.translation is None:
            return self.policy.get_action_and_value(self.encoder(x), action=action)
        else:
            return self.policy.get_action_and_value(
                self.translation(self.encoder(x))["target"], action=action
            )

    def get_action_and_value_deterministic(self, x, action=None):
        if self.translation is None:
            return self.policy.get_action_and_value_deterministic(
                self.encoder(x), action=action
            )
        else:
            return self.policy.get_action_and_value_deterministic(
                self.translation(self.encoder(x))["target"], action=action
            )

    def forward(self, x):
        return self.get_action_and_value(x, action=None)


# class FeatureExtractorResNetNoStack(nn.Module):
#     def __init__(self, use_relative=False, obs_anchors=None, anchors_alpha=0.99):
#         super().__init__()
#         self.use_relative = use_relative
#         self.anchors_alpha = anchors_alpha

#         # resnet part
#         self.network = resnet18(pretrained=True)

#         # UNUSED
#         self.transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224)
#             ])

#         if self.use_relative:
#             self.register_buffer("obs_anchors", obs_anchors)
#             anchors = None
#             self.projector = RelativeProjector(
#                 projection_fn=relative.cosine_proj,
#                 abs_transforms=[Centering(), StandardScaling()],
#             )
#         # if self.use_relative:
#         #     self.register_buffer("obs_anchors", obs_anchors)
#         #     anchors = self.forward_conv(self.obs_anchors)
#         #     self.register_buffer("anchors", anchors)
#         #     self.rel_proj = RelativeProjector(
#         #     projection=Projections.COSINE, # EUCLIDEAN, # COSINE
#         #     # anchors=anchors,
#         #     abs_transforms=[Transforms.Centering()], #StandardScaling()],
#         #     # rel_transforms=[Transforms.L2()],
#         #     )


#         # could be done with a simple self.network.requires_grad_(False)
#         # TODO: check which one is better
#         self.network.requires_grad_(False)
#         self.network.eval()
#         # for param in self.network.parameters():
#         #     param.requires_grad = False

#         self.network.fc = nn.Identity() # should be unused
#         self.repr_dim = 3136 # 1024
#         self.image_channel = 3

#         x = torch.randn([1, 3, 84, 84]) # generate a sample input
#         with torch.no_grad():
#             out_shape = self.forward_conv(x).shape
#         self.out_dim = out_shape[1]

#         # self.fc = nn.Linear(self.out_dim, self.repr_dim)
#         # self.ln = nn.LayerNorm(self.repr_dim)
#         # # Initialization
#         # nn.init.orthogonal_(self.fc.weight.data)
#         # self.fc.bias.data.fill_(0.0)


#     def _compute_relative_representation(self, hidden):
#         return self.rel_proj(x=hidden, anchors=self.anchors)

#     # @torch.no_grad()
#     def forward_conv(self, obs, flatten=True):
#         with torch.no_grad():
#             # obs = obs / 255.0 - 0.5
#             obs = obs - 0.5 # we already divide by 255 in the preprocessing
#             for name, module in self.network._modules.items():
#                 obs = module(obs)
#                 if name == 'layer2':
#                     break
#             if flatten:
#                 # conv = conv.view(conv.size(0), -1)
#                 conv = torch.flatten(obs, start_dim=1)

#         """ code for working with frames stacks """
#         # obs = obs / 255.0 - 0.5
#         # # time_step = obs.shape[1] // self.image_channel
#         # # obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
#         # # obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

#         # # obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
#         # for name, module in self.model._modules.items():
#         #     obs = module(obs)
#         #     if name == 'layer2':
#         #         break

#         # # conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
#         # # conv_current = conv[:, 1:, :, :, :]
#         # # conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
#         # # conv = torch.cat([conv_current, conv_prev], axis=1)
#         # # conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
#         # if flatten:
#         #     conv = conv.view(conv.size(0), -1)

#         return conv


#     # no need for gradients because we are using the pretrained encoder
#     @torch.no_grad()
#     def forward(self, x):
#         hidden = self.forward_conv(x)
#         if self.use_relative:
#                 hidden = self._compute_relative_representation(hidden)
#         return hidden

#     @torch.no_grad()
#     def update_anchors(self):
#         """ BEWARE. During testing, this must be called after the model params are loaded. """
#         self.anchors = self.forward_conv(self.obs_anchors)

#     def set_anchors(self):
#         self.anchors = self.forward_conv(self.obs_anchors)
