# import numpy as np
# import torch
# import torch.nn as nn
# from torch.distributions.categorical import Categorical

# # Relative stuff
# from latentis.transform.base import StandardScaling
# from latentis.transform.projection import cosine_proj, relative_projection
# from latentis.transform import XTransformSequence
# from latentis.space import Space


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


# class FeatureExtractor(nn.Module):
#     def __init__(
#         self,
#         use_relative=False,
#         pretrained=False,
#         obs_anchors=None,
#         # obs_anchors_filename=None,
#         anchors_alpha=0.99,
#         device="cpu",
#     ):
#         super().__init__()
#         self.use_relative = use_relative
#         self.pretrained = pretrained
#         self.anchors_alpha = anchors_alpha
#         # self.anchors = None # to be computed
#         # self.anchors_mean = None # to be computer
#         # self.observation_anchors = obs_anchors
#         self.network = nn.Sequential(
#             layer_init(nn.Conv2d(3, 32, 8, stride=4)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(32, 64, 4, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(64, 64, 3, stride=1)),
#             nn.Flatten(),
#         )

#         if self.use_relative:
#             # obs_anchors_filename is used to recover the obs_anchors when loading the model
#             # self.register_buffer("obs_anchors_filename", obs_anchors_filename)
#             # self.obs_anchors = obs_anchors
#             self.rel_transform = StandardScaling()# XTransformSequence(transforms=[StandardScaling()])

#             # self.projector = RelativeProjector(
#             #     projection_fn=relative.cosine_proj,
#             #     abs_transforms=[Centering(), StandardScaling()],
#             # )
#         # if self.use_relative:
#         #     # obs_anchors_filename is used to recover the obs_anchors when loading the model
#         #     # self.register_buffer("obs_anchors_filename", obs_anchors_filename)
#         #     # self.register_buffer("obs_anchors", obs_anchors)
#         #     self.obs_anchors = obs_anchors
#         #     # anchors = None
#         #     self.projector = RelativeProjector(
#         #         projection_fn=relative.cosine_proj,
#         #         abs_transforms=[Centering(), StandardScaling()],
#         #     )
#         #     # self.set_anchors()

#     def _compute_relative_representation(self, hidden):
#         print(hidden.shape, self.obs_anchors.shape)
#         rel_space = relative_projection(
#             x=self.rel_transform.transform(hidden),
#             anchors=self.rel_transform.fit(self.obs_anchors),
#             projection_fn=cosine_proj,
#         )
#         return rel_space
#         # return self.projector(x=hidden, anchors=self.anchors)

#     def forward(self, x):
#         num_stack = x.shape[1]
#         x = x.view(
#             x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]
#         )  # (batch_size * stack, 3, 84, 84)

#         if self.pretrained:
#             # TODO: se uso pretrained ricorda model.eval()
#             with torch.no_grad():
#                 hidden = self.network(x)
#             if self.use_relative:
#                 hidden = self._compute_relative_representation(hidden)
#         else:
#             hidden = self.network(x)
#             if self.use_relative:
#                 hidden = self._compute_relative_representation(hidden)

#         hidden = hidden.view(-1, num_stack, hidden.shape[1])
#         # flatten the last two dimensions
#         hidden = hidden.view(
#             hidden.shape[0], num_stack * hidden.shape[2]
#         )  # (batch_size, num_stack * 3136) -> (batch_size, 12544)
#         return hidden

#     def forward_single(self, x):
#         return self.network(x)

#     @torch.no_grad()
#     def update_anchors(self):
#         """BEWARE. During testing, this must be called after the model params are loaded."""
#         new_anchors = self.network(self.obs_anchors)
#         self.anchors = (
#             self.anchors_alpha * self.anchors + (1 - self.anchors_alpha) * new_anchors
#         )  # keep % of the old anchors # 0.99 and 0.999

#     @torch.no_grad()
#     def fit(self, obs_anchors):
#         self.obs_anchors = obs_anchors
#         self.anchors = self.network(obs_anchors)
#         self.rel_transform.fit(self.anchors)
#         # self.register_buffer("anchors", anchors)


# class Policy(nn.Module):
#     def __init__(self, num_actions, stack_n: int = 4) -> None:
#         super().__init__()
#         self.network_linear = nn.Sequential(
#             nn.LayerNorm(stack_n * 64 * 7 * 7),
#             nn.ReLU(),
#             layer_init(nn.Linear(stack_n * 64 * 7 * 7, 512)),
#             nn.ReLU(),
#         )
#         self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
#         self.critic = layer_init(nn.Linear(512, 1), std=1)

#     def get_value(self, x):
#         x = self.network_linear(x)
#         return self.critic(x)

#     def get_action_and_value(self, x, action=None):
#         x = self.network_linear(x)
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x)

#     def get_action_and_value_deterministic(self, x, action=None):
#         x = self.network_linear(x)
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             # action = probs.sample()
#             # take maximum likelihood action
#             action = probs.probs.argmax(dim=1, keepdim=True)[0]
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# class Agent(nn.Module):
#     def __init__(
#         self, feature_extractor: FeatureExtractor, policy: Policy, translation=None
#     ):
#         super().__init__()

#         self.encoder = feature_extractor
#         self.policy = policy

#         self.translation = translation

#     def get_encoded_obs(self, x):
#         if self.translation is None:
#             return self.encoder(x)
#         else:
#             return self.translation(self.encoder(x)).vectors  # ["target"]

#     def get_value(self, x):
#         if self.translation is None:
#             return self.policy.get_value(self.encoder(x))
#         else:
#             return self.policy.get_value(self.translation(self.encoder(x)))

#     def get_action_and_value(self, x, action=None):
#         if self.translation is None:
#             return self.policy.get_action_and_value(self.encoder(x), action=action)
#         else:
#             hid = self.encoder(x)
#             # reshape (1, 12544) tensor into (4, 3136)
#             hid = hid.view(4, 3136)
#             hid = Space(vectors=hid, name="hid")
#             hid = self.translation(hid).vectors.view(
#                 1, 12544
#             )  # ['target'].view(1, 12544)
#             return self.policy.get_action_and_value(hid, action=action)

#     def get_action_and_value_deterministic(self, x, action=None):
#         if self.translation is None:
#             return self.policy.get_action_and_value_deterministic(
#                 self.encoder(x), action=action
#             )
#         else:
#             hid = self.encoder(x)
#             # reshape (1, 12544) tensor into (4, 3136)
#             hid = hid.view(4, 3136)
#             hid = Space(vectors=hid, name="hid")
#             hid = self.translation(hid).vectors.view(
#                 1, 12544
#             )  # ['target'].view(1, 12544)
#             return self.policy.get_action_and_value_deterministic(hid, action=action)

#     def forward(self, x):
#         return self.get_action_and_value(x, action=None)
