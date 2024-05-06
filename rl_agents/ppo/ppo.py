import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FeatureExtractor(nn.Module):
    def __init__(self, use_anchors=False, anchors=None, anchors_mean=None):#, anchors_std=None):
        super().__init__()

        self.use_anchors = use_anchors
        if self.use_anchors:
            self.register_buffer("anchors", anchors)
            self.register_buffer("anchors_mean", anchors_mean)
            # self.register_buffer("anchors_std", anchors_std)

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.Tanh(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.Tanh(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Flatten()
        )

    def forward(self, x):
        if self.use_anchors:
            with torch.no_grad():
                hidden = self.network(x)
                # TODO: hypeperam per cenetering and scaling
                hidden = (hidden - self.anchors_mean)#  / self.anchors_std
                hidden: torch.Tensor = F.normalize(hidden, p=2, dim=-1)
                hidden = hidden @ self.anchors.T  # relative representations
        else:
            hidden = self.network(x)
        return hidden
    
    def set_anchors(self, use_anchors, anchors, anchors_mean):
        self.use_anchors = use_anchors
        self.register_buffer("anchors", anchors)
        self.register_buffer("anchors_mean", anchors_mean)


class Policy(nn.Module):
    def __init__(self, num_actions) -> None:
        super().__init__()
        self.network_linear = nn.Sequential(
            nn.Tanh(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.Tanh()
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
    
    def get_value(self, x):
        x = self.network_linear(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.network_linear(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Agent(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, policy: Policy):
        super().__init__()

        self.encoder = feature_extractor
        self.policy = policy

    def get_value(self, x):
        return self.policy.get_value(self.encoder(x))
                                     
    def get_action_and_value(self, x, action=None):
        return self.policy.get_action_and_value(self.encoder(x), action=action)
    
    def forward(self, x):
        return self.get_action_and_value(x, action=None)