import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from latentis.space import LatentSpace

# Relative stuff
from latentis.project import RelativeProjector
from latentis.project import relative
from latentis.transform import Centering, StandardScaling

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class FeatureExtractor(nn.Module):
    def __init__(
        self,
        use_relative=False,
        pretrained=False,
        anchors_alpha_min=0.01,
        anchors_alpha_max=0.999,
        device="cpu",
    ):
        super().__init__()
        self.use_relative = use_relative
        self.pretrained = pretrained
        self.anchors_alpha_min = anchors_alpha_min
        self.anchors_alpha_max = anchors_alpha_max
        self.obs_anchors = None
        self.device = device
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.Flatten()
        )


    def _compute_relative_representation(self, hidden):
        # assert self.obs_anchors is not None, "You must set the anchors first. Use set_anchors() method."
        return self.projector(x=hidden, anchors=self.anchors)  # .vectors

    def forward(self, x):
        num_stack = x.shape[1]
        x = x.view(
            x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        )  # (batch_size * stack, 3, 84, 84)
        if self.pretrained:
            # TODO: se uso pretrained ricorda model.eval()
            with torch.no_grad():
                hidden = self.network(x)
            if self.use_relative:
                hidden = self._compute_relative_representation(hidden)
        else:
            hidden = self.network(x)
            if self.use_relative:
                hidden = self._compute_relative_representation(hidden)

        hidden = hidden.view(-1, num_stack, hidden.shape[1])
        # flatten the last two dimensions
        hidden = hidden.view(
            hidden.shape[0], num_stack * hidden.shape[2]
        )  # (batch_size, num_stack * 3136) -> (batch_size, 12544)
        return hidden

    def forward_single(self, x):
        return self.network(x)


    @torch.no_grad()
    def set_obs_anchors(self, obs_anchors):
        self.obs_anchors = obs_anchors
        anchors = self.network(obs_anchors)
        self.anchors = anchors
        # self.register_buffer("anchors", anchors)

    @torch.no_grad()
    def compute_feature_variance(self, features):
        variance = torch.var(features, dim=0, unbiased=False).mean()
        print(f"Feature Variance: {variance.item()}")  # Logging the variance
        return variance

    @torch.no_grad()
    def adapt_anchors_alpha(self, feature_variance, var_min=0.001, var_max=0.1):
        """Scale anchors_alpha dynamically based on feature variance."""
        alpha = self.anchors_alpha_min + (feature_variance - var_min) * (
            (self.anchors_alpha_max - self.anchors_alpha_min) / (var_max - var_min)
        )
        alpha = torch.clamp(alpha, self.anchors_alpha_min, self.anchors_alpha_max)
        print(f"Dynamic Alpha: {alpha.item()}")  # Logging the dynamic alpha
        return alpha

    @torch.no_grad()
    def update_anchors(self):
        new_anchors = self.network(self.obs_anchors)
        feature_variance = self.compute_feature_variance(new_anchors)
        dynamic_alpha = self.adapt_anchors_alpha(feature_variance)
        self.anchors = dynamic_alpha * self.anchors + (1 - dynamic_alpha) * new_anchors


    def save_anchors_buffer(self):
        self.register_buffer("saved_anchors", self.anchors)

    # def load_anchors_buffer(self):
    #     self.anchors = self.saved_anchors
    @torch.no_grad()
    def set_anchors(self, anchors):
        self.anchors = anchors

class Policy(nn.Module):
    def __init__(self, num_actions, stack_n: int = 4) -> None:
        super().__init__()
        self.network_linear = nn.Sequential(
            nn.LayerNorm(stack_n * 64 * 7 * 7),
            nn.ReLU(),
            layer_init(nn.Linear(stack_n * 64 * 7 * 7, 512)),
            nn.ReLU(),
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

    def get_action_and_value_deterministic(self, x, action=None):
        x = self.network_linear(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            # action = probs.sample()
            # take maximum likelihood action, unwrap from [[a0], [a1], ...] to [a0, a1, ...].
            # If it's only a single element [a], do nothing.
            action = probs.probs.argmax(dim=1, keepdim=True)
            if len(action) == 1:
                action = action[0]
            else:
                action = action.squeeze()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Agent(nn.Module):
    def __init__(
        self, feature_extractor: FeatureExtractor, policy: Policy, translation=None
    ):
        super().__init__()

        self.encoder = feature_extractor
        self.policy = policy

        self.translation = translation

    def get_encoded_obs(self, x):
        if self.translation is None:
            return self.encoder(x)
        else:
            return self.translation(self.encoder(x)).vectors  # ["target"]

    def get_value(self, x):
        if self.translation is None:
            return self.policy.get_value(self.encoder(x))
        else:
            return self.policy.get_value(self.translation(self.encoder(x)))

    def get_action_and_value(self, x, action=None):
        if self.translation is None:
            return self.policy.get_action_and_value(self.encoder(x), action=action)
        else:
            hid = self.encoder(x)
            # reshape (1, 12544) tensor into (4, 3136)
            hid = hid.view(4, 3136)
            hid = LatentSpace(vectors=hid, name="hid")
            hid = self.translation(hid).vectors.view(
                1, 12544
            )  # ['target'].view(1, 12544)
            return self.policy.get_action_and_value(hid, action=action)

    def get_action_and_value_deterministic(self, x, action=None):
        if self.translation is None:
            return self.policy.get_action_and_value_deterministic(
                self.encoder(x), action=action
            )
        else:
            hid = self.encoder(x)
            # reshape (1, 12544) tensor into (4, 3136)
            hid = hid.view(4, 3136)
            hid = LatentSpace(vectors=hid, name="hid")
            hid = self.translation(hid).vectors.view(
                1, 12544
            )  # ['target'].view(1, 12544)
            return self.policy.get_action_and_value_deterministic(hid, action=action)

    def forward(self, x):
        return self.get_action_and_value(x, action=None)
