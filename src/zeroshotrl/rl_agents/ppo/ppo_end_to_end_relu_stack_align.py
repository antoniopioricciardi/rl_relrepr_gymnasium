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
        # obs_anchors_filename=None,
        # obs_anchors=None,
        anchors_alpha=0.99,
        anchors_alpha_min=0.01,
        anchors_alpha_max=0.999,
        device="cpu",
    ):  # , anchors_std=None):
        super().__init__()
        self.use_relative = use_relative
        self.pretrained = pretrained
        self.anchors_alpha = anchors_alpha
        self.anchors_alpha_min = anchors_alpha_min
        self.anchors_alpha_max = anchors_alpha_max
        self.obs_anchors = None
        
        self.var_min = 0.001
        self.var_max = 1
        self.dynamic_alpha = 0
        self.feature_variance = 0
        # self.obs_anchors_filename = None
        # self.anchors = None # to be computed
        # self.anchors_mean = None # to be computer
        # self.observation_anchors = obs_anchors
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Flatten(),
        )

        if self.use_relative:
            # obs_anchors_filename is used to recover the obs_anchors when loading the model
            # self.obs_anchors_filename = obs_anchors_filename
            # self.register_buffer("obs_anchors_filename", obs_anchors_filename)
            # self.register_buffer("obs_anchors", obs_anchors)
            # self.obs_anchors = obs_anchors
            # anchors = None
            self.projector = RelativeProjector(
                projection_fn=relative.cosine_proj,
                abs_transforms=[Centering(), StandardScaling()],
            )
            # self.set_anchors()

    # @torch.no_grad()
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

    # @torch.no_grad()
    # def update_anchors(self):
    #     """ TO BE CALLED DURING TRAINING """
    #     # use simple moving average to update the anchors
    #     if self.anchors_alpha == -1:
    #         new_anchors = self.network(self.obs_anchors)
    #         self.feature_variance = self.compute_feature_variance(new_anchors)
    #         self.dynamic_alpha = self.adapt_anchors_alpha(self.feature_variance)
    #         self.anchors = self.dynamic_alpha * self.anchors + (1 - self.dynamic_alpha) * new_anchors
    #     else:
    #         new_anchors = self.network(self.obs_anchors)
    #         self.anchors = (
    #             self.anchors_alpha * self.anchors + (1 - self.anchors_alpha) * new_anchors
    #         )  # keep % of the old anchors # 0.99 and 0.999


    @torch.no_grad()
    def update_anchors(self, step=None, total_steps=None):
        """
        Update the anchors during training with exponential growth if anchors_alpha == -2.

        Arguments:
        - step: Current training step (required for exponential growth).
        - decay_rate: Controls the rate of growth of alpha.
        """
        new_anchors = self.network(self.obs_anchors)

        if self.anchors_alpha == -2:
            # Ensure we have the necessary step information.
            assert step is not None and total_steps is not None, (
                "Both 'step' and 'total_steps' must be provided for linear EMA scheduling."
            )

            # Define the maximum EMA decay factor (i.e. final momentum).
            max_alpha = 0.999  # You can adjust this final value as needed.

            # Compute progress through the linear schedule.
            # For the first 80% of training, we increase alpha linearly; then we keep it fixed.
            progress = min(step / (0.5 * total_steps), 1.0)
            current_alpha = progress * max_alpha
            self.dynamic_alpha = current_alpha

            # EMA update: anchors = current_alpha * anchors + (1 - current_alpha) * new_anchors
            self.anchors = current_alpha * self.anchors + (1 - current_alpha) * new_anchors


        elif self.anchors_alpha == -1:
            # Dynamic alpha based on feature variance
            self.feature_variance = self.compute_feature_variance(new_anchors)
            self.dynamic_alpha = self.adapt_anchors_alpha(self.feature_variance)
            self.anchors = (
                self.dynamic_alpha * self.anchors + (1 - self.dynamic_alpha) * new_anchors
            )
        else:
            # Fixed alpha
            self.anchors = (
                self.anchors_alpha * self.anchors + (1 - self.anchors_alpha) * new_anchors
            )


    def save_anchors_buffer(self):
        self.register_buffer("saved_anchors", self.anchors)

    # def load_anchors_buffer(self):
    #     self.anchors = self.saved_anchors
    @torch.no_grad()
    def set_anchors(self, anchors):
        self.anchors = anchors


    """ DYNAMIC ANCHOR UPDATING (variance based) """
    @torch.no_grad()
    def compute_feature_variance(self, features):
        variance = torch.var(features, dim=0, unbiased=False).mean()
        # print(f"Feature Variance: {variance.item()}")  # Logging the variance
        return variance

    @torch.no_grad()
    def adapt_anchors_alpha(self, feature_variance): # , var_min=0.001, var_max=0.1):
        """Scale anchors_alpha dynamically based on feature variance."""
        alpha = self.anchors_alpha_min + (feature_variance - self.var_min) * (
            (self.anchors_alpha_max - self.anchors_alpha_min) / (self.var_max - self.var_min)
        )
        alpha = torch.clamp(alpha, self.anchors_alpha_min, self.anchors_alpha_max)
        # print(f"Dynamic Alpha: {alpha.item()}")  # Logging the dynamic alpha
        return alpha


        

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
        self, feature_extractor: FeatureExtractor, policy: Policy, translation=None, num_envs=1, num_stack=4,
    ):
        super().__init__()

        self.encoder = feature_extractor
        self.policy = policy
        self.num_envs = num_envs
        self.num_stack = num_stack

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
            hid = self.encoder(x)
            # hid = hid.view(4, 3136)
            # reshape hid from (1, whatever) to (num_envs, num_stack, 3136)
            hid = hid.view(self.num_envs, self.num_stack, 3136)
            
            hid = LatentSpace(vectors=hid, name="hid")
            hid = self.translation(hid).vectors.view(
                self.num_envs, self.num_stack*3136
            )  # ['target'].view(1, 12544)
            return self.policy.get_value(hid)
            # return self.policy.get_value(self.translation(self.encoder(x)))

    def get_action_and_value(self, x, action=None):
        # if num_envs is None:
        #     num_envs = self.num_envs
        # if num_stack is None:
        #     num_stack = self.num_stack
        if self.translation is None:
            return self.policy.get_action_and_value(
                self.encoder(x), action=action
            )
        else:
            num_envs = x.shape[0]
            num_stack = x.shape[1]
            hid = self.encoder(x)
            # hid = hid.view(4, 3136)
            # reshape hid from (1, whatever) to (num_envs, num_stack, 3136)
            hid = hid.view(num_envs, num_stack, 3136)
            
            hid = LatentSpace(vectors=hid, name="hid")
            hid = self.translation(hid).vectors.view(
                num_envs, num_stack*3136
            )  # ['target'].view(1, 12544)
            return self.policy.get_action_and_value(hid, action=action)


    def get_action_and_value_deterministic(self, x, action=None):#, num_envs=None, num_stack=None):
        # if num_envs is None:
        #     num_envs = self.num_envs
        # if num_stack is None:
        if self.translation is None:
            return self.policy.get_action_and_value_deterministic(
                self.encoder(x), action=action
            )
        else:
            num_envs = x.shape[0]
            num_stack = x.shape[1]
            hid = self.encoder(x)
            # hid = hid.view(4, 3136)
            # reshape hid from (1, whatever) to (num_envs, num_stack, 3136)
            hid = hid.view(num_envs, num_stack, 3136)
            
            hid = LatentSpace(vectors=hid, name="hid")
            hid = self.translation(hid).vectors.view(
                num_envs, num_stack*3136
            )  # ['target'].view(1, 12544)
            return self.policy.get_action_and_value_deterministic(hid, action=action)

    def forward(self, x):
        return self.get_action_and_value(x, action=None)
