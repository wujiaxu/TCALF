import torch
import torch.nn as nn


from rl.networks.distributions import Bernoulli, Categorical, DiagGaussian

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class RecurrentActorCriticPolicy(nn.Module):
    """ Class for a robot policy network """
    def __init__(self, base, output_dim, base_kwargs=None):
        super(RecurrentActorCriticPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.base = base

        self.dist = DiagGaussian(self.base.output_size, output_dim)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    def eval_mode(self):
        self.dist.eval()
        self.base.eval_mode()
        return
    
    def train_mode(self):
        self.dist.train()
        self.base.train_mode()

    def forward(self, inputs, rnn_hxs,episode_masks,metas=None):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, episode_masks,deterministic=False):
        
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, episode_masks,infer=True)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs,episode_masks):

        value, _, _ = self.base(inputs, rnn_hxs, episode_masks,infer=True)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, action,episode_masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs,episode_masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



