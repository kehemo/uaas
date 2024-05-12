from torch import nn
import torch
import gym
from torch.distributions.categorical import Categorical




class Policy(nn.Module):
    def __init__(self, num_actions, use_critic=True):
        super(Policy, self).__init__()
        self.use_critic = use_critic

        self.common = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.LogSoftmax(dim=-1)
        )

        if self.use_critic:
            self.critic = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, obs):
        x = self.common(obs)
        action_probs = self.actor(x)

        if self.use_critic:
            value = self.critic(x)
        else:
            value = torch.zeros((x.shape[0], 1), device=x.device)

        return Categorical(logits=action_probs), value


def make_env():
    return gym.make("CartPole-v1")


def preprocess_obss(obss, device=None):
    """
    Convert observation into Torch.Tensor

    Parameters
    ----
    obss: a tuple of three floats
    device: target device of torch.Tensor ('cpu', 'cuda')

    Return
    ----
    Torch Tensor
    """

    # player_sum, dealer_card, usable_ace = obss
    # return torch.tensor([player_sum, dealer_card, usable_ace, 1 if usable_ace else 0], device=device, dtype=torch.float).unsqueeze(0)
    return torch.tensor(obss, device=device, dtype=torch.float)
