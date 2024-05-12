from torch import nn
import torch
import gym


class Policy(nn.Module):
    def __init__(self, num_actions):
        super(Policy, self).__init__()
        self.state_space = 4
        self.action_space = num_actions

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        # Episode policy and reward history
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = nn.Sequential(
            self.l1, nn.Dropout(p=0.6), nn.ReLU(), self.l2, nn.Softmax(dim=-1)
        )
        return nn.Categorical(model(x)), None


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
