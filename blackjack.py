import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym

# @dataclass
# class Config:
#     """
#     Stores algorithmic hyperparameters.
#     """

#     score_threshold = 0.93
#     discount = 0.995
#     lr = 5e-7
#     max_grad_norm = 0.5
#     log_interval = 10
#     max_episodes = 25000
#     gae_lambda = 0.95
#     use_critic = False
#     clip_ratio = 0.2
#     target_kl = 0.01
#     train_ac_iters = 5
#     use_discounted_reward = False
#     entropy_coef = 0.01
#     use_gae = False

def make_balckjack_env():
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    return env

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

class BlackjackACModel(nn.Module):
    def __init__(self, num_actions, use_critic=True):
        super(BlackjackACModel, self).__init__()
        self.use_critic = use_critic

        self.common = nn.Sequential(
            nn.Linear(3, 64),
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