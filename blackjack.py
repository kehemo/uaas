import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
            nn.Linear(64, num_actions),
            nn.LogSoftmax(dim=-1)
        )

        if self.use_critic:
            self.critic = nn.Sequential(
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
