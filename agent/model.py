import torch
from torch import nn
from torch.distributions import Categorical
from agent.PPOConfig import PPOConfig

class ActorCritic(nn.Module):
    def __init__(self, input_channels:int, num_actions:int):
        super().__init__()
        
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self,x):
        x = self.ConvBlock(x)
        x = x.view(x.size(0), -1)
        action_probs = self.actor(x)
        values = self.critic(x)
        return action_probs, values

    def act(self, state):
        x = self.ConvBlock(state)
        x = x.view(x.size(0), -1) 
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logs = dist.log_prob(action)

        values = self.critic(x)
        return action.item(), action_logs.item(), values.item()