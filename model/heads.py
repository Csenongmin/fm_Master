
import torch
import torch.nn as nn

class DetHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,1))
    def forward(self, h):  # [B,T,d]
        return self.net(h).squeeze(-1)

class ClassHead(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
    def forward(self, h):
        return self.net(h)

class ActorHead(nn.Module):
    def __init__(self, d, n_actors):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_actors))
    def forward(self, h):
        return self.net(h)
