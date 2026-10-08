"""Official Beta actor and Algorithm 2 queue, adapted from image x0 to action A0."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
from sklearn.feature_selection import SelectKBest, f_regression


class ActionBetaActor(nn.Module):
    def __init__(self, action_dim: int, horizon: int, hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        if horizon < 8:
            raise ValueError("horizon must be at least 8 for the official three pooling layers")
        self.convs = nn.Sequential(
            nn.Conv1d(action_dim, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        layers = []
        in_dim = 128 * (horizon // 8)
        for _ in range(hidden_depth):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)))
            in_dim = hidden_dim
        final = nn.Linear(in_dim, 2)
        nn.init.constant_(final.weight, 0.0)
        nn.init.constant_(final.bias, 0.5)
        self.mlp = nn.Sequential(*layers, final)

    def forward(self, clean_actions: torch.Tensor):
        raw = self.mlp(self.convs(clean_actions.transpose(1, 2)).flatten(1))
        positive = torch.nn.functional.softplus(raw) + 1e-5
        return positive[:, 0], positive[:, 1]


@dataclass
class TimestepDraw:
    timesteps: torch.Tensor
    continuous: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


def sample_timesteps(actor: ActionBetaActor, clean_actions: torch.Tensor, total_timesteps: int):
    alpha, beta = actor(clean_actions)
    distribution = Beta(alpha, beta)
    continuous = distribution.sample()  # official implementation uses sample, not rsample
    discrete = torch.round(continuous * (total_timesteps - 1)).long()
    return TimestepDraw(discrete.detach(), continuous.detach(),
                        distribution.log_prob(continuous), distribution.entropy(), alpha, beta)


class DeltaQueue:
    """Rows are full-timestep before-minus-after KL vectors, as in the official replay buffer."""
    def __init__(self, total_timesteps: int, capacity: int = 5, n_select: int = 3):
        if not (1 <= n_select <= total_timesteps and capacity >= 2):
            raise ValueError("invalid queue dimensions")
        self.total_timesteps = total_timesteps
        self.capacity = capacity
        self.n_select = n_select
        self.features = np.zeros((capacity, total_timesteps), dtype=np.float64)
        self.targets = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.selected = list(range(n_select))

    def add(self, delta: torch.Tensor):
        row = delta.detach().double().cpu().numpy().reshape(-1)
        if row.shape != (self.total_timesteps,):
            raise ValueError("expected one KL delta per timestep")
        self.features[self.position] = row
        self.targets[self.position] = row.sum()  # official target is sum across all T
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def select(self):
        if self.size < 2:
            return self.selected
        x = self.features[:self.size]
        y = self.targets[:self.size]
        keep = np.any(x != 0, axis=1)
        x, y = x[keep], y[keep]
        if len(x) < 2:
            return self.selected
        with np.errstate(divide="ignore", invalid="ignore"):
            selector = SelectKBest(score_func=f_regression, k=self.n_select)
            selector.fit(x, y)
        indices = selector.get_support(indices=True).tolist()
        # Official source accidentally returns (indices,); return the actual 1-D list.
        if len(indices) != self.n_select or any(i < 0 or i >= self.total_timesteps for i in indices):
            raise RuntimeError("invalid selected timestep indices")
        self.selected = [int(i) for i in indices]
        return self.selected

    def state_dict(self):
        return dict(features=self.features, targets=self.targets, position=self.position,
                    size=self.size, selected=self.selected)

    def load_state_dict(self, state):
        self.features = state["features"]
        self.targets = state["targets"]
        self.position = int(state["position"])
        self.size = int(state["size"])
        self.selected = [int(i) for i in state["selected"]]


def reinforce_loss(log_probs: torch.Tensor, entropy: torch.Tensor,
                   kl_delta_by_t: torch.Tensor, ent_coef: float = 0.01):
    """Official code: reward = sum_S(delta).detach() + ent_coef*entropy; loss=-log_prob*reward."""
    reward = kl_delta_by_t.sum(dim=1).detach() + ent_coef * entropy
    return -(log_probs * reward).mean()
