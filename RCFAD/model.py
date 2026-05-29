"""Model definitions for RC-FAD."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from RCFAD.constants import TABULAR_INPUT_DIM


class Net(nn.Module):
    """Small binary anomaly detector for image and tabular datasets.

    For MNIST/Fashion-MNIST, grayscale images are converted to 3-channel 32x32
    tensors, so the same model works for CIFAR-10 and MNIST-like datasets.
    Public tabular anomaly datasets use a padded/truncated vector branch, while
    MLP branch. Both branches expose 84-dimensional features for contrastive
    baselines and one binary anomaly logit.
    """

    def __init__(self):
        super().__init__()
        self.tab_fc1 = nn.Linear(TABULAR_INPUT_DIM, 64)
        self.tab_fc2 = nn.Linear(64, 84)
        self.tab_fc3 = nn.Linear(84, 1)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # Binary anomaly score logit. Use sigmoid(logit) as anomaly probability.
        self.fc3 = nn.Linear(84, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = F.relu(self.tab_fc1(x))
            x = F.relu(self.tab_fc2(x))
            return x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if x.dim() == 2:
            return self.tab_fc3(features).view(-1)
        return self.fc3(features).view(-1)

