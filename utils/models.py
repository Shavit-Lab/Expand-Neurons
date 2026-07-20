import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """Two-layer MLP with a single hidden ReLU, no biases.

    Seed model for :class:`NeuralNetExpandThenContract`.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.expansion_factor = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralNetEmbedding(nn.Module):
    """Two-layer MLP prepended with a square input embedding, no biases.

    The embedding is a learnable ``input_size x input_size`` linear map applied
    before the hidden layer. Seed model for
    :class:`NeuralNetEmbeddingExpandThenContract`; the embedding is preserved
    unchanged by that expansion.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.expansion_factor = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embedding = nn.Linear(input_size, input_size, bias=False)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        out = self.embedding(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class CNNMLP(nn.Module):
    """CNN backbone (three 3x3 conv blocks + 1x1 projection + global avg pool)
    followed by a two-layer MLP classifier head.

    Each conv block is three (Conv3x3, BN, ReLU) layers followed by 2x2
    max-pool. A final 1x1 conv projects to ``flatten_size`` channels before
    adaptive average pooling to 1x1. Seed model for
    :class:`CNNMLPExpandThenContract`; only the fc1/fc2 head is expanded.
    """

    def __init__(self, flatten_size: int = 256, hidden_size: int = 64, num_classes: int = 10):
        super().__init__()
        self.expansion_factor = 1
        self.flatten_size = flatten_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.conv_layers = nn.Sequential(
            conv_block(3, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            nn.MaxPool2d(2),
            conv_block(128, 256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, flatten_size, 1, bias=False),
            nn.BatchNorm2d(flatten_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc1 = nn.Linear(flatten_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def set_conv_trainable(self, trainable: bool) -> None:
        for p in self.conv_layers.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
