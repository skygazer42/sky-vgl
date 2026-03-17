from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, symmetric_propagate


def _mlp(in_channels, out_channels, hidden_channels, num_layers, dropout):
    layers = []
    current = in_channels
    for layer_index in range(num_layers - 1):
        layers.append(nn.Linear(current, hidden_channels))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current = hidden_channels
    layers.append(nn.Linear(current, out_channels))
    return nn.Sequential(*layers)


class TWIRLSConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        steps=4,
        alpha=0.5,
        lambda_=1.0,
        num_mlp_before=1,
        num_mlp_after=1,
        dropout=0.0,
    ):
        super().__init__()
        if steps < 1:
            raise ValueError("TWIRLSConv requires steps >= 1")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("TWIRLSConv requires alpha to be in [0, 1]")
        if lambda_ < 0.0:
            raise ValueError("TWIRLSConv requires lambda_ >= 0")
        if num_mlp_before < 1 or num_mlp_after < 1:
            raise ValueError("TWIRLSConv requires num_mlp_before and num_mlp_after >= 1")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("TWIRLSConv requires dropout to be in [0, 1]")

        hidden_channels = hidden_channels or out_channels
        self.out_channels = out_channels
        self.steps = steps
        self.alpha = alpha
        self.lambda_ = lambda_
        self.pre_mlp = _mlp(in_channels, hidden_channels, hidden_channels, num_mlp_before, dropout)
        self.post_mlp = _mlp(hidden_channels, out_channels, hidden_channels, num_mlp_after, dropout)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "TWIRLSConv")
        hidden = self.pre_mlp(x)
        initial = hidden
        current = hidden

        for _ in range(self.steps):
            propagated = symmetric_propagate(current, edge_index)
            current = (1.0 - self.alpha) * (self.lambda_ * propagated + (1.0 - self.lambda_) * current)
            current = current + self.alpha * initial

        return self.post_mlp(current)
