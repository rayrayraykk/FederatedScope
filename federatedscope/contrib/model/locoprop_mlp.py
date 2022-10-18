import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList

from federatedscope.register import register_model

HIDDEN = [1000, 500, 250, 30, 250, 500, 1000]


class LocoPropMLP(torch.nn.Module):
    def __init__(self, channel_list, dropout=0.):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.linears = ModuleList()
        for in_channel, out_channel in zip(channel_list[:-1],
                                           channel_list[1:]):
            self.linears.append(Linear(in_channel, out_channel))

    def forward(self, x, activations=None, post_activations=None):
        for layer in self.linears[:-1]:
            x = layer(x)
            if activations is not None:
                activations.append(x)
            x = F.relu(x)
            if post_activations is not None:
                post_activations.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        if activations is not None:
            activations.append(x)
        return F.log_softmax(x, dim=-1)


def call_locoprop_mlp(model_config, input_shape):
    if model_config.type == "locoprop_mlp":
        # Input shape of MNIST is torch.Size([64, 1, 28, 28])
        # So we ignore `input_shape` and use hard-code implement
        channel_list = [28 * 28] + HIDDEN + [model_config.out_channels]
        model = LocoPropMLP(channel_list=channel_list,
                            dropout=model_config.dropout)
        return model


register_model('locoprop_mlp', call_locoprop_mlp)
