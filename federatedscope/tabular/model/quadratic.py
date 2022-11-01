import torch


class QuadraticModel(torch.nn.Module):
    def __init__(self, in_channels, class_num):
        """
        Initialize the model

        Args:
            self: write your description
            in_channels: write your description
            class_num: write your description
        """
        super(QuadraticModel, self).__init__()
        x = torch.ones((in_channels, 1))
        self.x = torch.nn.parameter.Parameter(x.uniform_(-10.0, 10.0).float())

    def forward(self, A):
        """
        Compute the forward projection of A

        Args:
            self: write your description
            A: write your description
        """
        return torch.sum(self.x * torch.matmul(A, self.x), -1)
