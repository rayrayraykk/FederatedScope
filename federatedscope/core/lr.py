import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_channels, class_num, use_bias=True):
        """
        Initialize the logistic regression model

        Args:
            self: write your description
            in_channels: write your description
            class_num: write your description
            use_bias: write your description
        """
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Linear(in_channels, class_num, bias=use_bias)

    def forward(self, x):
        """
        Forward Rosenblatt transformation.

        Args:
            self: write your description
            x: write your description
        """
        return self.fc(x)
