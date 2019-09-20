import torch.nn as nn
import torch.nn.functional as F


class BasicMosquitoNet(nn.Module):
    """A basic 1D conv net.
    We use 1D convolution, followed by max pool, 1D convolution, max pool, FC, FC.
    """

    def __init__(self, conv1_out=100, kernel_1=128, stride_1=3, padding_1=1,
                 conv2_out=100, kernel_2=128, stride_2=1):
        """
        conv1: (22050 - 128 - 2*1)/3  + 1 = 7309
        max_pool_1 = floor((Lin + −dilation×(kernel_size−1)−1)/stride_2) + 1
                   = floor(7309-2 /2) + 1 = 3653 + 1 = 3654
        conv2 = (3654 - 128)/1 + 1 = 3527
        max_pool_2 = floor(3527-2 /2) + 1 = 1763

        """
        super(BasicMosquitoNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_out,
                               kernel_size=kernel_1, stride=stride_1,
                               padding=padding_1)
        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out,
                               kernel_size=kernel_2, stride=stride_2)
        self.fc1 = nn.Linear(1763*conv2_out, 1)
        # self.fc1 = nn.Linear(918*conv2_out, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data.
        """
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)

        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        """
        # We use BCEWithLogitsLoss instead of applying sigmoid here
        # It is better computationally
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
