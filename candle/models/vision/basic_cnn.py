import torch
import torch.nn as nn
from typing import Tuple


class BasicCNNClassifier(nn.Module):
    """
    A basic convolutional neural network (CNN) classifier with modular blocks.

    Attributes:
        input_shape (Tuple[int, int, int]): The shape of the input tensor (channels, height, width).
        block1 (nn.Sequential): The first convolutional block.
        block2 (nn.Sequential): The second convolutional block.
        block3 (nn.Sequential): The third convolutional block.
        fc1 (nn.Linear): Fully connected layer for intermediate features.
        bn_fc (nn.BatchNorm1d): Batch normalization for the fully connected layer.
        relu_fc (nn.ReLU): Activation for the fully connected layer.
        fc2 (nn.Linear): Fully connected output layer for classification.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_output_classes: int) -> None:
        """
        Initialize the BasicCNNClassifier.

        Args:
            input_shape (Tuple[int, int, int]): Shape of the input tensor (channels, height, width).
            num_output_classes (int): Number of output classes for classification.
        """
        super(BasicCNNClassifier, self).__init__()

        self.input_shape = input_shape

        # Define convolutional blocks
        self.block1 = self._conv_block(
            in_channels=input_shape[0],
            out_channels=16,
            activation=nn.ReLU(),
            pooling=nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = self._conv_block(
            in_channels=16,
            out_channels=32,
            activation=nn.LeakyReLU(),
            pooling=nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.block3 = self._conv_block(
            in_channels=32,
            out_channels=64,
            activation=nn.ELU(),
            pooling=nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate flattened size
        flattened_size = 64 * (input_shape[1] // 8) * (input_shape[2] // 8)

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, num_output_classes)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, activation: nn.Module, pooling: nn.Module
    ) -> nn.Sequential:
        """
        Creates a convolutional block with Conv2d, BatchNorm2d, Activation, and Pooling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (nn.Module): Activation function.
            pooling (nn.Module): Pooling layer.

        Returns:
            nn.Sequential: A sequential model combining Conv2d, BatchNorm2d, Activation, and Pooling.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation,
            pooling
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_output_classes).
        """
        # Apply convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.fc2(x)

        return x
