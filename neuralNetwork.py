# Monica Brutto, Luke Davidson, Sam Webster
# CS-3840
# Dr. Wen Zhang 
# Final Project

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    """
    Neural Network model for binary classification.
    This model has three fully connected layers with ReLU activation and Dropout in between.
    """
    def __init__(self, size):
        """
        Initialize the neural network.
        
        Arguments:
        size (int): The number of input features (input size).
        """
        super().__init__()
        
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(size, 8),    # Input layer
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.1),     # Dropout after first activation

            nn.Linear(8, 4),       # Hidden layer
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Linear(4, 1)        # Output layer (binary classification)

        )

    def forward(self, x):
        logits = self.linear_sigmoid_stack(x)
        return logits
