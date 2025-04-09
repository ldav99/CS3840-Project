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
    This model has three fully connected layers with ReLU activation in between.
    """
    def __init__(self, size):
        """
        Initialize the neural network.
        
        Arguments:
        size (int): The number of input features (input size).
        """
        super().__init__()
        
        # Define the layers of the network using nn.Sequential for simplicity.
        self.linear_sigmoid_stack = nn.Sequential(
            # First Layer: Linear transformation (size input features -> 8 output features)
            nn.Linear(size, 8),   # 22 input features (size), 8 output features
            nn.ReLU(),            # Apply ReLU activation function

            # Second Layer (commented out): Would add another layer with 512 input and 256 output features
            # nn.Linear(512, 256),  # 512 input features, 256 output features
            # nn.ReLU(),            # Apply ReLU activation function

            # Third Layer: Linear transformation (8 input features -> 4 output features)
            nn.Linear(8, 4),      # 8 input features, 4 output features
            nn.ReLU(),            # Apply ReLU activation function

            # Fourth Layer: Output layer with 1 output feature (binary classification)
            nn.Linear(4, 1),      # 4 input features, 1 output feature (binary classification)
            # We do not need a Sigmoid here because the loss function will handle it (BCEWithLogitsLoss)
        )

    def forward(self, x):
        """
        Define the forward pass for the model.
        
        Arguments:
        x (Tensor): The input data tensor.
        
        Returns:
        Tensor: The output of the model, raw logits (no activation function here, for BCEWithLogitsLoss).
        """
        # Pass the input through the network layers.
        logits = self.linear_sigmoid_stack(x)  # Output raw logits (no activation needed here for BCE loss)
        
        return logits
