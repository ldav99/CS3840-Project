# Monica Brutto, Luke Davidson, Sam Webster
# CS-3840
# Dr. Wen Zhang 
# Final Project

from torch import nn

class NeuralNetwork(nn.Module):
    # Neural Network model for binary classification.
    # This model has three fully connected layers with ReLU activation in between.
    def __init__(self, size):
        super().__init__()
        
        # Define the layers of the network using nn.Sequential for simplicity.
        self.linear_sigmoid_stack = nn.Sequential(
            # First Layer: Linear transformation (size input features -> 8 output features)
            nn.Linear(size, 8),   # 27 input features (size), 8 output features
            nn.ReLU(),            # Apply ReLU activation function

            # Second Layer: Linear transformation (8 input features -> 4 output features)
            nn.Linear(8, 4),      # 8 input features, 4 output features
            nn.ReLU(),            # Apply ReLU activation function

            # Output Layer: Output layer with 1 output feature (binary classification)
            nn.Linear(4, 1),      # 4 input features, 1 output feature (binary classification)
            # We do not need a Sigmoid here because the loss function will handle it (BCEWithLogitsLoss)
        )

    def forward(self, x):
        #Define the forward pass for the model.
        logits = self.linear_sigmoid_stack(x)  # Output raw logits (no activation needed here for BCE loss)
        
        return logits
