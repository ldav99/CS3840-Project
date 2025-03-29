#Monica Brutto, Luke Davidson, Sam Webster
#CS-3840
#Dr. Wen Zhang 
#Final Project

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()# Might not need to flatten
        self.linear_sigmoid_stack = nn.Sequential(
        #TODO Determine input features and output features: Train entires: 103904 Test entries:25976
        #First Layer
            nn.Linear(22, 512), #22 input features, 512 output features is arbitrary, can be changed.
        #Sigmoid function for binary classification
            nn.Sigmoid(),
#Maybe add more layers/output layer here. Needs more research.
        )

    def forward(self, x):
        x = self.flatten(x) #Maybe dont need this
        logits = self.linear_relu_stack(x)
        return logits