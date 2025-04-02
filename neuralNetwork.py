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
    def __init__(self, inputSize):
        super().__init__()
        # self.flatten = nn.Flatten()# Might not need to flatten
        self.linear_sigmoid_stack = nn.Sequential(
        #TODO Determine input features and output features: Train entires: 103904 Test entries:25976
        #First Layer
            nn.Linear(inputSize, 512), #22 input features, 512 output features is arbitrary, can be changed.
            nn.Sigmoid(),
        #Second Layer
            nn.Linear(512, 256), #512 input features, 256 output features
            nn.Sigmoid(),
        #Third Layer
            nn.Linear(256, 128), #256 input features, 128 output features
            nn.Sigmoid(),
        #Fourth Layer
            nn.Linear(128, 1), #128 input features, 1 output feature for binary classification.
            nn.Sigmoid() #Sigmoid function for binary classification
        )

    def forward(self, x):
        #x = self.flatten(x) #Maybe dont need this
        logits = self.linear_sigmoid_stack(x)
        return logits