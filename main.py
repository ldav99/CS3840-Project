#Monica Brutto, Luke Davidson, Sam Webster
#CS-3840
#Dr. Wen Zhang 
#Final Project

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import neuralNetwork
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

#---------------------------------------
#Pre Process the Data
#---------------------------------------
def callModel(dataset, device):
#Pre Process the Data
#---------------------------------------
    expected_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    missing_cols = [col for col in expected_columns if col not in dataset.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing and cannot be one-hot encoded: {missing_cols}")
    dataset_ohe = pd.get_dummies(dataset, columns=[col for col in expected_columns if col in dataset.columns])

    print(dataset_ohe.info())  # Debugging information

    label_encoder = LabelEncoder()
    if 'satisfaction' in dataset.columns:
        dataset['satisfaction'] = label_encoder.fit_transform(dataset['satisfaction'])
    else:
        print("Error: 'satisfaction' column not found.")

    # Normalization of numerical features
    # ensures features have a similar scale, reducing bias from large-valued features
    numerical_features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                          'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                          'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
                          'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    
    # Handle missing values before scaling
    dataset_ohe.fillna(dataset_ohe.mean(), inplace=True)

    scaler = StandardScaler()
    dataset_ohe[numerical_features] = scaler.fit_transform(dataset_ohe[numerical_features])
    print(dataset_ohe.info())  

    dataset_ohe = dataset_ohe.astype(float)  # Convert all columns to float
    tensor = torch.tensor(dataset_ohe.values, dtype=torch.float32)

    # Extract input features and target labels
    inputs = tensor[:, :-1]  # All columns except the last (target)
    targets = tensor[:, -1]  # The last column (satisfaction)

    # Create a dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # This returns a DataLoader
    return dataloader

def loadModel(device, inputSize):
    model = neuralNetwork.NeuralNetwork(inputSize).to(device)  # Pass input size to the constructor
    return model

# ----------------------------------------
# Training Function
# ----------------------------------------
def trainModel(model, trainLoader, device, epochs=10, learning_rate=0.001): 
    # create a loop for loss function and optimizer 
    criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy with logits loss for binary classification 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    losses = [] # to store the loss values for each epoch
    accuracies = [] # to store the accuracy values for each epoch


    model.train()
    
    totalLoss = 0.0
    correctPredictions = 0
    totalSamples = 0

    for inputs, targets in trainLoader:
        inputs = inputs.to(device) # Initialize inputs
        targets = targets.float()

        optimizer.zero_grad() # zero the gradients before each epoch
        # Forward pass
        outputs = model(inputs) # Pass inputs through the model
        # Compute the loss
        loss = criterion(outputs, targets)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        totalLoss += loss.item()
        # convert logits to binary predictions
        predictions = (torch.sigmoid(outputs) > 0.5).float() # apply sigmoid to get probabilities
        correctPredictions += (predictions == targets).sum().item() # count correct predictions
        totalSamples += targets.size(0)
    
    epochLoss = totalLoss / len(trainLoader)
    epochAccuracy = correctPredictions / totalSamples
    losses.append(epochLoss)
    accuracies.append(epochAccuracy)
   


def main():
#Get device to use to run the model on
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    #Load Data
    #---------------------------------------   
    # Check if the file exists
    if not os.path.exists('data/train.csv'):
        print("Error: 'data/train.csv' file not found.")
        return
    else: 
        df_train = pd.read_csv('data/train.csv')

    if not os.path.exists('data/test.csv'):
        print("Error: 'data/test.csv' file not found.")
        return
    else: 
        df_test = pd.read_csv('data/test.csv')
    
    # print(df_train.head())
    # print(df_train.info())

#Call model on train and test
    trainLoader = callModel(df_train, device)
    # testLoader = callModel(df_test, device)
    inputSize = df_train.shape[1]# Number of features in the dataset
    
    model = loadModel(device, inputSize) # Load the model with the input size and device
    trainModel(model, trainLoader, device, epochs=10, learning_rate=0.001) 


#Results 
#---------------------------------------
    #Plot Accuracy
    plt.boxplot(trainLoader)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Accuracy")
    plt.show()
    
    # plt.boxplot(testResult)
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.title("Accuracy")
    # plt.show()

    #Plot Precision and Recall
    #Google says these are one graph
    #plt.plot(data)
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.title("Precision-Recall")
    #plt.show()



if __name__=="__main__":
    main()