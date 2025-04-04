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
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------
# Pre Process the Data
# ---------------------------------------
def callModel(dataset):
    # Expected columns for one-hot encoding:
    expected_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    missing_cols = [col for col in expected_columns if col not in dataset.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing and cannot be one-hot encoded: {missing_cols}")

    # Apply one-hot encoding to available expected columns
    dataset_ohe = pd.get_dummies(dataset, columns=[col for col in expected_columns if col in dataset.columns])

    print(dataset_ohe.info())  # Debugging information

    # Encode the target column if it exists in the original dataset
    label_encoder = LabelEncoder()
    if 'satisfaction' in dataset.columns:
        dataset['satisfaction'] = label_encoder.fit_transform(dataset['satisfaction'])
    else:
        print("Error: 'satisfaction' column not found.")

    # Normalization of numerical features
    numerical_features = [
        'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]

    # Handle missing values before scaling
    dataset_ohe.fillna(dataset_ohe.mean(), inplace=True)

    scaler = StandardScaler()
    dataset_ohe[numerical_features] = scaler.fit_transform(dataset_ohe[numerical_features])
    print(dataset_ohe.info())

    # Convert all columns to float
    dataset_ohe = dataset_ohe.astype(float)
    tensor = torch.tensor(dataset_ohe.values, dtype=torch.float32)

    # Extract input features and target labels (assuming the target is the last column)
    inputs = tensor[:, :-1]
    targets = tensor[:, -1]

    # Create a dataset and dataloader
    tensor_dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(tensor_dataset, batch_size=64, shuffle=True)

    # Determine effective input size (number of features)
    effective_input_size = inputs.shape[1]
    return dataloader, effective_input_size


def loadModel(device, inputSize):
    # Build your neural network with the correct input size
    model = neuralNetwork.NeuralNetwork(inputSize).to(device)
    return model


# ----------------------------------------
# Training Function
# ----------------------------------------
def trainModel(model, trainLoader, device, epochs=10, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    losses = []
    accuracies = []

    model.train()
    for epoch in range(epochs):
        total_Loss = 0.0
        correct_Predictions = 0
        total_Samples = 0

        for inputs, targets in trainLoader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            # Unsqueeze targets to match [batch_size, 1] shape of outputs
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_Loss += loss.item()

            # Convert logits to binary predictions
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_Predictions += (predictions.squeeze() == targets).sum().item()
            totalSamples += targets.size(0)

        epoch_Loss = total_Loss / len(trainLoader)
        epoch_Accuracy = correct_Predictions / totalSamples
        losses.append(epoch_Loss)
        accuracies.append(epoch_Accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_Loss:.4f}, Accuracy: {epoch_Accuracy:.4f}")


# ----------------------------------------
# Main Function
# ----------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Make sure these files exist in a 'data' folder:
    if not os.path.exists('data/train.csv'):
        print("Error: 'train.csv' file not found.")
        return
    else:
        df_train = pd.read_csv('data/train.csv')

    if not os.path.exists('data/test.csv'):
        print("Error: 'test.csv' file not found.")
        return
    else:
        df_test = pd.read_csv('data/test.csv')

    # Preprocess training data and get DataLoader + effective input size
    trainLoader, effective_input_size = callModel(df_train)

    # Build and train the model
    model = loadModel(device, effective_input_size)
    trainModel(model, trainLoader, device, epochs=10, learning_rate=0.001)

    # Simple demonstration plot of the first batch's feature distribution
    batch = next(iter(trainLoader))
    plt.boxplot(batch[0].numpy())
    plt.ylabel('Feature Values')
    plt.xlabel('Features')
    plt.title("Boxplot of One Batch of Features")
    plt.show()



if __name__ == "__main__":
    main()
