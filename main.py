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
    # Drop unnecessary columns
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(columns=['Unnamed: 0'], inplace=True)
    if 'id' in dataset.columns:
        dataset.drop(columns=['id'], inplace=True)
    
    label_encoder = LabelEncoder()
    if 'satisfaction' in dataset.columns:
        dataset['satisfaction'] = label_encoder.fit_transform(dataset['satisfaction'])
    else:
        print("Error: 'satisfaction' column not found.")
        return None, None
    
    expected_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    missing_cols = [col for col in expected_columns if col not in dataset.columns]
    if missing_cols:
        print(f"Warning: Missing categorical columns: {missing_cols}")
    dataset = pd.get_dummies(dataset, columns=[col for col in expected_columns if col in dataset.columns])

    # Convert bool columns to int (after one-hot encoding)
    for col in dataset.select_dtypes(include='bool').columns:
        dataset[col] = dataset[col].astype(int)

    # Define numerical features to scale
    numerical_features = [
        'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]

    # Handle missing values in numerical columns
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

    # Normalize numerical features
    scaler = StandardScaler()
    dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

    print(dataset.info())  # Final check before tensor conversion


    # Separate inputs and targets
    inputs = dataset.drop(columns='satisfaction').astype(float)
    targets = dataset['satisfaction'].astype(float)

    # Convert to PyTorch tensors
    tensor_inputs = torch.tensor(inputs.values, dtype=torch.float32)
    tensor_targets = torch.tensor(targets.values, dtype=torch.float32)

    # Create TensorDataset and DataLoader
    tensor_dataset = TensorDataset(tensor_inputs, tensor_targets)
    dataloader = DataLoader(tensor_dataset, batch_size=64, shuffle=True)

    effective_input_size = tensor_inputs.shape[1]
    return dataloader, effective_input_size


def loadModel(device, inputSize, model_path=None):
    model = neuralNetwork.NeuralNetwork(inputSize).to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    return model


# ----------------------------------------
# Training Function
# ----------------------------------------
#https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def trainModel(model, trainLoader, device, epochs=10, learning_rate=0.001):
    lossFunction = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    size = len(trainLoader.dataset)

    model.to(device)
    losses = np.array([])

    model.train()

    correct_Predictions = 0
    total_Samples = 0
    for batch, (inputs, targets) in enumerate(trainLoader):
        inputs = inputs.to(device)
        targets = targets.to(device).float()

        outputs = model(inputs)
        # Unsqueeze targets to match [batch_size, 1] shape of outputs
        loss = lossFunction(outputs, targets.unsqueeze(1))
        
        #Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if batch % 300 == 0:
            losses = np.append(losses, loss.item())
            loss, current = loss.item(), batch * 64 + len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
        #Accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct_Predictions += (predictions.squeeze() == targets).sum().item()
        total_Samples += targets.size(0)

    epoch_Loss = loss.item()
    print(f"Epoch Loss: {epoch_Loss:.4f}")
    epoch_Accuracy = correct_Predictions / total_Samples
    print(f"Accuracy: {epoch_Accuracy:.4f}")
    return losses, epoch_Accuracy


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
    model = loadModel(device, effective_input_size, model_path="saved_models/customer_satisfaction_model.pth")
    losses = np.array([])
    # results = np.array([])
    accuracies = []

    for epoch in range(10):
        print(f"Epoch {epoch+1}\n-------------------------------")
        losses, accuracy = trainModel(model, trainLoader, device)
        losses = np.append(losses, losses)
        accuracies.append(accuracy)
    print(losses)

    # Save the model
    model_path = "saved_models/customer_satisfaction_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Simple demonstration plot of the first batch's feature distribution
    batch = next(iter(trainLoader))
    plt.boxplot(batch[0].numpy())
    plt.ylabel('Feature Values')
    plt.xlabel('Features')
    plt.title("Boxplot of One Batch of Features")
    plt.show()

    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('# of Epochs')
    plt.title("Loss over time")
    plt.show()

    plt.plot(accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('# of Epochs')
    plt.title("Accuracy over time")
    plt.show()



if __name__ == "__main__":
    main()