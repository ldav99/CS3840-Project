#Monica Brutto, Luke Davidson, Sam Webster
#CS-3840
#Dr. Wen Zhang 
#Final Project

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import neuralNetwork as modelNN
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------
# Pre Process the Data
#---------------------------------------
def preProcessing(dataset):
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
    x = dataset.drop(columns='satisfaction').astype(float)
    y = dataset['satisfaction'].astype(float)

    # Convert to PyTorch tensors
    tensor_inputs = torch.tensor(x.values, dtype=torch.float32)
    tensor_targets = torch.tensor(y.values, dtype=torch.float32)

    # Create TensorDataset and DataLoader
    tensor_dataset = TensorDataset(tensor_inputs, tensor_targets)
    dataloader = DataLoader(tensor_dataset, batch_size=12, shuffle=True)

    effective_input_size = tensor_inputs.shape[1]
    return dataloader, effective_input_size


def loadModel(device, size):
    # Build your neural network with the correct input size
    model_path = 'saved_models/saved_model.pth'
    model = modelNN.NeuralNetwork(size).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No saved model found, initializing new model.")

    return model


# ----------------------------------------
# Training Function
# ----------------------------------------
#https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def trainModel(model, dataloader, device, learning_rate):
    lossFunction = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    size = len(dataloader.dataset)

    model = model.to(device)
    losses = np.array([])
    accuracies = []

    model.train()

    correct_Predictions = 0
    total_Samples = 0
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device).float()

        outputs = model(x)
        # Unsqueeze targets to match [batch_size, 1] shape of outputs
        loss = lossFunction(outputs, y.unsqueeze(1))
        
        #Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if batch % 300 == 0:
            losses = np.append(losses, loss.item())
            loss, current = loss.item(), batch * 64 + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
        #Accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct_Predictions += (predictions.squeeze() == y).sum().item()
        total_Samples += y.size(0)

    epoch_Loss = loss.item()
    print(f"Epoch Loss: {epoch_Loss:.4f}")
    epoch_Accuracy = correct_Predictions / total_Samples
    print(f"Accuracy: {epoch_Accuracy:.4f}")
    return losses, epoch_Accuracy


def testModel(dataloader, model):
    lossFunction = nn.BCEWithLogitsLoss()
    model.eval()
    size = len(dataloader.dataset)
    batch = len(dataloader)
    testloss, correct = 0,0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            testloss = lossFunction(outputs, y.unsqueeze(1))
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            correct += (predictions == y).sum().item()
    testloss /= batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testloss:>8f} \n")
    return testloss, correct


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
    processedDataLoader, size = preProcessing(df_train)
    proceesedTestDataLoader, size = preProcessing(df_test)

    # Build and train the model
    model = loadModel(device, size)
    trainLosses = np.array([])
    trainAccuracies = np.array([])
    testLosses = np.array([])
    testAccuracies = np.array([])

    for epoch in range(10):
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        trainLoss, trainAccuracy = trainModel(model, processedDataLoader, device, learning_rate=0.001)
        trainLosses = np.append(trainLosses, trainLoss)
        trainAccuracies = np.append(trainAccuracies, trainAccuracy)
        print(f"Mean train Loss: {trainLosses.mean():.4f}")
        testLoss, testAccuracy = testModel(proceesedTestDataLoader, model)
        testLosses = np.append(testLosses, testLoss)
        testAccuracies = np.append(testAccuracies, testAccuracy)
          
    torch.save(model.state_dict(), 'saved_models/saved_model.pth')
    print("Model saved successfully.")


    # Simple demonstration plot of the first batch's feature distribution
    # batch = next(iter(processedDatset))
    # plt.boxplot(batch[0].numpy())
    # plt.ylabel('Feature Values')
    # plt.xlabel('Features')
    # plt.title("Boxplot of One Batch of Features")
    # plt.show()

    plt.plot(trainLosses, label='Train Loss')
    plt.plot(testLosses, label='Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('# of Epochs')
    plt.title("Loss over Time")
    plt.legend()
    plt.show()

    plt.plot(trainAccuracies, label='Train Accuracy')
    plt.plot(testAccuracies, label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('# of Epochs')
    plt.title("Accuracy over Time")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
