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
# ---------------------------------------
def preProcessing(dataset):
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
    # numerical_features = [
    #     'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
    #     'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    #     'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
    #     'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    # ]

    # Handle missing values before scaling
    dataset_ohe.fillna(dataset_ohe.mean(), inplace=True)

    scaler = StandardScaler()
    # dataset_ohe[numerical_features] = scaler.fit_transform(dataset_ohe[numerical_features])
    print(dataset_ohe.info())

    # Convert all columns to float
    dataset_ohe = dataset_ohe.astype(float)
    tensor = torch.tensor(dataset_ohe.values, dtype=torch.float32)

    train_dataloader, = DataLoader(tensor, batch_size=64)
    sizeOfData = tensor.shape[1]
    return train_dataloader, sizeOfData


def loadModel(device):
    # Build your neural network with the correct input size
    model = modelNN.NeuralNetwork().to(device)
    return model


# ----------------------------------------
# Training Function
# ----------------------------------------
#https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def trainModel(model, dataloader, device, learning_rate=0.001):
    lossFunction = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    size = len(dataloader.dataset)

    model.to(device)
    losses = np.array([])

    model.train()

    correct_Predictions = 0
    total_Samples = 0
    for batch, data in enumerate(dataloader):
        inputs, labels = data
        outputs = model(inputs)
        # Unsqueeze targets to match [batch_size, 1] shape of outputs
        loss = lossFunction(outputs, labels.unsqueeze(1))
        
        #Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 300 == 0:
            losses = np.append(losses, loss.item())
            loss, current = loss.item(), batch * 64 + len(y)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
        #Accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct_Predictions += (predictions.squeeze() == y).sum().item()
        total_Samples += y.size(0)

        if batch % 300 == 0:
            epoch_Accuracy = correct_Predictions / total_Samples
            print(f"Accuracy: {epoch_Accuracy:.4f}")
    return losses

def testModel(dataloader, model):
    lossFunction = nn.BCEWithLogitsLoss()
    model.eval()
    size = len(dataloader.dataset)
    batch = len(dataloader)
    testloss, correct = 0,0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            testloss += lossFunction(outputs, y).item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
    testloss /= batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testloss:>8f} \n")


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

    tensor = tensor.to(device)

    # Build and train the model
    model = loadModel(device, size)
    losses = np.array([])
    results = np.array([])
    accuracies = []

    for epoch in range(10):
        print(f"Epoch {epoch+1}\n-------------------------------")
        trainModel(model, processedDataLoader, device, learning_rate=0.001)
        testModel(processedDataLoader, model)
    print(losses)

    # Simple demonstration plot of the first batch's feature distribution
    # batch = next(iter(processedDatset))
    # plt.boxplot(batch[0].numpy())
    # plt.ylabel('Feature Values')
    # plt.xlabel('Features')
    # plt.title("Boxplot of One Batch of Features")
    # plt.show()

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
