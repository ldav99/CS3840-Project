#Monica Brutto, Luke Davidson, Sam Webster
#CS-3840
#Dr. Wen Zhang 
#Final Project

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


    # Separate inputs and targets
    x = dataset.drop(columns='satisfaction').astype(float)
    y = dataset['satisfaction'].astype(float)

    # Convert to PyTorch tensors
    tensor_inputs = torch.tensor(x.values, dtype=torch.float32)
    tensor_targets = torch.tensor(y.values, dtype=torch.float32)

    # Create TensorDataset and DataLoader
    tensor_dataset = TensorDataset(tensor_inputs, tensor_targets)
    dataloader = DataLoader(tensor_dataset, batch_size=512, shuffle=True)

    effective_input_size = tensor_inputs.shape[1]
    return dataloader, effective_input_size


def loadModel(device, size, path):
    # Build your neural network with the correct input size
    model_path = path
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
def trainModel(model, dataloader, device, learning_rate):
    lossFunction = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    size = len(dataloader.dataset)

    model = model.to(device)
    losses = np.array([])
    
    all_preds = []
    all_labels = []

    model.train()

    correct_Predictions = 0
    total_Samples = 0

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device).float()

        outputs = model(x)
        
        # Store predictions and labels for later analysis
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        # Unsqueeze targets to match [batch_size, 1] shape of outputs
        loss = lossFunction(outputs, y.unsqueeze(1))
        
        #Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if batch % 10 == 0:
            losses = np.append(losses, loss.item())
            loss, current = loss.item(), batch * 64 + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
        #Accuracy
        correct_Predictions += (predictions.squeeze() == y).sum().item()
        total_Samples += y.size(0)


    epoch_Loss = loss.item()
    print(f"Epoch Loss: {epoch_Loss:.4f}")
    epoch_Accuracy = correct_Predictions / total_Samples
    print(f"Accuracy: {epoch_Accuracy:.4f}")

    return epoch_Loss, epoch_Accuracy, all_preds, all_labels


def testModel(dataloader, model):
    lossFunction = nn.BCEWithLogitsLoss()
    model.eval()
    size = len(dataloader.dataset)
    batch = len(dataloader)
    testloss = 0 
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:

            outputs = model(x)
            testloss += lossFunction(outputs, y.unsqueeze(1))
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            correct += (predictions == y).sum().item()
    
    testloss /= batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testloss:>8f} \n")


    return testloss, correct, all_preds, all_labels


# ----------------------------------------
# Main Function
# ----------------------------------------
def main():
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    processedTestDataLoader, size = preProcessing(df_test)

    #Initialize path for model
    modelPath = "saved_models/saved_model8-4lLD30.pth"

    # Build and train the model
    model = loadModel(device, size, modelPath)
    trainLosses = np.array([])
    trainAccuracies = np.array([])
    testLosses = np.array([])
    testAccuracies = np.array([])
    

    #Initialize Loss Decay
    learning_rate = 0.00005
    
    for epoch in range(150):
        #scheduler.step()
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        trainLoss, trainAccuracy,  trainPreds, trainLabels = trainModel(model, processedDataLoader, device, learning_rate)
        
        trainLosses = np.append(trainLosses, trainLoss)
        trainAccuracies = np.append(trainAccuracies, trainAccuracy)
        print(f"Mean train Loss: {trainLosses.mean():.4f}")
        
        testLoss, testAccuracy, testPreds, testLabels = testModel(processedTestDataLoader, model)
        
        testLosses = np.append(testLosses, testLoss)
        testAccuracies = np.append(testAccuracies, testAccuracy)
     

    torch.save(model.state_dict(), modelPath)
    print("Model saved successfully.")

    cm_train = confusion_matrix(trainLabels, trainPreds)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["Dissatisfied", "Satisfied"])
    disp_train.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Training Data)")
    plt.show()

    cm_train = confusion_matrix(testLabels, testPreds)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["Dissatisfied", "Satisfied"])
    disp_train.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Data)")
    plt.show()

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
