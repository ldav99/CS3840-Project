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
# Preprocess the Data
# ---------------------------------------
def preProcessing(dataset):
    """
    Preprocesses the dataset by:
    1. Dropping unnecessary columns like 'Unnamed: 0' and 'id'.
    2. Label encoding the target variable 'satisfaction' to convert categorical values into numerical ones.
    3. One-hot encoding categorical features such as 'Gender', 'Customer Type', etc.
    4. Handling missing values by filling them with the mean of the column.
    5. Normalizing numerical features using StandardScaler.
    6. Returning a DataLoader for model training and the effective input size.

    Arguments:
    dataset (pd.DataFrame): The dataset to preprocess.

    Returns:
    dataloader (DataLoader): A DataLoader that can be used for model training.
    effective_input_size (int): The number of features after preprocessing.
    """
    # Drop unnecessary columns that are not useful for the model
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(columns=['Unnamed: 0'], inplace=True)
    if 'id' in dataset.columns:
        dataset.drop(columns=['id'], inplace=True)
    
    # Label encode the 'satisfaction' column (target variable) from categorical values to numeric
    label_encoder = LabelEncoder()
    if 'satisfaction' in dataset.columns:
        dataset['satisfaction'] = label_encoder.fit_transform(dataset['satisfaction'])
    else:
        print("Error: 'satisfaction' column not found.")
        return None, None
    
    # List of expected categorical columns to be one-hot encoded
    expected_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    missing_cols = [col for col in expected_columns if col not in dataset.columns]
    if missing_cols:
        print(f"Warning: Missing categorical columns: {missing_cols}")
    
    # One-hot encode the categorical columns that are present in the dataset
    dataset = pd.get_dummies(dataset, columns=[col for col in expected_columns if col in dataset.columns])

    # Convert any boolean columns to integers (0 or 1) after one-hot encoding
    for col in dataset.select_dtypes(include='bool').columns:
        dataset[col] = dataset[col].astype(int)

    # Define numerical features to normalize, as these are the continuous variables
    numerical_features = [
        'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]

    # Handle missing values in numerical columns by replacing them with the mean of the column
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

    # Normalize the numerical features using StandardScaler to standardize them (zero mean, unit variance)
    scaler = StandardScaler()
    dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

    # Print final dataset information after preprocessing to confirm changes
    print(dataset.info())

    # Separate the target variable ('satisfaction') from the input features
    x = dataset.drop(columns='satisfaction').astype(float)
    y = dataset['satisfaction'].astype(float)

    # Convert the input features and target variable to PyTorch tensors
    tensor_inputs = torch.tensor(x.values, dtype=torch.float32)
    tensor_targets = torch.tensor(y.values, dtype=torch.float32)

    # Create a TensorDataset to hold the inputs and targets together for easy batching
    tensor_dataset = TensorDataset(tensor_inputs, tensor_targets)
    dataloader = DataLoader(tensor_dataset, batch_size=12, shuffle=True)  # Set batch size and shuffle the data

    # Get the number of features (input size) for the neural network
    effective_input_size = tensor_inputs.shape[1]
    return dataloader, effective_input_size


# ---------------------------------------
# Load the Saved Model
# ---------------------------------------
def loadModel(device, size, path):
    """
    Loads a pre-trained model from the specified path or initializes a new model if none is found.

    Arguments:
    device (str): The device to load the model on, e.g., 'cpu' or 'cuda'.
    size (int): The size of the input features (number of input neurons).
    path (str): The path where the model is saved.

    Returns:
    model (NeuralNetwork): The loaded or initialized model.
    """
    # Build the model with the given input size
    model = modelNN.NeuralNetwork(size).to(device)
    
    # If a saved model exists, load its weights; otherwise, initialize a new model
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded model from {path}")
    else:
        print("No saved model found, initializing new model.")

    return model


# ----------------------------------------
# Training Function
# ----------------------------------------
def trainModel(model, dataloader, device, learning_rate):
    """
    Trains the model using the provided data.

    Arguments:
    model (NeuralNetwork): The model to train.
    dataloader (DataLoader): The data loader that provides batches of data.
    device (str): The device to perform training on, e.g., 'cpu' or 'cuda'.
    learning_rate (float): The learning rate for the optimizer.

    Returns:
    epoch_Loss (float): The loss for the current epoch.
    epoch_Accuracy (float): The accuracy for the current epoch.
    """
    # Define the loss function and optimizer
    lossFunction = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    size = len(dataloader.dataset)  # Total number of samples in the dataset
    model = model.to(device)  # Move the model to the specified device (e.g., CPU or GPU)
    losses = np.array([])  # Array to track loss values
    accuracies = []  # List to track accuracy values

    model.train()  # Set the model to training mode

    correct_Predictions = 0  # To track the number of correct predictions
    total_Samples = 0  # To track the total number of samples processed

    for batch, (x, y) in enumerate(dataloader):
        # Move the data to the appropriate device
        x = x.to(device)
        y = y.to(device).float()

        # Forward pass: Get model predictions
        outputs = model(x)
        
        # Compute the loss for the current batch
        loss = lossFunction(outputs, y.unsqueeze(1))  # Unsqueeze to match output shape
        
        # Backpropagate the loss and update the model weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print loss statistics every 300 batches
        if batch % 300 == 0:
            losses = np.append(losses, loss.item())
            print(f"loss: {loss.item():>7f}  [{batch * 64 + len(x):>5d}/{size:>5d}]")
    
        # Calculate accuracy for the batch
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid and threshold at 0.5
        correct_Predictions += (predictions.squeeze() == y).sum().item()  # Count correct predictions
        total_Samples += y.size(0)  # Update total samples

    # Compute the final loss and accuracy for the epoch
    epoch_Loss = loss.item()
    epoch_Accuracy = correct_Predictions / total_Samples
    print(f"Epoch Loss: {epoch_Loss:.4f}")
    print(f"Accuracy: {epoch_Accuracy:.4f}")
    
    return epoch_Loss, epoch_Accuracy


# ----------------------------------------
# Test the Model
# ----------------------------------------
def testModel(dataloader, model):
    """
    Tests the model on the given data and computes test accuracy and loss.

    Arguments:
    dataloader (DataLoader): The data loader that provides batches of test data.
    model (NeuralNetwork): The trained model to test.

    Returns:
    test_loss (float): The average test loss.
    test_accuracy (float): The accuracy on the test data.
    """
    lossFunction = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
    model.eval()  # Set the model to evaluation mode (disables dropout, batchnorm)
    size = len(dataloader.dataset)  # Total number of samples in the test dataset
    batch = len(dataloader)  # Number of batches in the test dataset
    test_loss, correct = 0, 0

    with torch.no_grad():  # Disable gradient calculation during inference
        for x, y in dataloader:
            outputs = model(x)  # Get predictions from the model
            test_loss += lossFunction(outputs, y.unsqueeze(1))  # Compute the loss
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()  # Apply sigmoid and threshold
            correct += (predictions == y).sum().item()  # Count correct predictions

    # Calculate average test loss and accuracy
    test_loss /= batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


# ----------------------------------------
# Main Function
# ----------------------------------------
def main():
    """
    Main function to train and test the neural network model on the airline satisfaction dataset.
    """
    device = "cpu"  # Choose device (CPU for now, can change to 'cuda' if GPU is available)
    print(f"Using {device} device")

    # Ensure the training and test datasets exist
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

    # Preprocess the training and test data to get DataLoaders and input size
    processedDataLoader, size = preProcessing(df_train)
    proceTestDataLoader, _ = preProcessing(df_test)

    # Load or initialize the model
    model = loadModel(device, size, "model.pth")

    # Set learning rate
    learning_rate = 0.001  # You can experiment with different learning rates

    # Train the model
    trainModel(model, processedDataLoader, device, learning_rate)

    # Test the model on test data
    testModel(proceTestDataLoader, model)

if __name__ == "__main__":
    main()
