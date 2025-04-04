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
from torch.utils.data import Dataset, DataLoader

#Function to run model
#TODO Add number of Epochs??
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

    
#Convert DataFrame to tensor for pytorch
    # Ensure all categorical columns are numeric
    dataset_ohe = dataset_ohe.astype(float)  # Convert all columns to float

    # Convert DataFrame to tensor
    tensor = torch.tensor(dataset_ohe.values, dtype=torch.float32)
    print(f"Tensor shape: {tensor.shape}")

#Call external functions
#---------------------------------------
    input_size = dataset_ohe.shape[1]  # Number of features in the dataset
    model = neuralNetwork.NeuralNetwork(input_size).to(device)  # Pass input size to the constructor

    # Pass the tensor through the model
    output = model(tensor.to(device))
    print(f"Model output: {output}")
    output_array = output.squeeze().detach().cpu().numpy()
    return output_array


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
    
    print(df_train.head())
    print(df_train.info())

#Call model on train and test
    trainResult = callModel(df_train, device)
    testResult = callModel(df_test, device)

#Results 
#---------------------------------------
    #Plot Accuracy
    plt.boxplot(trainResult)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Accuracy")
    plt.show()
    
    plt.boxplot(testResult)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Accuracy")
    plt.show()

    #Plot Precision and Recall
    #Google says these are one graph
    #plt.plot(data)
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.title("Precision-Recall")
    #plt.show()



if __name__=="__main__":
    main()