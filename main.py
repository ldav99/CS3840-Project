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

def main():
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
    
    # df_test = pd.read_csv('data/test.csv')
    # df_train_small = df_train.head()
    
    print(df_train.head())
    print(df_train.info())

    #Pre Process the Data
    #---------------------------------------

    expected_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    missing_cols = [col for col in expected_columns if col not in df_train.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing and cannot be one-hot encoded: {missing_cols}")
    df_train_ohe = pd.get_dummies(df_train, columns=[col for col in expected_columns if col in df_train.columns])

    print(df_train_ohe.info())  # Debugging information

    label_encoder = LabelEncoder()
    if 'satisfaction' in df_train.columns:
        df_train['satisfaction'] = label_encoder.fit_transform(df_train['satisfaction'])
    else:
        print("Error: 'satisfaction' column not found.")

    # Normalization of numerical features
    # ensures features have a similar scale, reducing bias from large-valued features
    numerical_features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                          'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                          'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
                          'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    
    # Handle missing values before scaling
    df_train_ohe.fillna(df_train_ohe.mean(), inplace=True)

    scaler = StandardScaler()
    df_train_ohe[numerical_features] = scaler.fit_transform(df_train_ohe[numerical_features])

    
#Convert DataFrame to tensor for pytorch
    # Ensure all categorical columns are numeric
    df_train_ohe = df_train_ohe.astype(float)  # Convert all columns to float

    # Convert DataFrame to tensor
    tensor = torch.tensor(df_train_ohe.values, dtype=torch.float32)
    print(f"Tensor shape: {tensor.shape}")

    # Call external functions
 


#Call external functions
#---------------------------------------
    model = neuralNetwork.NeuralNetwork().to(device)
    print(model)



#Results 
#---------------------------------------
    #Plot Accuracy
    #plt.plot(data)
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.title("Accuracy")
    #plt.show()

    #Plot Precision and Recall
    #Google says these are one graph
    #plt.plot(data)
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.title("Precision-Recall")
    #plt.show()



if __name__=="__main__":
    main()