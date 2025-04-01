#Monica Brutto, Luke Davidson, Sam Webster
#CS-3840
#Dr. Wen Zhang 
#Final Project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import neuralNetwork
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")
    df_train = pd.read_csv('data/train.csv')
    # df_test = pd.read_csv('data/test.csv')
    df_train_small = df_train.head()
    


#     #print(df_train.head())
#     print(df_train.info())

# #Pre Process the Data
# #---------------------------------------
    df_train_ohe = pd.get_dummies(df_train_small, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction'])
    #print(df_train_ohe.info())

    # Normalization of numerical features
    numerical_features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                         'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                         'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
                         'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    # ensures features have a similar scale, reducing bias from large-valued features
    scaler = StandardScaler()
    df_train_ohe[numerical_features] = scaler.fit_transform(df_train_ohe[numerical_features])
    
#Convert DataFrame to tensor for pytorch
    print(df_train_ohe.info())
    print('-------')
    print(df_train_ohe.dtypes)
    # temp = df_train_ohe.to_numpy()
    # tensor = torch.from_numpy(temp)


#Call external functions
#---------------------------------------
    model = neuralNetwork.NeuralNetwork().to(device)

    #print(model(tensor))



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