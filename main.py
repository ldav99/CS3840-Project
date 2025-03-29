#Monica Brutto, Luke Davidson, Sam Webster
#CS-3840
#Dr. Wen Zhang 
#Final Project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import neuralNetwork
import torch
from torch import nn

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")
    #df_train = pd.read_csv('data/train.csv')
    #df_test = pd.read_csv('data/test.csv')

#     #print(df_train.head())
#     print(df_train.info())

# #Pre Process the Data
# #---------------------------------------
    # df_train_ohe = pd.get_dummies(df_train, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction'])
    # print(df_train_ohe.info())


#Call external functions
#---------------------------------------
    model = NeuralNetwork().to(device)
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