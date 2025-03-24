#Monica Brutto, Luke Davidson, Sam Webster
#CS-3840
#Dr. Wen Zhang 
#Final Project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functions

def main():
    df = pd.read_csv('data/train.csv')

    print(df.head())
    print(df.info())

#Call external functions
#---------------------------------------



#Results
#---------------------------------------
    #Plot Accuracy
    #plt.plot(data)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Accuracy")
    #plt.show()

    #Plot Precision and Recall
    #Google says these are one graph
    #plt.plot(data)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title("Precision-Recall")
    #plt.show()



if __name__=="__main__":
    main()