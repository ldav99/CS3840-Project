import pandas as pd

def main():
    df = pd.read_csv('data/train.csv')

    print(df.head())
    print(df.info())

if __name__=="__main__":
    main()