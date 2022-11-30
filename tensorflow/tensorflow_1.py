import pandas as pd
import tensorflow as tf
import os


os.chdir("..")
filename = "covid.csv"

df = pd.read_csv(filename)

print(df)

if __name__ == "__main__":
    pass