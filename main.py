import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Churn_Modelling.csv")
print(dataset)

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print(x)
print(y)