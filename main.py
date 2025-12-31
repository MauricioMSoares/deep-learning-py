import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


dataset = pd.read_csv("Churn_Modelling.csv")
print("\nRaw dataset")
print(dataset)

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print("\nX division")
print(x)
print("\nY division")
print(y)

encode_gender = LabelEncoder()

x[:, 2] = encode_gender.fit_transform(x[:, 2])

print("\nGender encoded X division")
print(x)
