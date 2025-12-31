import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from ann import setup_ann


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

ct = ColumnTransformer([("ohe", OneHotEncoder(), [1])], remainder="passthrough")
x = np.array(ct.fit_transform(x), dtype=np.str_)
x = x[:, 1:]

print("\nCountry encoded X division")
print(x)

sc = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train_df = pd.DataFrame(x_train)
print("\nX Dataframes after train and test transform")
print(x_train_df)

classifier = setup_ann()
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

y_prediction = classifier.predict(x_test)
y_prediction = (y_prediction > 0.5)

cm = confusion_matrix(y_test, y_prediction)
print("\nConfusion matrix")
print(cm)

accuracy = accuracy_score(y_test, y_prediction)
print("\nAccuracy score")
print(accuracy)
