import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Crop_recommendation.csv")
#print(df.head())
all_columns = df.columns[:-1]
label_encoder = LabelEncoder()
X = df[all_columns]
y = label_encoder.fit_transform(df["label"])
#print(X.shape, y.shape)
label_dict = {}
for i in range(22):
    label_dict[i] = label_encoder.inverse_transform([i])[0]
#print(label_dict)

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.2, random_state = 0)
#print(f"Train Data: {X_train.shape}, {y_train.shape}")
#print(f"Train Data: {X_test.shape}, {y_test.shape}")

knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 4))
knn_pipeline.fit(X_train, y_train)

predictions = knn_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on Test Data: {accuracy*100}%")

predictions = knn_pipeline.predict(X.values)
accuracy = accuracy_score(y, predictions)
print(f"Accuracy on Whole Data: {accuracy*100}%")

pickle.dump(knn_pipeline, open("knn_pipeline.pkl", "wb"))