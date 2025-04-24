import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score


df = pd.read_csv("diabetes_012_health_indicators_BRFSS2021.csv")
df = df.sample(n=25000, random_state=42)


selected_features_cls = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
    'Sex', 'Age'
]

X_cls = df[selected_features_cls]
y_cls = df["Diabetes_012"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_cls, y_cls)

scores = cross_val_score(clf, X_cls, y_cls, cv=5, scoring='accuracy')
print(f"Classification Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

with open("model/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)


selected_features_reg = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age'
]

X_reg = df[selected_features_reg]
y_reg = df["BMI"]

reg = LinearRegression()
reg.fit(X_reg, y_reg)

with open("model/regressor.pkl", "wb") as f:
    pickle.dump(reg, f)

print(" Models trained and saved successfully.")
