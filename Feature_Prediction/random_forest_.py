import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor

# Load the data using Pandas and specify the header row
header_values = ["Serial No.", "GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research", "Chance of Admit"]
data = pd.read_csv("Admission_Predict_Ver1.1.csv", names=header_values, skiprows=1)

# Normalize the columns as needed
data['GRE Score'] = data['GRE Score'] / 340
data['TOEFL Score'] = data['TOEFL Score'] / 120
data['University Rating'] = data['University Rating'] / 5
data['SOP'] = data['SOP'] / 5
data['LOR'] = data['LOR'] / 5
data['CGPA'] = data['CGPA'] / 10

# Define features and target
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y = data['Chance of Admit']

# Train a Random Forest Regressor
model = RandomForestRegressor()
model.fit(X, y)

# Get feature importances
feature_importance = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
