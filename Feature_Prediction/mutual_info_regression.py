import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.feature_selection import mutual_info_regression

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

# Calculate mutual information
mutual_info = mutual_info_regression(X, y)
mutual_info_series = pd.Series(mutual_info, index=X.columns)
mutual_info_series = mutual_info_series.sort_values(ascending=False)
print(mutual_info_series)
