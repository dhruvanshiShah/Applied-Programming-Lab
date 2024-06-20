import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data using Pandas and specify the header row
data = pd.read_csv("Admission_Predict_Ver1.1.csv", header=0)  # Set header=0 to use the first row as column names

# Normalize the columns as needed
data['GRE Score'] = data['GRE Score'] / 340
data['TOEFL Score'] = data['TOEFL Score'] / 120
data['University Rating'] = data['University Rating'] / 5
data['SOP'] = data['SOP'] / 5
data['LOR '] = data['LOR '] / 5
data['CGPA'] = data['CGPA'] / 10

# Now you can compute the correlation matrix
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix['Chance of Admit '].sort_values(ascending=False)
print(correlation_with_target)