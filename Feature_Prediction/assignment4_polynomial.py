import csv
import numpy as np
import matplotlib.pyplot as plt                 # importing libraries
from scipy.optimize import curve_fit

gre, toefl, uni_ranking, sop, lor, cgpa, research, chance_admit = [], [], [], [], [], [], [], []        # initialising lists

with open('Admission_Predict_Ver1.1.csv') as file:                  # reading the input file and storing it in lists
    data = csv.reader(file)             
    next(data)
    for row in data:
        gre.append(float(row[1])/340)
        toefl.append(float(row[2])/120)
        uni_ranking.append(float(row[3])/5)
        sop.append(float(row[4])/5)
        lor.append(float(row[5])/5)
        cgpa.append(float(row[6])/10)
        research.append(float(row[7]))
        chance_admit.append(float(row[8]))

gre = np.array(gre)
toefl = np.array(toefl)
uni_ranking = np.array(uni_ranking)
sop = np.array(sop)                                      # converting lists to np arrays
cgpa = np.array(cgpa)
research = np.array(research)
chance_admit = np.array(chance_admit)
params = [gre, toefl, uni_ranking, sop, lor, cgpa, research]

def func(params, x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, y5, y6, y7):          # function definition
    return x1 * (params[0]**y1) + x2 * (params[1]**y2) + x3 * (params[2]**y3) + x4 * (params[3]**y4) + x5 * (params[4]**y5) + x6 *(params[5]**y6) + x7 * (params[6]**y7) + x8

initial_guess = np.ones(15)                

args, _ = curve_fit(func, params, chance_admit, maxfev = 1000000)               # calling curve_fit to optimize
y = func(params, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14])

x = np.std(y - chance_admit)                    # calculating std deviation
print(f"Standard deviation of estimated function = {x}")
print(f"Arguments of curve_fit = {args}")           
print('\n')

result = np.corrcoef(chance_admit, y)[0][1]             # calculating correlation coefficient
print(f"Correlation coefficient of estimated function and given data = {result}")
print('\n')

def calculate_correlation_and_derivative(variable, args, arg_index, power_index):           # function to calculate correlation coefficient 

    params = [0.5] * 7
    params[arg_index] = variable                                                            # and partial derivative

    # Calculate the output using the function
    y = func(params, *args)

    # Calculate the correlation coefficient
    correlation = np.corrcoef(variable, chance_admit)[0][1]

    # Calculate the average partial derivative
    derivative = np.sum(args[arg_index] * args[power_index] * (variable ** (args[power_index] - 1))) / 500

    return correlation, derivative

# Calculate and print the results for different variables
cgpa_corr, cgpa_derivative = calculate_correlation_and_derivative(cgpa, args, 5, 13)
print(f"Correlation coefficient with CGPA = {cgpa_corr}")
print(f"Average partial derivative for CGPA = {cgpa_derivative}")
print("\n")

gre_corr, gre_derivative = calculate_correlation_and_derivative(gre, args, 0, 8)
print(f"Correlation coefficient with GRE Score = {gre_corr}")
print(f"Average partial derivative for GRE Score = {gre_derivative}")
print("\n")

toefl_corr, toefl_derivative = calculate_correlation_and_derivative(toefl, args, 1, 9)
print(f"Correlation coefficient with TOEFL Score = {toefl_corr}")
print(f"Average partial derivative for TOEFL Score = {toefl_derivative}")                   # printing correlation coefficient and partial derivative
print("\n")

uni_rank_corr, uni_rank_derivative = calculate_correlation_and_derivative(uni_ranking, args, 2, 10)
print(f"Correlation coefficient with University Ranking = {uni_rank_corr}")
print(f"Average partial derivative for University Ranking = {uni_rank_derivative}")
print("\n")

sop_corr, sop_derivative = calculate_correlation_and_derivative(sop, args, 3, 11)
print(f"Correlation coefficient with SOP = {sop_corr}")
print(f"Average partial derivative for SOP = {sop_derivative}")
print("\n")

lor_corr, lor_derivative = calculate_correlation_and_derivative(lor, args, 4, 12)
print(f"Correlation coefficient with LOR = {lor_corr}")
print(f"Average partial derivative for LOR = {lor_derivative}")
print("\n")

research_corr, research_derivative = calculate_correlation_and_derivative(research, args, 6, 14)
print(f"Correlation coefficient with Research = {research_corr}")
print(f"Average partial derivative for Research = {research_derivative}")
print('\n')

variables = [cgpa, gre, toefl, lor, sop, uni_ranking, research]                 
titles = ['CGPA vs. Chance of Admission', 'GRE Score vs. Chance of Admission',
          'TOEFL Score vs. Chance of Admission', 'Letter of Recommendation vs. Chance of Admission',
          'Statement of Purpose vs. Chance of Admission', 'University Ranking vs. Chance of Admission',
          'Research vs. Chance of Admission']

# Define x-axis limits
x_limits = (0, 1.2)

# Looping through variables and titles to create and save plots
for var, title in zip(variables, titles):
    plt.figure()
    plt.scatter(var, y, marker='*')
    plt.scatter(var, chance_admit)
    plt.title(title)                                                       
    plt.xlabel(title.split(' vs. ')[0])  # Extract the variable name for the x-label
    plt.ylabel('Chance of Admission')
    plt.xlim(*x_limits)
    plt.savefig(title.split(' vs. ')[0].lower() + '_poly.png')
    plt.clf()  # Clear the figure

print(f"The order in which the student should put efforts `assuming equal efforts are required for equal percentage changes in all parameters` is(calculated according to average partial derivative):")
print('(1): CGPA')
print('(2): GRE Score')
print('(3): TOEFL Score')
print('(4): LOR')
print('(5): SOP')                   # Printing Order of dependence
print('(6): Research')
print('(7): University Ranking')