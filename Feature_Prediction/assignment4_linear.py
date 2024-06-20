import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

gre, toefl, uni_ranking, sop, lor, cgpa, research, chance_admit = ([] for _ in range(8))

with open("Admission_Predict_Ver1.1.csv") as file:
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

gre, toefl, uni_ranking, sop, lor, cgpa, research, chance_admit = map(np.array, [gre, toefl, uni_ranking, sop, lor, cgpa, research, chance_admit])

params = [gre, toefl, uni_ranking, sop, lor, cgpa, research]

def func(params, x1, x2, x3, x4, x5, x6, x7, x8):
    return x1*(params[0]) + x2*(params[1]) + x3 * (params[2]) + x4 * (params[3]) + x5* (params[4]) + x6* (params[5]) + x7 * (params[6]) + x8

args, pcov = curve_fit(func, params, chance_admit)
y = func(params, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
x = np.std(y - chance_admit)
print(f"The standard devaition is {x}")
print(f"The arguments are : {args}")

result = np.corrcoef(chance_admit, y)[0][1] 
print(f"Correlation coefficient = {result}")

print(f"The effect of various parameters on Chances of admission are as follows:")
print(f"(1) GRE Score: {args[0]}")
print(f"(2) TOEFL Score: {args[1]}")
print(f"(3) University Ranking: {args[2]}")
print(f"(4) SOP: {args[3]}")
print(f"(5) LOR: {args[4]}")
print(f"(6) CGPA: {args[5]}")
print(f"(7) Research: {args[6]}")

variables = [cgpa, gre, toefl, lor, sop, uni_ranking]
labels = ['cgpa', 'gre', 'toefl', 'lor', 'sop', 'uni_ranking']

# Create scatter plots and save them
for var, label in zip(variables, labels):
    plt.figure()
    plt.xlim(0, 1.2)
    plt.scatter(var, y, label='y')
    plt.scatter(var, chance_admit, label='chance_admit')
    plt.legend()
    plt.xlabel(label)
    plt.ylabel('Value')
    plt.savefig(f'{label}_scatter_plot.png')
    plt.close()