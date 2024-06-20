from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dataset2.txt")
x = data[:, 0]
y = data[:, 1]                              # Extracting data from dataset

xnew = x
ynew = y
for i in range(100):                        # Number of iterations for smoothening curve to remove noise
    x_temp = []
    y_temp = []
    x_temp.append(xnew[0])                  # Appending  initial values as they won't be appended in this algorithm
    y_temp.append(ynew[0])
    for j in range(1, len(ynew)):
        y_temp.append((ynew[j] + ynew[j - 1]) * 0.5)
        x_temp.append((xnew[j] + xnew[j - 1]) * 0.5)
    x_temp.append(xnew[len(ynew)- 1])
    y_temp.append(ynew[len(ynew) - 1])
    ynew = (np.array(y_temp))               # Converting to np array and storing 
    xnew = (np.array(x_temp))
 
dict = {key: value for key, value in zip(xnew, ynew)}           # dict having values of y corresponding to x
xmin = 100
xmax = -100                                         # defining xmin and xmax for calculating T
pos = xnew[xnew>0]
neg = xnew[xnew<0]                                  # lists for storing positive and negative values of x
neg = neg[::-1]

for i in pos:
    # print(dict[i])
    if dict[i] <= 0.02:
        xmax = i
        break
skip = 0
for i in neg:
    if dict[i] >= -0.02:
        xmin = i
        if skip == 1:                           # skipping as 0 is also satifying the conditions, and if not ignored(find out after analysing data)
            break
        skip = 1
T = xmax - xmin                                 # Defining time period as the difference between values having same y(0)

def f1(x, p):                                
    return p[0] * np.sin((2*(np.pi)*x)/T )+ p[1] * np.sin((3*2*(np.pi)*x)/T) + p[2] * np.sin((5*2*(np.pi)*x)/T)

def f2(x, p0, p1, p2):                          # function for curve_fit
    return p0 * np.sin((2*(np.pi)*x)/T )+ p1 * np.sin((3*2*(np.pi)*x)/T) + p2 * np.sin((5*2*(np.pi)*x)/T)

M = np.column_stack([np.sin(((2*(np.pi))*x)/T), np.sin(3*((2*(np.pi))*x)/T), np.sin(5*((2*(np.pi))*x)/T)])
p, _, _, _ = np.linalg.lstsq(M, y, rcond=None)

yn = f1(x, p)
print(f"The estimated parameters are: {p[0]} sin(ωx) + {p[1]} sin(3ωx) + {p[2]}sint(5ωx)")

args, pcov = curve_fit(f2, x, y)
print(f"The estimated parameters from curve_fit by are: {args[0]} sin(ωx) + {args[1]} sin(3ωx) + {args[2]}sint(5ωx)")
yn1 = f2(x, args[0], args[1], args[2])

print(f"The standard deviation is {np.std(y - yn)}.")

plt.xlabel("x")
plt.ylabel("y = {p[0]} sin(ωx) + {p[1]} sin(3ωx) + {p[2]}sint(5ωx)")
plt.plot(x, y, color = "violet")
plt.plot(xnew, ynew, color = "red")                                 # plotting and saving plots
plt.plot(x, yn, color = "blue")
plt.plot(x, yn1, color = "green")
plt.grid('True')
plt.legend(['noisy data','smoothened', 'calculated from lstsq', 'calculated from curve_fit'], title='Plot for Dataset-2')
plt.savefig('dataset2.png')