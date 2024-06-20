from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")                         # for ignoring Overflow warnings

PLANCK_CONSTANT = np.float64("6.62607015e-34")
SPEED_OF_LIGHT = np.float64("299792458")                # defining Constants
BOLTZMANN_CONSTANT = np.float64("1.380649e-23")

data = np.loadtxt("dataset3.txt")                       # importing datasets and extracing data
f = data[:, 0]
B = data[:, 1]

def mapping(x, T):                                     
    return (2*PLANCK_CONSTANT*(x**3)/((SPEED_OF_LIGHT**2)*(np.exp((PLANCK_CONSTANT*x)/(BOLTZMANN_CONSTANT*T)) - 1)))

initial_guess_T = np.float64("150") 
args, _ = curve_fit(mapping, f, B, initial_guess_T, maxfev = 1000000)           # calling curve_fit to get temperature
B1 = mapping(f, args[0])

print(f"T = {args[0]}")

plt.grid('True')
plt.plot(f, B, label = "Noisy data", color = "cyan")
plt.plot(f, B1, label = "Estimated Plot", color = "blue")
plt.xlabel("Temperature (T) [K]")                                           # plotting the curve with appropriate labels
plt.ylabel("Value of radiation (B)")                
plt.legend(["Noisy data", "Estimated Plot"], title = 'Graph of f v/s B')
plt.savefig('dataset3_1.png')