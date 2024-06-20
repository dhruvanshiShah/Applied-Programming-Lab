from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")                   # for ignoring Overflow warnings

PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT = 299792458                          # defining Constants
BOLTZMANN_CONSTANT = 1.380649e-23

data = np.loadtxt("dataset3.txt")                   # importing datasets and extracing data
f = data[:, 0]
B = data[:, 1]

def mapping(x, T, h, c, k_b):
    return (2*h*(x**3)/((c**2)*(np.exp((h*x)/(k_b*T)) - 1)))                    

initial_guess = [5246, PLANCK_CONSTANT, SPEED_OF_LIGHT, BOLTZMANN_CONSTANT]
args, pcov= curve_fit(mapping, f, B, initial_guess, maxfev = 100000)        # calling curve_fit to get temperature and Constants
B0 = mapping(f, args[0], args[1], args[2], args[3])

print(f" Temperature = {args[0]}, planck's constant = {args[1]}, speed of light = {args[2]}, boltzmann constant = {args[3]}")

plt.xlabel("Frequency (f)")
plt.ylabel("Value of B")
plt.grid(True)                                      # plotting the curves and saving them
plt.plot(f, B, color = "cyan")
plt.plot(f, B0, color = "blue")
plt.legend(['Noisy data', 'Plotted curve'], title = 'Graph of frequency(f) v/s Spectral Intensity(B)')
plt.savefig('dataset3_2.png')