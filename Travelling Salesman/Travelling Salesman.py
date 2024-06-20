import math
import random
import numpy as np    
import matplotlib.pyplot as plt

def distance(cities, cityorder):
    total_distance = 0
    num_cities = len(cities)

    for i in range(num_cities):
        city1 = cities[cityorder[i]]
        city2 = cities[cityorder[(i + 1) % num_cities]]
        x1, y1 = city1
        x2, y2 = city2
        total_distance += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return total_distance


def tsp(cities):
    num_cities = len(cities)
    current_order = list(range(num_cities))
    current_distance = distance(cities, current_order)

    # Parameters for simulated annealing
    initial_temperature = 1000.0
    cooling_rate = 0.995
    num_iterations = (num_cities)**4

    best_order = current_order.copy()
    best_distance = current_distance

    for iteration in range(num_iterations):
        # Generate a random neighboring solution by swapping two cities
        i, j = random.sample(range(num_cities), 2)
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_distance = distance(cities, new_order)

        # Calculate the change in distance
        delta_distance = new_distance - current_distance

        # Accept the new solution with a certain probability
        if delta_distance < 0 or random.random() < math.exp(-delta_distance / initial_temperature):
            current_order = new_order
            current_distance = new_distance

        # Update the best solution if needed
        if current_distance < best_distance:
            best_order = current_order.copy()
            best_distance = current_distance

        # Reduce the temperature
        initial_temperature *= cooling_rate

    return best_order, best_distance

cities = []
file_name = "TSP.txt"
with open(file_name, "r") as file:
    num_cities = int(file.readline())
    cities = []
    for _ in range(num_cities):
        x, y = map(float, file.readline().split())
        cities.append((x, y))

print(cities)
optimal_order, optimal_distance = tsp(cities)
print("Optimal Order:", optimal_order)
print("Optimal Distance:", optimal_distance)

x_cities = np.array([city[0] for city in cities])
y_cities = np.array([city[1] for city in cities])
xplot = x_cities[optimal_order] 
yplot = y_cities[optimal_order]
xplot = np.append(xplot, xplot[0])
yplot = np.append(yplot, yplot[0])
plt.plot(xplot, yplot, 'o-')
plt.show()