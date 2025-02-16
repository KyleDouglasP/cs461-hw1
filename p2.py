from numpy import random
import numpy as np
import matplotlib.pyplot as plt 

depth = int(input("Please input depth (M) of simulation: "))

x = []

# Run 1000 Simulations 
for i in range(1000):
    location = 0
    # Simulate randomly going left or right with equal chance M times
    for j in range(depth):
        location += 1 if random.random() < 0.5 else -1
    x.append(location)

plt.title(f"1000 Simulations of Depth {depth} Galton Board")
plt.xlabel("Location")
plt.ylabel("Instances")

print(f"Mean: {np.mean(x)}")
print(f"Mean: {np.std(x)}")

bin_width = 1  
bins = range(min(x) - 1, max(x) + 2, bin_width)

plt.hist(x, bins=bins, edgecolor='black', align='left')
plt.show()

