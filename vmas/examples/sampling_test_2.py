import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def sample_positions(num_samples, search_radius):
    positions = []
    for _ in range(num_samples):
        radius = np.random.uniform(0, search_radius)  # Uniform radius
        angle = np.random.uniform(0, 2 * math.pi)     # Uniform angle

        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        positions.append((dx, dy))

    return positions

# Parameters
num_samples = 100
search_radius = 0.25  # Maximum radius for sampling

# Generate samples
positions = sample_positions(num_samples, search_radius)

# Plotting
x_vals = [pos[0] for pos in positions]
y_vals = [pos[1] for pos in positions]
plt.figure(figsize=(6, 6))
plt.scatter(x_vals, y_vals, alpha=0.5)
plt.title("Uniformly Distributed Sampled Positions")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.axis('equal')
plt.show()