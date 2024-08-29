import torch
import math
import matplotlib.pyplot as plt

def sample_positions(num_samples, radial_map, angular_map, search_radius):
    grid_size = len(radial_map)
    positions = []

    for _ in range(num_samples):
        radial_distribution = torch.softmax(radial_map, 0)
        angular_distribution = torch.softmax(angular_map, 0)

        radius_idx = torch.multinomial(radial_distribution, 1).item()
        angle_idx = torch.multinomial(angular_distribution, 1).item()

        radius = (radius_idx / grid_size) * search_radius
        angle = (angle_idx / grid_size) * 2 * math.pi

        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        positions.append((dx, dy))

    return positions

# Simulate the sampling
num_samples = 100
grid_size = 20
search_radius = 0.25
radial_map = torch.zeros(grid_size)  # Uniform radial map
angular_map = torch.zeros(grid_size)  # Uniform angular map

positions = sample_positions(num_samples, radial_map, angular_map, search_radius)

# Plotting
x_vals = [pos[0] for pos in positions]
y_vals = [pos[1] for pos in positions]
plt.figure(figsize=(6, 6))
plt.scatter(x_vals, y_vals, alpha=0.5)
plt.title("Sampled Positions Distribution")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.axis('equal')
plt.show()
