import torch
import torch.optim as optim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

COST_THRESHOLD = 0.8

# Function to calculate Euclidean distance to nearest obstacle
def calculate_distance_to_obstacle(pos, cost_map, threshold):
    height, width = cost_map.size()
    pos_x, pos_y = pos[:, 0], pos[:, 1]
    
    distances = torch.full(pos_x.shape, float('inf'))
    for y in range(height):
        for x in range(width):
            if cost_map[y, x] > threshold:
                obstacle_pos = torch.tensor([x, y], device=pos.device, dtype=pos.dtype)
                dists = torch.norm(pos - obstacle_pos, dim=1)
                distances = torch.min(distances, dists)
    return distances

def bilinear_interpolation(pos, cost_map):
    height, width = cost_map.size()
    x = pos[:, 0]
    y = pos[:, 1]
    
    x0 = torch.floor(x).long().clamp(0, width - 1)
    x1 = (x0 + 1).clamp(0, width - 1)
    y0 = torch.floor(y).long().clamp(0, height - 1)
    y1 = (y0 + 1).clamp(0, height - 1)
    
    Ia = cost_map[y0, x0]
    Ib = cost_map[y1, x0]
    Ic = cost_map[y0, x1]
    Id = cost_map[y1, x1]
    
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())
    
    interpolated_cost = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return interpolated_cost.sum()

def collision_cost(pos, cost_map):
    cost = bilinear_interpolation(pos, cost_map)
    return cost

def inter_robot_distance(pos, d_min):
    dists = torch.norm(pos[:, None, :] - pos[None, :, :], dim=-1)
    penalty = torch.relu(d_min - dists).sum() / 2  # Each pair counted twice
    return penalty

def distance(pos_i, pos_j):
    return torch.norm(pos_i - pos_j)

def is_observable(pos_i, pos_j, orientation_i, D, FOV_min, FOV_max):
    dist = torch.norm(pos_i - pos_j)
    if dist > D:
        return False
    direction = torch.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
    rel_angle = direction - orientation_i
    rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))  # Normalize to [-pi, pi]
    return FOV_min <= rel_angle <= FOV_max

def required_rotation_to_observe(pos_i, pos_j, orientation_i, FOV_min, FOV_max):
    direction = torch.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
    rel_angle = direction - orientation_i
    rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))  # Normalize to [-pi, pi]
    
    if FOV_min <= rel_angle <= FOV_max:
        return torch.tensor(0.0, device=pos_i.device)
    
    # Calculate the minimal rotation needed
    rotation_needed = torch.min(torch.abs(rel_angle - FOV_min), torch.abs(FOV_max - rel_angle))
    return rotation_needed

def form_mst_tensor(positions, orientations, D, FOV_min, FOV_max):
    num_robots = len(positions)
    edges = []
    edge_weights = []

    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            weight = distance(positions[i], positions[j])
            rotation_i = required_rotation_to_observe(positions[i], positions[j], orientations[i], FOV_min, FOV_max)
            rotation_j = required_rotation_to_observe(positions[j], positions[i], orientations[j], FOV_min, FOV_max)
            total_rotation = torch.min(rotation_i, rotation_j)
            edges.append((i, j))
            edge_weights.append(torch.max(torch.tensor(0.0, device=positions.device), weight - torch.tensor(D, device=positions.device, requires_grad=True))+ total_rotation)

    sorted_edges = sorted(zip(edge_weights, edges), key=lambda x: x[0])

    parent = list(range(num_robots))
    rank = [0] * num_robots

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    mst_edges = []
    mst_weights = []
    mst_observe_weights = []
    for weight, (i, j) in sorted_edges:
        if find(i) != find(j):
            union(i, j)
            mst_edges.append((i, j))
            mst_weights.append(weight)

    return mst_edges, mst_weights

def mst_cost(positions, orientations, D, FOV_min, FOV_max):
    mst_edges, edge_weights = form_mst_tensor(positions, orientations, D, FOV_min, FOV_max)
    total_cost = torch.tensor(0.0, device=positions.device, requires_grad=True)
    for (u, v), weight in zip(mst_edges, edge_weights):
        if not is_observable(positions[u], positions[v], orientations[u], D, FOV_min, FOV_max):
            total_cost = total_cost + weight 
    return total_cost

def objective_function(pos, initial_positions, orientations, cost_map, d1, d2, D, FOV_min, FOV_max):
    positions = pos.reshape(-1, 2)
    change_cost = torch.norm(initial_positions - positions).sum()
    collision_cost_value = collision_cost(positions, cost_map)
    inter_robot_distance_penalty = inter_robot_distance(positions, d2)
    observation_graph_cost = mst_cost(positions, orientations, D, FOV_min, FOV_max)
    
    # total_cost = change_cost + collision_cost_value + inter_robot_distance_penalty + observation_graph_cost
    # total_cost = collision_cost_value + inter_robot_distance_penalty 
    total_cost =  observation_graph_cost + 3*collision_cost_value + inter_robot_distance_penalty
    # total_cost = observation_graph_cost
    

    return total_cost

def generate_cost_map(height, width, box_obstacles, sigma=6.0, drop_off_distance=3):
    cost_map = torch.zeros((height, width))
    for box in box_obstacles:
        x_min, y_min, x_max, y_max = box
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        for i in range(height):
            for j in range(width):
                if x_min <= j <= x_max and y_min <= i <= y_max:
                    distance_to_center = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    cost_map[i, j] = COST_THRESHOLD + np.exp(-distance_to_center**2 / (2 * sigma**2))
                else:
                    distances_to_edges = [
                        max(0, j - x_max),  # Right of the box
                        max(0, x_min - j),  # Left of the box
                        max(0, i - y_max),  # Above the box
                        max(0, y_min - i)   # Below the box
                    ]
                    min_distance_to_edge = np.sqrt(sum([d**2 for d in distances_to_edges]))
                    if min_distance_to_edge < drop_off_distance:
                        cost_map[i, j] += COST_THRESHOLD * np.exp(-min_distance_to_edge**2 / (2 * sigma**2))
                    else:
                        cost_map[i, j] += 0
    return cost_map

def plot_results(cost_map, initial_positions, optimized_positions, gradients, orientations, D, FOV_min, FOV_max, filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(cost_map.numpy(), cmap='gray', origin='lower')

    # Plot initial positions
    plt.scatter(initial_positions[:, 0], initial_positions[:, 1], color='blue', label='Initial Positions')

    # Plot optimized positions
    plt.scatter(optimized_positions[:, 0], optimized_positions[:, 1], color='green', label='Optimized Positions')

    # Plot gradients as arrows
    for i in range(initial_positions.shape[0]):
        plt.arrow(initial_positions[i, 0], initial_positions[i, 1], 
                  gradients[i, 0], gradients[i, 1], head_width=0.1, color='red')

    # Add cost values to the grid
    height, width = cost_map.size()
    for y in range(height):
        for x in range(width):
            plt.text(x, y, f"{cost_map[y, x]:.2f}", color='white', ha='center', va='center', fontsize=8)

    # Draw lines between optimized positions if they form an observation relationship
    for i in range(optimized_positions.shape[0]):
        for j in range(i + 1, optimized_positions.shape[0]):
            pos_i = torch.tensor(optimized_positions[i])
            pos_j = torch.tensor(optimized_positions[j])
            if is_observable(pos_i, pos_j, torch.tensor(orientations[i]), D, FOV_min, FOV_max) or is_observable(pos_j, pos_i, torch.tensor(orientations[j]), D, FOV_min, FOV_max):
                plt.plot([optimized_positions[i, 0], optimized_positions[j, 0]], 
                         [optimized_positions[i, 1], optimized_positions[j, 1]], 'yellow')

    plt.legend()
    plt.title('Cost Map and Robot Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(filename)
    plt.close()

# Example usage
initial_positions = torch.tensor([[5.0, 20.0], [5.0, 8.0], [10.0, 18.0], [10.0, 10.0]], requires_grad=True)
target_positions = torch.clone(initial_positions).detach().requires_grad_(True)
orientations = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)

height, width = 30, 30
box_obstacles = [(5, 7, 25, 12), (5, 18, 25, 25)]  # Example box obstacle positions
cost_map = generate_cost_map(height, width, box_obstacles)
d1 = 2  # Minimum distance from obstacles
d2 = 3  # Minimum distance between robots
D = 3.4 # Maximum observation distance
# FOV_min = -1.35 * torch.pi
FOV_min = -0.45 * torch.pi

FOV_max = 0.45 * torch.pi
# FOV_max = 1.35 * torch.pi

alpha = 0.1
COST_THRESHOLD = 0.6  # Example threshold for obstacles in cost map

# Set a random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

iteration_nums = [10, 20, 30, 50, 75, 100, 150, 200]

for iter_num in iteration_nums:
    # Reset target_positions for each iteration test
    target_positions = torch.clone(initial_positions).detach().requires_grad_(True)
    optimizer = optim.Adam([target_positions], lr=0.1)

    for i in range(iter_num):
        optimizer.zero_grad()
        loss = objective_function(target_positions, initial_positions, orientations, cost_map, d1, d2, D, FOV_min, FOV_max)
        loss.backward()
        optimizer.step()

    gradients = target_positions.grad if target_positions.grad is not None else torch.zeros_like(target_positions)

    plot_results(cost_map, initial_positions.detach().numpy(), target_positions.detach().numpy(), gradients.detach().numpy(), orientations.detach().numpy(), D, FOV_min, FOV_max, f"result_{iter_num}_iterations.png")