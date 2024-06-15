import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import os

# Define the adjacency matrix function
def adjacency_matrix(positions, rotations, fov, max_distance):
    num_agents = positions.shape[0]
    dx = positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0)
    dy = positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0)
    distances = torch.sqrt(dx**2 + dy**2)
    angles = torch.atan2(dy, dx) - rotations.unsqueeze(1)
    angles = (angles + torch.pi) % (2 * torch.pi) - torch.pi
    mask = (angles.abs() <= fov / 2) & (distances <= max_distance)
    adj_matrix = mask.float()
    adj_matrix.fill_diagonal_(0)
    return adj_matrix

# Define the connectivity gap
def connectivity_gap(positions, rotations, fov, max_distance):
    num_agents = positions.shape[0]
    dx = positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0)
    dy = positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0)
    distances = torch.sqrt(dx**2 + dy**2)
    angles = torch.atan2(dy, dx) - rotations.unsqueeze(1)
    angles = (angles + torch.pi) % (2 * torch.pi) - torch.pi
    fov_mask = (angles.abs() <= fov / 2)
    distance_mask = (distances <= max_distance)
    connection_mask = fov_mask & distance_mask
    gap = (1 - connection_mask.float()) * distances
    gap.fill_diagonal_(0)
    return gap

# Define the potential field for connectivity
def potential_field(gap, d0=1.0):
    attractive_force = torch.where(gap > 0, 0.5 * (gap**2 / d0**2), torch.zeros_like(gap))
    potential = torch.sum(attractive_force) / 2.0  # Sum of potentials for all pairs
    return potential

# Define the combined objective function
def combined_objective(positions, rotations, fov, max_distance, alpha=1.0, d0=1.0):
    gap = connectivity_gap(positions, rotations, fov, max_distance)
    potential = potential_field(gap, d0)
    return alpha * potential

# Define the gradient ascent function
def optimize_positions_and_rotations(positions, rotations, fov, max_distance, alpha=1.0, d0=1.0, learning_rate=0.01, iterations=100):
    positions = positions.clone().requires_grad_(True)
    rotations = rotations.clone().requires_grad_(True)
    
    for iteration in range(iterations):
        objective = combined_objective(positions, rotations, fov, max_distance, alpha, d0)
        
        objective.backward()
        
        with torch.no_grad():
            if positions.grad is not None:
                positions -= learning_rate * positions.grad
                positions.grad.zero_()
            else:
                print(f"Iteration {iteration + 1}: Positions gradient is None")
                
            if rotations.grad is not None:
                rotations -= learning_rate * rotations.grad
                rotations.grad.zero_()
            else:
                print(f"Iteration {iteration + 1}: Rotations gradient is None")
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration + 1}: Objective = {objective.item()}")
    
    return positions.detach(), rotations.detach()

# Define a function to plot the positions
def plot_positions(positions, rotations, iteration, folder, fov, max_distance):
    plt.figure(figsize=(8, 8))
    G = nx.Graph()
    num_agents = positions.shape[0]
    adj_matrix = adjacency_matrix(positions, rotations, fov, max_distance)
    for i in range(num_agents):
        G.add_node(i, pos=(positions[i, 0].item(), positions[i, 1].item()))
        for j in range(i + 1, num_agents):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j].item())
    
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='grey')
    plt.title(f"Iteration {iteration}")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"iteration_{iteration}.png"))
    plt.close()

# Example usage
num_agents = 5
positions = torch.rand(num_agents, 2)
rotations = torch.rand(num_agents) * 2 * torch.pi
fov = 0.35 * torch.pi
max_distance = 10.0

optimized_positions, optimized_rotations = optimize_positions_and_rotations(positions, rotations, fov, max_distance)
print("Optimized Positions:\n", optimized_positions)
print("Optimized Rotations:\n", optimized_rotations)
