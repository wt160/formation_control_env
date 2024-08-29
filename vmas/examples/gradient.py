import torch

def collision_cost(pos, cost_map):
    height, width = cost_map.size()
    pos_x, pos_y = pos[:, 0], pos[:, 1]
    x0, y0 = pos_x.floor().long(), pos_y.floor().long()
    x1, y1 = x0 + 1, y0 + 1
    x0, x1 = x0.clamp(0, width - 1), x1.clamp(0, width - 1)
    y0, y1 = y0.clamp(0, height - 1), y1.clamp(0, height - 1)
    x_frac, y_frac = pos_x - x0.float(), pos_y - y0.float()
    val00 = cost_map[y0, x0]
    val01 = cost_map[y0, x1]
    val10 = cost_map[y1, x0]
    val11 = cost_map[y1, x1]
    val0 = val00 * (1 - x_frac) + val01 * x_frac
    val1 = val10 * (1 - x_frac) + val11 * x_frac
    return (val0 * (1 - y_frac) + val1 * y_frac).sum()

def inter_robot_distance(pos, d_min):
    dists = torch.norm(pos[:, None, :] - pos[None, :, :], dim=-1)
    return torch.relu(d_min - dists).sum() / 2  # Each pair counted twice

def observation_cost(pos, orientations, alpha):
    num_robots = pos.shape[0]
    visibility = torch.zeros(1, device=pos.device)
    for i in range(num_robots):
        for j in range(num_robots):
            if i != j:
                distance = torch.norm(pos[i] - pos[j])
                direction = torch.atan2(pos[j, 1] - pos[i, 1], pos[j, 0] - pos[i, 0])
                visibility += torch.exp(-alpha * distance**2) * torch.cos(orientations[i] - direction)
    return visibility

# Example usage with four robots
pos = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], requires_grad=True)
orientations = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)
cost_map = torch.rand(10, 10)
d_min = 2.0
alpha = 0.1

collision_costs = collision_cost(pos, cost_map)
distance_penalties = inter_robot_distance(pos, d_min)
visibility_costs = observation_cost(pos, orientations, alpha)

total_cost = 3 * collision_costs + distance_penalties - visibility_costs  # Adjust the weights as necessary

# Compute gradients
total_cost.backward()
print("Gradients w.r.t positions:", pos.grad)
print("Gradients w.r.t orientations:", orientations.grad)