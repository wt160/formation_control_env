import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import GATConv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from tqdm import tqdm
from tensordict import TensorDict
from tensordict.nn.distributions import NormalParamExtractor
import pickle
from tensordict.nn import TensorDictModule
import numpy as np
import math
# Load the collected data
with open('collected_data_tunnel_0.pkl', 'rb') as f:
    collected_data = pickle.load(f)
# print("collected_data:{}".format(collected_data))
# print("collected tensor shape:{}".format(collected_data[0]['graph_tensor'].shape))
# Process the data
dataset = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


noise_total_num = 75
data_length = len(collected_data)

for data_index, data_point in enumerate(collected_data):
    if data_index > data_length / 3:
        break
    graph_data_list = data_point['graph_tensor']  # List of graphs with length batch_dim
    optimized_target_pos = data_point['optimized_target_pos']  # Dict of agent positions with batch_dim

    agent_names = sorted(optimized_target_pos.keys())
    # optimized_target_pos: dict with keys as agent names and values as tensors of shape [batch_dim, 2]

    batch_dim = len(graph_data_list)
    for batch_idx in range(batch_dim):
        # Get graph data for this batch index
        graph_data = graph_data_list[batch_idx]

        # Reconstruct the PyG Data object
        x = torch.tensor(graph_data['x'], dtype=torch.float, device="cpu")
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long, device="cpu")
        edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float, device="cpu") if graph_data['edge_attr'] is not None else None

        # Use the full x as features
        features = x  # No separation of categories
        num_nodes = x.shape[0]
        # Create Data object
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

        # Prepare target positions for this batch index
        target_positions = [optimized_target_pos[name][batch_idx] for name in agent_names]  # List of tensors [2]
        target_positions = np.stack(target_positions)  # Shape: [num_agents, 2]
        data.y = torch.tensor(target_positions, dtype=torch.float, device="cpu")
        target_positions_tensor = torch.tensor(target_positions, dtype=torch.float, device="cpu")
        # Append to dataset
        # if num_nodes > 5:
        dataset.append(data)
        for noise_num in range(noise_total_num):
            # Clone the original features so we don't modify them in-place
            noise_features = features.clone()

            # Add noise to the current relative position part of the features
            # Assuming the current relative positions are in columns [4:6] for the first 5 agents
            # Adjust indices based on your actual feature layout
            # For each noise_num, add a random displacement
            noise = -0.25 + 0.5*torch.rand((5, 2), device="cpu")  # shape: [5, 2]
            noise_features[:5, 4:6] += noise
            angle_noise =-math.pi*0.3 +  math.pi*0.6*torch.rand((5,1), device="cpu")
            angle_noise = angle_noise.squeeze(dim=1)
            noise_features[:5, 6] += angle_noise
            # Create a new Data object with noisy features
            noisy_data = Data(x=noise_features, edge_index=edge_index, edge_attr=edge_attr, y=target_positions_tensor)
            dataset.append(noisy_data)

# Shuffle and split dataset
np.random.shuffle(dataset)
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)




class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents):
        super(GATModel, self).__init__()
        self.num_agents = num_agents

        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.norm1 = nn.LayerNorm(hidden_channels * 8)  # Normalize across the feature dimension
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.norm2 = nn.LayerNorm(hidden_channels * 8)
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.norm3 = nn.LayerNorm(hidden_channels * 8)

        # Global pooling layer
        self.pool = global_mean_pool

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16, hidden_channels * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels * 4)
        )
        self.fc2 = nn.Linear(4 * hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GAT layers with LayerNorm
        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(self.norm1(x))
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(self.norm2(x))
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(self.norm3(x))

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, data.batch, data.num_graphs)

        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)

        # Fully connected layers
        combined = self.fc1(combined)
        predicted_positions = self.fc2(combined)

        # Reshape to [batch_size, num_agents, out_channels]
        predicted_positions = predicted_positions.view(data.num_graphs, self.num_agents, -1)
        return predicted_positions

    def extract_agent_embeddings(self, x, batch, batch_size):
        agent_node_indices = []
        for graph_idx in range(batch_size):
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)
        
        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]
        return agent_embeddings



num_agents=5
sample_graph = dataset[0]
in_channels = 7
hidden_dim = 64
output_dim = 3  # Assuming 2D positions

# Initialize the model, loss function, and optimizer


gnn_actor_net = GATModel(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=output_dim, num_agents=num_agents)
        


gnn_actor_net = gnn_actor_net.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(gnn_actor_net.parameters(), lr=0.001)

# Training and validation
num_epochs = 2000
best_val_loss = float('inf')

for epoch in range(num_epochs):
    gnn_actor_net.train()
    total_loss = 0
    for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        # data['graph_tensor'] = data['graph_tensor'].to(device)
        optimizer.zero_grad()
        # print("data:{}".format(data))
        # print("data graph tensor shape:{}".format(data['graph_tensor'].shape))
        # print("data target poses shape:{}".format(data['optimized_target_pos'].shape))
        # Forward pass
        # print("data:{}".format(data))
        # print("data batch:{}".format(data.batch))
        # print("data x:{}".format(data.x))
        data = data.to(device)
        predicted_positions = gnn_actor_net(data)  # Shape: [batch_size, num_agents, 2]
        
        # Get target positions
        target_positions = data.y  # Shape: [batch_size * num_agents, 2]
        target_positions = target_positions.view(-1, num_agents, 3)  # Shape: [batch_size, num_agents, 2]
        
        # graph_tensor = data['graph_tensor']  # Shape: [batch_size, feature_length]
        # target_positions = data['optimized_target_pos']  # Shape: [batch_size, num_agents, 2]

        # Forward pass
        # input_data = TensorDict({'observation': graph_tensor}, batch_size=[graph_tensor.size(0)])
        # predicted_positions = gnn_actor_net(graph_tensor)  # Contains 'action'
        
        # Get target positions
        # target_positions = data['optimized_target_pos'].to(device)  # Shape: [batch_size * num_agents, 2]
        # target_positions = target_positions.view(-1, num_agents, 3)  # Shape: [batch_size, num_agents, 2]
        # Compute loss
        # print("predicted_positions shape:{}".format(predicted_positions))
        # print("target_positions shape:{}".format(target_positions))
        # input("1")
        # print("predicted_positions :{}".format(predicted_positions))
        # print("predicted_positions shape:{}".format(predicted_positions.shape))
        # print("target shape:{}".format(target_positions.shape))
        loss = criterion(predicted_positions, target_positions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Average Training Loss: {avg_loss:.4f}')
    
    # Validation
    gnn_actor_net.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            # graph_tensor = data['graph_tensor']
            # target_positions = data['optimized_target_pos']
            data = data.to(device)
            predicted_positions = gnn_actor_net(data)  # Shape: [batch_size, num_agents, 2]
            # predicted_positions = gnn_actor_net(graph_tensor)
            target_positions = data.y  # Shape: [batch_size * num_agents, 2]
            target_positions = target_positions.view(-1, num_agents, 3)  # Shape: [batch_size, num_agents, 2]
            loss = criterion(predicted_positions, target_positions)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(gnn_actor_net.state_dict(), 'best_imitation_model_tunnel_noise_0_layernorm.pth')
        print('Best model saved.')

print('Training complete.')