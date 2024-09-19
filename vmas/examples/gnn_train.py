import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import pickle
import numpy as np

# Load the collected data
with open('collected_data.pkl', 'rb') as f:
    collected_data = pickle.load(f)

# Process the data
dataset = []

for data_point in collected_data:
    graph_data_list = data_point['graph_list']  # List of graphs with length batch_dim
    optimized_target_pos = data_point['optimized_target_pos']  # Dict of agent positions with batch_dim

    agent_names = sorted(optimized_target_pos.keys())
    # optimized_target_pos: dict with keys as agent names and values as tensors of shape [batch_dim, 2]

    batch_dim = len(graph_data_list)
    for batch_idx in range(batch_dim):
        # Get graph data for this batch index
        graph_data = graph_data_list[batch_idx]

        # Reconstruct the PyG Data object
        x = torch.tensor(graph_data['x'], dtype=torch.float)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float) if graph_data['edge_attr'] is not None else None

        # Use the full x as features
        features = x  # No separation of categories

        # Create Data object
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

        # Prepare target positions for this batch index
        target_positions = [optimized_target_pos[name][batch_idx] for name in agent_names]  # List of tensors [2]
        target_positions = np.stack(target_positions)  # Shape: [num_agents, 2]
        data.y = torch.tensor(target_positions, dtype=torch.float)

        # Append to dataset
        dataset.append(data)

# Shuffle and split dataset
np.random.shuffle(dataset)
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the GNN Model
# class GNNModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GNNModel, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.lin = nn.Linear(hidden_channels, out_channels)
    
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         categories = data.categories  # Node categories
        
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv3(x, edge_index)
#         x = torch.relu(x)
        
#         # Extract agent nodes (category == 0)
#         agent_mask = (categories == 0)
#         agent_embeddings = x[agent_mask]
        
#         # Predict positions
#         predicted_positions = self.lin(agent_embeddings)
        
#         return predicted_positions

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents):
        super(GNNModel, self).__init__()
        self.num_agents = num_agents
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        # Since we have no categories, we need a way to extract agent node embeddings
        # Let's assume that agent nodes are the first `num_agents` nodes
        agent_embeddings = x[:self.num_agents]

        # Predict positions
        predicted_positions = self.lin(agent_embeddings)

        return predicted_positions

num_agents = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(in_channels=graph_data['x'].shape[1], hidden_channels=64, out_channels=2, num_agents=num_agents).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        predicted_positions = model(data)  # Shape: [batch_size, num_agents, 2]
        
        # Get target positions
        target_positions = data.y.to(device)  # Shape: [batch_size * num_agents, 2]
        target_positions = target_positions.view(-1, num_agents, 2)  # Shape: [batch_size, num_agents, 2]
        
        # Compute loss
        loss = criterion(predicted_positions, target_positions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Average Training Loss: {avg_loss:.4f}')
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            predicted_positions = model(data)
            target_positions = data.y.to(device)
            target_positions = target_positions.view(-1, num_agents, 2)
            loss = criterion(predicted_positions, target_positions)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_gnn_model.pth')
        print('Best model saved.')

print('Training complete.')