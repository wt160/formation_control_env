import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv, global_mean_pool
from tqdm import tqdm
import pickle
import numpy as np

# Load the collected data
with open('collected_data_10.pkl', 'rb') as f:
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
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)



# class GNNModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_agents):
#         super(GNNModel, self).__init__()
#         self.num_agents = num_agents
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.lin = nn.Linear(hidden_channels, out_channels)
    
#     def forward(self, data):
#         """
#         Args:
#             data (Batch): Batched PyG Data object containing multiple graphs.

#         Returns:
#             predicted_positions (torch.Tensor): Predicted positions for all agents in all graphs.
#                                                Shape: [batch_size, num_agents, out_channels]
#         """
#         x, edge_index = data.x, data.edge_index

#         # GNN Layers
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv3(x, edge_index)
#         x = torch.relu(x)

#         # Extract agent node embeddings
#         # Assumption: In each graph, the first `num_agents` nodes correspond to agents
#         batch_size = data.num_graphs
#         agent_embeddings = self.extract_agent_embeddings(x, data.batch, batch_size)

#         # Predict positions
#         predicted_positions = self.lin(agent_embeddings)  # Shape: [batch_size * num_agents, out_channels]

#         # Reshape to [batch_size, num_agents, out_channels]
#         predicted_positions = predicted_positions.view(batch_size, self.num_agents, -1)
#         return predicted_positions

#     def extract_agent_embeddings(self, x, batch, batch_size):
#         """
#         Extracts agent node embeddings from the batched node features.

#         Args:
#             x (torch.Tensor): Node features after GNN layers. Shape: [total_nodes, hidden_channels]
#             batch (torch.Tensor): Batch vector, which assigns each node to a specific graph. Shape: [total_nodes]
#             batch_size (int): Number of graphs in the batch.

#         Returns:
#             agent_embeddings (torch.Tensor): Agent embeddings for all graphs. Shape: [batch_size * num_agents, hidden_channels]
#         """
#         agent_node_indices = []
#         for graph_idx in range(batch_size):
#             # Find node indices for the current graph
#             node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
#             # Take the first `num_agents` nodes as agent nodes
#             agent_nodes = node_indices[:self.num_agents]
#             agent_node_indices.append(agent_nodes)
        
#         # Concatenate all agent node indices
#         agent_node_indices = torch.cat(agent_node_indices, dim=0)
#         agent_embeddings = x[agent_node_indices]  # Shape: [batch_size * num_agents, hidden_channels]
#         return agent_embeddings



class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents):
        super(GINModel, self).__init__()
        self.num_agents = num_agents

        # GIN layers
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)

        nn3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv3 = GINConv(nn3)

        # Global pooling layer
        self.pool = global_mean_pool

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        """
        Args:
            data (Batch): Batched PyG Data object containing multiple graphs.

        Returns:
            predicted_positions (torch.Tensor): Predicted positions for all agents in all graphs.
                                               Shape: [batch_size, num_agents, out_channels]
        """
        x, edge_index = data.x, data.edge_index

        # GIN Layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  # Shape: [batch_size, hidden_channels]

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, data.batch, data.num_graphs)

        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)  # Shape: [batch_size*num_agents, hidden_channels]

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)  # Shape: [batch_size*num_agents, 2*hidden_channels]

        # Fully connected layers
        combined = self.fc1(combined)
        combined = torch.relu(combined)
        predicted_positions = self.fc2(combined)  # Shape: [batch_size*num_agents, out_channels]

        # Reshape to [batch_size, num_agents, out_channels]
        predicted_positions = predicted_positions.view(data.num_graphs, self.num_agents, -1)
        return predicted_positions

    def extract_agent_embeddings(self, x, batch, batch_size):
        """
        Extracts agent node embeddings from the batched node features.

        Args:
            x (torch.Tensor): Node features after GIN layers. Shape: [total_nodes, hidden_channels]
            batch (torch.Tensor): Batch vector, which assigns each node to a specific graph. Shape: [total_nodes]
            batch_size (int): Number of graphs in the batch.

        Returns:
            agent_embeddings (torch.Tensor): Agent embeddings for all graphs. Shape: [batch_size * num_agents, hidden_channels]
        """
        agent_node_indices = []
        for graph_idx in range(batch_size):
            # Find node indices for the current graph
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            # Take the first `num_agents` nodes as agent nodes
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)
        
        # Concatenate all agent node indices
        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]  # Shape: [batch_size * num_agents, hidden_channels]
        return agent_embeddings
num_agents=5
sample_graph = dataset[0]
in_channels = sample_graph.x.shape[1]
hidden_channels = 64
out_channels = 2  # Assuming 2D positions

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_agents=num_agents).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation
num_epochs = 2000
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