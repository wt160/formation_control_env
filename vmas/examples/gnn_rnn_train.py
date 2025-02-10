import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm
import pickle
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

# Load the collected data
with open('collected_data_narrow_400_steps_5_env.pkl', 'rb') as f:
    collected_data = pickle.load(f)

# Process the data into sequences
sequence_length = 10  # Adjust based on your data
sequences = []

from collections import defaultdict

# Initialize a dictionary to hold data per environment
env_data = defaultdict(lambda: {'graph_tensor': [], 'optimized_target_pos': defaultdict(list)})

# Iterate over each time step
for data_point in collected_data:
    graph_data_list = data_point['graph_tensor']  # List of graphs for each environment at this time step
    optimized_target_pos = data_point['optimized_target_pos']  # Dict of agent positions
    batch_dim = len(graph_data_list)
    for env_idx in range(batch_dim):
        # Append graph data for this environment
        env_data[env_idx]['graph_tensor'].append(graph_data_list[env_idx])
        # Append target positions for each agent in this environment
        for agent_name, positions in optimized_target_pos.items():
            env_data[env_idx]['optimized_target_pos'][agent_name].append(positions[env_idx])


for env_idx, data in env_data.items():
    graph_data_list = data['graph_tensor']
    optimized_target_pos = data['optimized_target_pos']
    
    num_time_steps = len(graph_data_list)
    print("num_time_steps:{}".format(num_time_steps))
    for t in range(0, num_time_steps - sequence_length + 1, 5):
        sequence_graphs = []
        
        for j in range(sequence_length):
            time_idx = t + j
            graph_data = graph_data_list[time_idx]
            
            # Reconstruct the PyG Data object
            x = torch.tensor(graph_data['x'], dtype=torch.float)
            edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float) if graph_data['edge_attr'] is not None else None
            
            # Use the full x as features
            features = x
            
            # Create Data object
            data_obj = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)
            
            # Prepare target positions for this time step
            target_positions = [
                optimized_target_pos[agent][time_idx] for agent in sorted(optimized_target_pos.keys())
            ]  # List of tensors [3]
            target_positions = torch.tensor(target_positions, dtype=torch.float)  # Shape: [num_agents, 3]
            data_obj.y = target_positions
            # print("x:{}".format(data_obj.x))
            # print("y:{}".format(data_obj.y))
            # input("1")
            sequence_graphs.append(data_obj)
        
        sequences.append(sequence_graphs)








# for data_point in collected_data:
#     graph_data_list = data_point['graph_tensor']  # List of graphs with length batch_dim
#     optimized_target_pos = data_point['optimized_target_pos']  # Dict of agent positions with batch_dim
    

#     agent_names = sorted(optimized_target_pos.keys())
#     batch_dim = len(graph_data_list)
#     print("batch_dim:{}".format(batch_dim))
#     # Create sequences
#     for i in range(0, batch_dim - sequence_length + 1):
#         sequence_graphs = []
#         sequence_targets = []
#         for j in range(sequence_length):
#             batch_idx = i + j
#             # Get graph data for this batch index
#             graph_data = graph_data_list[batch_idx]

#             # Reconstruct the PyG Data object
#             x = torch.tensor(graph_data['x'], dtype=torch.float)
#             edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
#             edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float) if graph_data['edge_attr'] is not None else None

#             # Use the full x as features
#             features = x

#             # Create Data object
#             data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

#             # Prepare target positions for this batch index
#             target_positions = [optimized_target_pos[name][batch_idx] for name in agent_names]  # List of tensors [3]
#             # print("target_positions:{}".format(target_positions))
#             target_positions = np.stack(target_positions)  # Shape: [num_agents, 3]
#             data.y = torch.tensor(target_positions, dtype=torch.float)
#             # print("data:{}".format(data.y))
#             # input("1")
#             sequence_graphs.append(data)

#         sequences.append(sequence_graphs)

# Shuffle and split sequences
np.random.shuffle(sequences)
train_size = int(0.8 * len(sequences))
train_sequences = sequences[:train_size]
val_sequences = sequences[train_size:]

# Custom collate function for DataLoader
def collate_sequences(batch):
    return batch  # batch is already a list of sequences

# Create DataLoaders
batch_size = 8  # Adjust based on your GPU memory
train_loader = DataLoader(train_sequences, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
val_loader = DataLoader(val_sequences, batch_size=batch_size, collate_fn=collate_sequences)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the modified GATModel with GRU
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GNNBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNBackbone, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True,
                             edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True,
                             edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True,
                             edge_dim=1, fill_value='mean')

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1) if edge_attr is not None else None)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)
        return x

class ILModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents, num_gru_layers=1):
        super(ILModel, self).__init__()
        self.num_agents = num_agents
        self.hidden_size = hidden_channels * 2  # Adjust as needed
        self.num_gru_layers = num_gru_layers

        # Shared GNN backbone
        self.gnn = GNNBackbone(in_channels, hidden_channels)

        # GRU for temporal dependencies
        self.gru = nn.GRU(input_size=hidden_channels * 8 * 2,  # Agent + Graph embeddings
                          hidden_size=self.hidden_size,
                          num_layers=self.num_gru_layers,
                          batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_channels),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def extract_agent_embeddings(self, x, batch, batch_size):
        """
        Extract embeddings for each agent from the GNN output.

        Args:
            x (torch.Tensor): Node embeddings after GNN layers. Shape: [num_nodes, hidden_channels * 8]
            batch (torch.Tensor): Batch assignments for each node. Shape: [num_nodes]
            batch_size (int): Number of graphs in the batch.

        Returns:
            agent_embeddings (torch.Tensor): Embeddings for all agents across the batch.
                                           Shape: [batch_size * num_agents, hidden_channels * 8]
        """
        agent_node_indices = []
        for graph_idx in range(batch_size):
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            agent_nodes = node_indices[:self.num_agents]  # Assuming first num_agents nodes are agents
            agent_node_indices.append(agent_nodes)

        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]
        return agent_embeddings

    def forward(self, batched_sequences):
        """
        Args:
            batched_sequences (list of lists of Data objects): Each sublist represents a sequence for one batch item.

        Returns:
            predicted_positions (torch.Tensor): Predicted positions for all agents in all sequences.
                                                Shape: [batch_size, sequence_length, num_agents, out_channels]
        """
        batch_size = len(batched_sequences)
        sequence_length = len(batched_sequences[0])

        # Initialize hidden states for all agents in the batch
        # Shape: [num_gru_layers, batch_size * num_agents, hidden_size]
        hidden_states = torch.zeros(
            self.num_gru_layers,
            batch_size * self.num_agents,
            self.hidden_size
        ).to(next(self.parameters()).device)

        # Prepare storage for predictions
        all_predictions = []

        for t in range(sequence_length):
            # Extract the t-th Data object from each sequence in the batch
            data_list = [sequence[t] for sequence in batched_sequences]
            batch = Batch.from_data_list(data_list).to(next(self.parameters()).device)
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

            # Apply GNN backbone
            x = self.gnn(x, edge_index, edge_attr)

            # Global graph embedding
            graph_embedding = global_mean_pool(x, batch.batch)  # Shape: [batch_size, hidden_channels * 8]

            # Extract agent node embeddings
            agent_embeddings = self.extract_agent_embeddings(x, batch.batch, batch_size)  # [batch_size * num_agents, hidden_channels * 8]

            # Concatenate agent embeddings with graph embeddings
            graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)  # [batch_size * num_agents, hidden_channels * 8]
            combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)  # [batch_size * num_agents, hidden_channels * 16]

            # Reshape to [batch_size * num_agents, 1, hidden_channels * 16] for GRU
            combined = combined.view(batch_size * self.num_agents, 1, -1)  # [batch_size * num_agents, 1, hidden_channels * 16]

            # Pass through GRU
            gru_out, hidden_states = self.gru(combined, hidden_states)  # gru_out shape: [batch_size * num_agents, 1, hidden_size]

            # Reshape gru_out to [batch_size, num_agents, hidden_size]
            gru_out = gru_out.view(batch_size, self.num_agents, self.hidden_size)  # [batch_size, num_agents, hidden_size]

            # Pass through fully connected layers
            fc1_out = self.fc1(gru_out)  # [batch_size, num_agents, hidden_channels]
            predicted_positions_t = self.fc2(fc1_out)  # Shape: [batch_size, num_agents, out_channels]

            # Append predictions for this time step
            all_predictions.append(predicted_positions_t)

        # Stack predictions over time: [batch_size, sequence_length, num_agents, out_channels]
        predicted_positions = torch.stack(all_predictions, dim=1)

        return predicted_positions

# Initialize the model, loss function, and optimizer
num_agents = 5
in_channels = 4  # Adjust based on your node feature dimension
hidden_dim = 64
output_dim = 3  # Assuming 3D positions
# Initialize GNN backbone
gnn_backbone = GNNBackbone(in_channels=in_channels, hidden_channels=hidden_dim)
gnn_backbone = gnn_backbone.to(device)

il_model = ILModel(
    in_channels=in_channels,
    hidden_channels=hidden_dim,
    out_channels=output_dim,
    num_agents=num_agents
)
il_model.gnn = gnn_backbone  # Assign shared GNN backbone
il_model = il_model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(il_model.parameters(), lr=0.001)

# Training and validation
num_epochs = 100  # Adjust as needed
best_val_loss = float('inf')

for epoch in range(num_epochs):
    il_model.train()
    total_loss = 0

    for batched_sequences in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_positions = il_model(batched_sequences)  # No hidden_state input

        # Prepare target positions
        target_positions = torch.stack([sequence.y for sequence in batched_sequences], dim=0).to(device)  # [batch_size, sequence_length, num_agents, out_channels]
        predicted_positions = predicted_positions.reshape(predicted_positions.shape[0], predicted_positions.shape[1]*predicted_positions.shape[2], predicted_positions.shape[3])
        # print("predicted_positions shape:{}".format(predicted_positions))
        # print("target_positions shape:{}".format(target_positions))
        # Compute loss
        loss = criterion(predicted_positions, target_positions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation step
    il_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batched_sequences in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
            
            predicted_positions = il_model(batched_sequences)

            # Prepare target positions
            target_positions = torch.stack([sequence.y for sequence in batched_sequences], dim=0).to(device)
            predicted_positions = predicted_positions.reshape(predicted_positions.shape[0], predicted_positions.shape[1]*predicted_positions.shape[2], predicted_positions.shape[3])
            # Compute loss
            
            loss = criterion(predicted_positions, target_positions)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # Print the losses
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the model checkpoint if it has the best validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(il_model.state_dict(), 'best_il_model.pth')
        print('Best IL model saved.')

print('IL Training complete.')
