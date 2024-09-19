import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import numpy as np

# Load the collected data
with open('collected_data.pkl', 'rb') as f:
    collected_data = pickle.load(f)

# Function to process individual data point into PyG Data object
def process_data_point(data_point):
    graph_data_list = data_point['graph_list']
    optimized_target_pos = data_point['optimized_target_pos']  # dict of agent positions

    # Assuming one graph per data point
    graph_data = graph_data_list[0]

    # Reconstruct the PyG Data object
    x = torch.tensor(graph_data['x'], dtype=torch.float)
    edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float) if graph_data['edge_attr'] is not None else None

    # Separate features and categories
    features = x[:, :-1]  # Node features
    categories = x[:, -1].long()  # Node categories (0: agent, 1: obstacle)

    # Create Data object
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)
    data.categories = categories  # Save categories in data

    # Prepare target positions (ensure consistent ordering)
    agent_names = sorted(optimized_target_pos.keys())
    target_positions = [optimized_target_pos[name] for name in agent_names]
    target_positions = np.stack(target_positions)  # Shape: [num_agents, 2]
    data.y = torch.tensor(target_positions, dtype=torch.float)

    return data

# Process all data points into Data objects
data_list = [process_data_point(dp) for dp in collected_data]

# Define sequence length
sequence_length = 5

# Create sequences
def create_sequences(data_list, sequence_length):
    sequences = []
    num_sequences = len(data_list) - sequence_length + 1
    for i in range(num_sequences):
        sequence = data_list[i:i+sequence_length]
        sequences.append(sequence)
    return sequences

sequences = create_sequences(data_list, sequence_length)

# Custom Dataset class
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

# Custom collate function
def collate_fn(batch):
    # batch is a list of sequences, each sequence is a list of Data objects
    batch_size = len(batch)
    sequence_length = len(batch[0])
    
    all_data = []
    for t in range(sequence_length):
        data_list = [batch[b][t] for b in range(batch_size)]
        # Use Batch.from_data_list to batch the data at each time step
        batch_data = Batch.from_data_list(data_list)
        all_data.append(batch_data)
    return all_data

# Shuffle and split dataset
np.random.shuffle(sequences)
train_size = int(0.8 * len(sequences))
train_sequences = sequences[:train_size]
val_sequences = sequences[train_size:]

# Create Dataset objects
train_dataset = SequenceDataset(train_sequences)
val_dataset = SequenceDataset(val_sequences)

# Create DataLoaders
batch_size = 8  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Define the GNN-GRU Model
class GNN_GRU_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents):
        super(GNN_GRU_Model, self).__init__()
        self.num_agents = num_agents
        self.hidden_channels = hidden_channels
        
        # GNN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # GRU layer
        self.gru = nn.GRU(input_size=hidden_channels, hidden_size=hidden_channels, batch_first=True)
        
        # Output layer
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data_sequence):
        # data_sequence: list of Batch objects (length = sequence_length)
        batch_size = data_sequence[0].num_graphs
        sequence_length = len(data_sequence)
        
        # Store agent embeddings at each time step
        agent_embeddings_sequence = []
        
        for t in range(sequence_length):
            data = data_sequence[t]
            x, edge_index, batch = data.x, data.edge_index, data.batch
            categories = data.categories  # Node categories
            
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            x = torch.relu(x)
            
            # Extract agent nodes (category == 0)
            agent_mask = (categories == 0)
            agent_embeddings = x[agent_mask]
            agent_batch = batch[agent_mask]
            
            # Ensure consistent ordering
            sorted_indices = torch.argsort(agent_batch * self.num_agents + torch.arange(self.num_agents, device=agent_batch.device))
            agent_embeddings = agent_embeddings[sorted_indices]
            
            # Reshape agent_embeddings to [batch_size, num_agents, hidden_channels]
            agent_embeddings = agent_embeddings.view(batch_size, self.num_agents, self.hidden_channels)
            agent_embeddings_sequence.append(agent_embeddings)
        
        # Stack embeddings over time: [batch_size, sequence_length, num_agents, hidden_channels]
        agent_embeddings_sequence = torch.stack(agent_embeddings_sequence, dim=1)
        
        # Reshape for GRU: combine batch_size and num_agents
        batch_agents = batch_size * self.num_agents
        embeddings_for_gru = agent_embeddings_sequence.view(batch_agents, sequence_length, self.hidden_channels)
        
        # Pass through GRU
        gru_output, _ = self.gru(embeddings_for_gru)  # Output shape: [batch_agents, sequence_length, hidden_channels]
        
        # Take the output at the last time step
        final_output = gru_output[:, -1, :]  # Shape: [batch_agents, hidden_channels]
        
        # Predict positions
        predicted_positions = self.lin(final_output)  # Shape: [batch_agents, out_channels]
        
        # Reshape back to [batch_size, num_agents, out_channels]
        predicted_positions = predicted_positions.view(batch_size, self.num_agents, -1)
        
        return predicted_positions

# Set up model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent_names = sorted(collected_data[0]['optimized_target_pos'].keys())
num_agents = len(agent_names)
model = GNN_GRU_Model(in_channels=2, hidden_channels=64, out_channels=2, num_agents=num_agents).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data_sequence in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        # Move data to device
        data_sequence = [data.to(device) for data in data_sequence]
        optimizer.zero_grad()
        
        # Forward pass
        predicted_positions = model(data_sequence)  # Shape: [batch_size, num_agents, 2]
        
        # Get target positions from the last time step
        target_data = data_sequence[-1]  # Data object at last time step
        target_positions = target_data.y.view(-1, num_agents, 2).to(device)  # Shape: [batch_size, num_agents, 2]
        
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
        for data_sequence in val_loader:
            data_sequence = [data.to(device) for data in data_sequence]
            predicted_positions = model(data_sequence)
            target_data = data_sequence[-1]
            target_positions = target_data.y.view(-1, num_agents, 2).to(device)
            loss = criterion(predicted_positions, target_positions)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_gnn_gru_model.pth')
        print('Best model saved.')

print('Training complete.')
