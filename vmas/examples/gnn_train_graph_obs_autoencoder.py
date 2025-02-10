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
with open('collected_data_narrow_4.pkl', 'rb') as f:
    collected_data = pickle.load(f)
# print("collected_data:{}".format(collected_data))
# print("collected tensor shape:{}".format(collected_data[0]['graph_tensor'].shape))
# Process the data
dataset = []
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


noise_total_num = 20
for data_point in collected_data:
    graph_data_list = data_point['graph_tensor']  # List of graphs with length batch_dim
    optimized_target_pos = data_point['optimized_target_pos']  # Dict of agent positions with batch_dim

    agent_names = sorted(optimized_target_pos.keys())
    # optimized_target_pos: dict with keys as agent names and values as tensors of shape [batch_dim, 2]

    batch_dim = len(graph_data_list)
    for batch_idx in range(batch_dim):
        # Get graph data for this batch index
        graph_data = graph_data_list[batch_idx]

        # Reconstruct the PyG Data object
        x = torch.tensor(graph_data['x'], dtype=torch.float, device=device)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long, device=device)
        edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float, device=device) if graph_data['edge_attr'] is not None else None

        # Use the full x as features
        features = x  # No separation of categories
        num_nodes = x.shape[0]
        # Create Data object
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

        # Prepare target positions for this batch index
        target_positions = [optimized_target_pos[name][batch_idx] for name in agent_names]  # List of tensors [2]
        target_positions = np.stack(target_positions)  # Shape: [num_agents, 2]
        data.y = torch.tensor(target_positions, dtype=torch.float, device=device)
        target_positions_tensor = torch.tensor(target_positions, dtype=torch.float, device=device)
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
            noise = -0.25 + 0.5*torch.rand((5, 2), device=device)  # shape: [5, 2]
            noise_features[:5, 4:6] += noise
            angle_noise =-math.pi*0.15 +  math.pi*0.3*torch.rand((5,1), device=device)
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


from torch_geometric.nn import GraphUNet

class GraphUNetAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, depth, pool_ratios, out_channels=None):
        """
        Initializes the GraphUNetAutoencoder.

        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden features.
            depth (int): Number of GraphUNet layers.
            pool_ratios (list or float): Pooling ratios for each layer.
            out_channels (int, optional): Number of output features per node.
                                         If None, set to in_channels for reconstruction.
        """
        super(GraphUNetAutoencoder, self).__init__()
        if out_channels is None:
            out_channels = in_channels  # For reconstruction

        # Define the GraphUNet model
        self.graphunet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            sum_res=True,
            act='relu'
        )

    def forward(self, data):
        """
        Forward pass of the autoencoder.

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed node features.
        """
        # GraphUNet automatically handles encoder and decoder
        x_reconstructed = self.graphunet(data.x, data.edge_index, data.batch)
        return x_reconstructed

# Initialize the model
num_agents = 5
in_channels = 7
hidden_channels = 64
out_channels = 7  # To match the input feature size for reconstruction

gnn_actor_net = GraphUNetAutoencoder(
    in_channels=in_channels, 
    hidden_channels=hidden_channels, 
    depth=3, 
    pool_ratios=0.5,
    out_channels = in_channels,

).to(device)
        


gnn_actor_net = gnn_actor_net.to(device)

criterion_reconstruction = nn.MSELoss()
optimizer = optim.Adam(gnn_actor_net.parameters(), lr=0.001)

# Training and validation
num_epochs = 2000
best_val_loss = float('inf')

for epoch in range(num_epochs):
    gnn_actor_net.train()
    total_reconstruction_loss = 0.0
    total_action_loss = 0.0
    for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        # data['graph_tensor'] = data['graph_tensor'].to(device)
        optimizer.zero_grad()
        # print("data:{}".format(data))
        # print("data graph tensor shape:{}".format(data['graph_tensor'].shape))
        # print("data target poses shape:{}".format(data['optimized_target_pos'].shape))
        # Forward pass
        # print("data:{}".format(data))
        # print("data batch:{}".format(data.batch))
        data = data.to(device)
        reconstructed_x = gnn_actor_net(data)
        
        # Get target positions
        
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
        
        # Compute Reconstruction Loss
        
        # Total Loss (weighted sum)
        loss_reconstruction = criterion_reconstruction(reconstructed_x, data.x)
        # total_loss = loss_action + 0.5 * loss_reconstruction  # Adjust the weight as needed
        total_loss = loss_reconstruction
        total_loss.backward()
        optimizer.step()
        total_reconstruction_loss += loss_reconstruction.item()
    
    avg_reconstruction_loss = total_reconstruction_loss / len(train_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Reconstruction Loss: {avg_reconstruction_loss:.4f}')
    
    # Validation
    gnn_actor_net.eval()
    val_loss = 0
    val_action_loss = 0.0
    val_reconstruction_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            # graph_tensor = data['graph_tensor']
            # target_positions = data['optimized_target_pos']
            data = data.to(device)
            reconstructed_x = gnn_actor_net(data)
            # predicted_positions = gnn_actor_net(graph_tensor)
            target_positions = data.y  # Shape: [batch_size * num_agents, 2]
            target_positions = target_positions.view(-1, num_agents, 3)  # Shape: [batch_size, num_agents, 2]
            loss_reconstruction = criterion_reconstruction(reconstructed_x, data.x)
            
            val_reconstruction_loss += loss_reconstruction.item()
    avg_val_action_loss = val_action_loss / len(val_loader)
    avg_val_reconstruction_loss = val_reconstruction_loss / len(val_loader)
    total_val_loss = avg_val_action_loss + 0.5 * avg_val_reconstruction_loss
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Validation Action Loss: {avg_val_action_loss:.4f}, '
          f'Validation Reconstruction Loss: {avg_val_reconstruction_loss:.4f}, '
          f'Total Validation Loss: {total_val_loss:.4f}')
    
    # Save the best model
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(gnn_actor_net.state_dict(), 'best_imitation_model_narrow_noise_autoencoder.pth')
        print('Best model saved.')

print('Training complete.')