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

# Load the collected data
with open('collected_data_random_500_steps_100_env.pkl', 'rb') as f:
    collected_data = pickle.load(f)
# print("collected_data:{}".format(collected_data))
# print("collected tensor shape:{}".format(collected_data[0]['graph_tensor'].shape))
# Process the data
dataset = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for data_point in collected_data:
    graph_tensor = data_point['graph_tensor'].to(device)  # List of graphs with length batch_dim
    optimized_target_pos = data_point['optimized_target_pos']  # Dict of agent positions with batch_dim

    agent_names = sorted(optimized_target_pos.keys())
#     # optimized_target_pos: dict with keys as agent names and values as tensors of shape [batch_dim, 2]

    batch_dim = graph_tensor.size(0)
    for batch_idx in range(batch_dim):
#         # Get graph data for this batch index
        graph_data = {}
        graph_data['graph_tensor'] = graph_tensor[batch_idx, :]

#         # Reconstruct the PyG Data object
#         x = torch.tensor(graph_data['x'], dtype=torch.float)
#         edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
#         edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float) if graph_data['edge_attr'] is not None else None

#         # Use the full x as features
#         features = x  # No separation of categories

#         # Create Data object
#         data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

#         # Prepare target positions for this batch index
        target_positions = [optimized_target_pos[name][batch_idx] for name in agent_names]  # List of tensors [2]
        target_positions = np.stack(target_positions)  # Shape: [num_agents, 2]
        target_positions = torch.tensor(target_positions).to(device)
        graph_data['optimized_target_pos'] = target_positions

#         # Append to dataset
        dataset.append(graph_data)

# Shuffle and split dataset
np.random.shuffle(dataset)
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

# Create DataLoaders
batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
def collate_fn(batch):
    # Batch is a list of data points
    graph_tensors = torch.stack([item['graph_tensor'] for item in batch], dim=0)
    optimized_target_pos = torch.stack([item['optimized_target_pos'] for item in batch], dim=0)
    return {'graph_tensor': graph_tensors, 'optimized_target_pos': optimized_target_pos}

# Create DataLoaders with the custom collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)


def single_tensor_to_data_list(tensor, max_obstacles, num_node_features, num_edge_features, num_agents):
    """
    Reconstructs a list of torch_geometric.Data objects from the serialized tensor.
    Assumes that all graph data is serialized into the 0th agent's feature vector.

    Args:
        tensor (torch.Tensor): Serialized tensor of shape (batch_size, feature_length). Should be on the correct device.
        max_obstacles (int): Maximum number of obstacles per graph.
        num_node_features (int): Number of features per node (excluding category).
        num_edge_features (int): Number of features per edge.
        num_agents (int): Number of agents per graph.

    Returns:
        List[Data]: Reconstructed list of Data objects.
    """
    batch_size, feature_length = tensor.shape
    device = tensor.device
    data_list = []

    # Calculate feature_length per graph
    expected_feature_length = (num_agents + max_obstacles) * num_node_features + max_obstacles * 2 + max_obstacles * num_edge_features
    if feature_length < expected_feature_length:
        raise ValueError(f"Serialized tensor feature_length ({feature_length}) is less than expected ({expected_feature_length}).")

    for batch_idx in range(batch_size):
        # Extract 0th agent's feature vector
        graph_feature = tensor[batch_idx, :]  # [feature_length]

        # Extract node features
        node_features_length = (num_agents + max_obstacles) * num_node_features
        flattened_node_features = graph_feature[:node_features_length]
        node_features = flattened_node_features.view(num_agents + max_obstacles, num_node_features)

        # Extract edge indices
        edge_index_length = max_obstacles * 2  # Each edge has two indices
        flattened_edge_index = graph_feature[node_features_length:node_features_length + edge_index_length]
        edge_index = flattened_edge_index.view(max_obstacles, 2).long()

        # Extract edge attributes
        edge_attr_start = node_features_length + edge_index_length
        flattened_edge_attr = graph_feature[edge_attr_start:edge_attr_start + max_obstacles * num_edge_features]
        edge_attr = flattened_edge_attr.view(max_obstacles, num_edge_features)

        # Reconstruct edge_index and edge_attr by removing padding
        # Create a mask to identify valid edges
        max_nodes = num_agents + max_obstacles
        valid_mask = (edge_index[:, 0] < max_nodes) & (edge_index[:, 1] < max_nodes)

        # Apply mask to edge_index and edge_attr
        edge_index = edge_index[valid_mask].t().contiguous()
        edge_attr = edge_attr[valid_mask]

        # Create Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

    return data_list


# def single_tensor_to_data_list(tensor, max_obstacles, num_node_features, num_edge_features, num_agents):
#     """
#     Reconstructs a list of torch_geometric.Data objects from the serialized tensor.
#     Assumes that all graph data is serialized into the 0th agent's feature vector.
    
#     Args:
#         tensor (torch.Tensor): Serialized tensor of shape (batch_size, feature_length).
#         max_obstacles (int): Maximum number of obstacles per graph.
#         num_node_features (int): Number of features per node (excluding category).
#         num_edge_features (int): Number of features per edge.
#         num_agents (int): Number of agents per graph.
    
#     Returns:
#         List[Data]: Reconstructed list of Data objects.
#     """
#     batch_size, feature_length = tensor.shape
#     data_list = []
    
#     # Calculate feature_length per graph
#     expected_feature_length = (num_agents + max_obstacles) * num_node_features + max_obstacles * 2 + max_obstacles * num_edge_features
#     if feature_length < expected_feature_length:
#         raise ValueError(f"Serialized tensor feature_length ({feature_length}) is less than expected ({expected_feature_length}).")
    
#     for batch_idx in range(batch_size):
#         # Extract 0th agent's feature vector
#         graph_feature = tensor[batch_idx, :]  # [feature_length]
        
#         # Extract node features
#         node_features_length = (num_agents + max_obstacles) * num_node_features
#         flattened_node_features = graph_feature[:node_features_length]  # [num_agents + max_obstacles, num_node_features]
#         node_features = flattened_node_features.view(num_agents + max_obstacles, num_node_features)
        
#         # Extract edge indices
#         edge_index_length = max_obstacles * 2  # Each edge has two indices
#         flattened_edge_index = graph_feature[node_features_length:node_features_length + edge_index_length]
#         edge_index = flattened_edge_index.view(max_obstacles, 2).long()  # [max_obstacles, 2]
        
#         # Extract edge attributes
#         edge_attr_start = node_features_length + edge_index_length
#         flattened_edge_attr = graph_feature[edge_attr_start:edge_attr_start + max_obstacles * num_edge_features]
#         edge_attr = flattened_edge_attr.view(max_obstacles, num_edge_features)  # [max_obstacles, num_edge_features]
        
#         # Reconstruct edge_index and edge_attr by removing padding
#         valid_edges = []
#         valid_edge_attrs = []
#         for edge_idx in range(edge_index.size(0)):
#             src, dst = edge_index[edge_idx]
#             if src < (num_agents + max_obstacles) and dst < (num_agents + max_obstacles):
#                 # Assuming padding was done with zeros beyond valid nodes
#                 # If an edge connects to node 0 (agent), it might be valid, so you might need another padding value
#                 # Here, we consider all edges as valid except those connecting to padded nodes
#                 # Modify this condition based on your padding strategy
#                 if src >= num_agents and src >= num_agents + max_obstacles - (num_agents + max_obstacles):
#                     continue
#                 if dst >= num_agents and dst >= num_agents + max_obstacles - (num_agents + max_obstacles):
#                     continue
#                 valid_edges.append([src, dst])
#                 valid_edge_attrs.append(edge_attr[edge_idx])
        
#         if valid_edges:
#             edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()  # [2, num_valid_edges]
#             edge_attr = torch.stack(valid_edge_attrs)  # [num_valid_edges, num_edge_features]
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = torch.empty((0, num_edge_features), dtype=torch.float)
        
#         # Create Data object
#         data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).cuda()
#         data_list.append(data)
    
#     return data_list

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents, max_obstacles, num_node_features, num_edge_features):
        super(GATModel, self).__init__()
        self.num_agents = num_agents
        self.max_obstacles = max_obstacles
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

        # Global pooling layer
        self.pool = global_mean_pool

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16, hidden_channels*4),
            nn.ReLU()
            # nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(4*hidden_channels, out_channels)


    def forward(self, tensor):
        """
        Args:
            data (Batch): Batched PyG Data object containing multiple graphs.

        Returns:
            predicted_positions (torch.Tensor): Predicted positions for all agents in all graphs.
                                               Shape: [batch_size, num_agents, out_channels]
        """
        batch_size, feature_length = tensor.shape
        device = tensor.device
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # torch.set_printoptions(threshold=torch.inf)
        data_list = single_tensor_to_data_list(
            tensor, 
            max_obstacles=self.max_obstacles, 
            num_node_features=self.num_node_features, 
            num_edge_features=self.num_edge_features,
            num_agents=self.num_agents
        )
        batch = Batch.from_data_list(data_list)
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        # print("x after conv1:{}".format(x))
        # print("after4 conv1 x shape:{}".format(x.shape))
        # input("0, 1")
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)
        # print("x:{}".format(x))
        # Global graph embedding
        graph_embedding = self.pool(x, batch.batch)  # Shape: [batch_size, hidden_channels]

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, batch.batch, batch.num_graphs)
        # print("agent_embedding:{}".format(agent_embeddings))
        # input("1")
        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)  # Shape: [batch_size*num_agents, hidden_channels]

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)  # Shape: [batch_size*num_agents, 2*hidden_channels]
        # print("combined shape:{}".format(combined.shape))
        # Fully connected layers
        combined = self.fc1(combined)
        combined = torch.relu(combined)
        predicted_positions = self.fc2(combined)  # Shape: [batch_size*num_agents, out_channels]
        # print("202 predicted positions:{}".format(predicted_positions))
        # Reshape to [batch_size, num_agents, out_channels]
        predicted_positions = predicted_positions.view(batch.num_graphs, self.num_agents, -1)
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
in_channels = 3
hidden_dim = 64
output_dim = 6  # Assuming 2D positions

# Initialize the model, loss function, and optimizer


gnn_actor_net =  torch.nn.Sequential(
        GATModel(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=output_dim, num_agents=num_agents, max_obstacles= 100, num_node_features=3, num_edge_features=1),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    ).to(device)

gnn_policy_module = TensorDictModule(
    gnn_actor_net,
    in_keys=["observation"],
    out_keys=["loc", "scale" ],
    # out_keys=[env.action_key],
)
# print(env.unbatched_action_spec[("agents", "action")].space.low / 2)
# print(env.unbatched_action_spec[("agents", "action")].space.high /2 )

gnn_policy = ProbabilisticActor(
    module=gnn_policy_module,
    in_keys=["loc", "scale"],
    out_keys=['action'],
    distribution_class=TanhNormal,
    # distribution_class=Normal,
    return_log_prob=True,
).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(gnn_policy.parameters(), lr=0.001)

# Training and validation
num_epochs = 2000
best_val_loss = float('inf')

for epoch in range(num_epochs):
    gnn_policy.train()
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
        graph_tensor = data['graph_tensor']  # Shape: [batch_size, feature_length]
        target_positions = data['optimized_target_pos']  # Shape: [batch_size, num_agents, 2]

        # Forward pass
        input_data = TensorDict({'observation': graph_tensor}, batch_size=[graph_tensor.size(0)])
        predicted_positions = gnn_policy(input_data)  # Contains 'action'
        
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
        loss = criterion(predicted_positions['action'], target_positions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Average Training Loss: {avg_loss:.4f}')
    
    # Validation
    gnn_policy.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            graph_tensor = data['graph_tensor']
            target_positions = data['optimized_target_pos']

            input_data = TensorDict({'observation': graph_tensor}, batch_size=[graph_tensor.size(0)])
            predicted_positions = gnn_policy(input_data)

            loss = criterion(predicted_positions['action'], target_positions)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(gnn_policy.state_dict(), 'best_gnn_model.pth')
        print('Best model saved.')

print('Training complete.')