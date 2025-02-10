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


def assign_env_label(data_filename, output_filename, label):
    with open(data_filename, 'rb') as f:
        collected_data = pickle.load(f)
    # print("collected_data:{}".format(collected_data))
    # print("collected tensor shape:{}".format(collected_data[0]['graph_tensor'].shape))
    # Process the data
    dataset = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            x = torch.tensor(graph_data['x'], dtype=torch.float)
            edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float) if graph_data['edge_attr'] is not None else None

            # Use the full x as features
            features = x  # No separation of categories
            num_nodes = x.shape[0]
            if num_nodes > 5:
                expert_class = label  # Second expert
            else:
                expert_class = 0  # First expert

            # Create Data object
            data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

            # Assign the class label (integer)
            data.y = torch.tensor(expert_class, dtype=torch.long, device=device)  # Class indices: 0, 1, 2, 3


            
            # Append to dataset
            dataset.append(data)
    with open(output_filename, 'wb') as f:
        pickle.dump(dataset, f)
# Shuffle and split dataset
# np.random.shuffle(dataset)
# train_size = int(0.8 * len(dataset))
# train_dataset = dataset[:train_size]
# val_dataset = dataset[train_size:]

# # Create DataLoaders
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("clutter")
assign_env_label("collected_data_clutter_0_no_action.pkl", "env_label_clutter.pkl", 1)
print("narrow")
assign_env_label("collected_data_narrow_0_no_action.pkl", "env_label_narrow.pkl", 2)
print("tunnel")
assign_env_label("collected_data_tunnel_0_no_action.pkl", "env_label_tunnel.pkl", 3)
print("door")
assign_env_label("collected_data_door_0_no_action.pkl", "env_label_door.pkl", 4)


print('Training complete.')