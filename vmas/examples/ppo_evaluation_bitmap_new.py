import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# Import your VMAS environment
from vmas import make_env
import torch.nn.functional as F
import argparse # New import
import os # New import for creating directories
import sys
import swanlab
# train_env_type = sys.argv[1]
# policy_filename = sys.argv[2]
# output_policy_filename = sys.argv[3]
# steps_per_epoch = int(sys.argv[4])
# Set device



# device = sys.argv[5]
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def set_seed(seed):
    # Set the seed for Python random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    
    # For CUDA (if you are using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # In case of multiple GPUs
    
    # For deterministic operations in PyTorch (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage:
# seed = 0  # Set the seed you want
# set_seed(seed)
# Define your environment wrapper
class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents, env_type=None, is_evaluation_mode=False, is_imitation=False, working_mode="imitation", evaluation_index=0, has_laser = True, train_map_directory = "train_maps_0_clutter",use_leader_laser_only = False):
        self.env = make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            dict_spaces=False,  # Use list-based observations for simplicity
            n_agents=n_agents,
            wrapper=None,
            seed=None,
            env_type = env_type,
            is_evaluation_mode=is_evaluation_mode,
            is_imitation = is_imitation,
            working_mode=working_mode,
            evaluation_index = evaluation_index,
            has_laser = has_laser,
            train_map_directory=train_map_directory,
        )
        self.max_connection_distance = 1.8
        self.device = device
        self.n_agents = n_agents
        self.num_envs = num_envs
        self.has_laser = has_laser
        self.use_leader_laser_only = use_leader_laser_only


    def reset(self):
        obs = self.env.reset()
        # print("obs length:{}".format(len(obs[0])))
        # obs = obs[0]
        # obs is a list of observations per agent
        # Stack observations to shape [num_envs, n_agents, obs_dim]
        # obs = torch.stack(obs, dim=1).to(self.device)
        return self.get_obs()

    def get_forward_env_type(self):
        env_type = self.env.get_forward_env_type()
        return env_type

    def get_obs(self):
        obs = self.env.get_obs()
        # obs = obs[
        batch_size = obs[0]['nominal_pos'].shape[0]
        dones = torch.zeros(batch_size, device=self.device)
        obs = self.get_graph_from_obs(obs, dones)
        return obs

    def get_leader_paths(self):
        leader_paths = self.env.get_leader_paths()
        return leader_paths
    
    def set_action_mean(self, actions):
        self.env.set_action_mean(actions)
        

    def get_graph_from_obs(self, obs, dones):
        batch_size = obs[0]['nominal_pos'].shape[0]
        n_agents = len(obs)
        
        # Pre-allocate tensors for each agent's observations
        laser_obs = []
        forward_opening_obs = []
        relative_pos_obs = []
        nominal_pos_obs_diff = []
        leader_vel = []
        leader_ang_vel = []
        nominal_pos_obs = []
        last_action_obs = []
        # Collect observations by type
        for agent_index in range(n_agents):
            if self.has_laser == True:
                if self.use_leader_laser_only == True:
                    print("use leader laser only")
                    laser_obs.append(obs[0]['laser'])
                    forward_opening_obs.append(obs[0]['forward_opening'])

                else:
                    laser_obs.append(obs[agent_index]['laser'])
                    forward_opening_obs.append(obs[0]['forward_opening'])


            last_action_obs.append(obs[agent_index]['last_action_u'])
            relative_pos_obs.append(obs[agent_index]['relative_pos'])
            nominal_pos_obs_diff.append(obs[agent_index]['nominal_pos_diff'])
            leader_vel.append(obs[agent_index]['leader_vel'])
            leader_ang_vel.append(obs[agent_index]['leader_ang_vel'])
            nominal_pos_obs.append(obs[agent_index]['nominal_pos'])
        # Stack observations along agent dimension (dim=1)
        # laser_tensor shape: [batch_size, n_agents, laser_dim]
        if self.has_laser == True:
            forward_opening_tensor = torch.stack(forward_opening_obs, dim=1)
            laser_tensor = torch.stack(laser_obs, dim=1)
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("laser:{}".format(laser_tensor))
        # relative_pos_tensor shape: [batch_size, n_agents, pos_dim]
        relative_pos_tensor = torch.stack(relative_pos_obs, dim=1)
        last_action_tensor = torch.stack(last_action_obs, dim=1)

        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        # print("relative pos:{}".format(relative_pos_tensor))
        nominal_pos_diff_tensor = torch.stack(nominal_pos_obs_diff, dim=1)
        nominal_pos_tensor = torch.stack(nominal_pos_obs, dim=1)
        leader_vel_tensor = torch.stack(leader_vel, dim=1)
        leader_ang_vel_tensor = torch.stack(leader_ang_vel, dim=1)
        # print("nominal pos tensor shape:{}".format(nominal_pos_tensor.shape))    #[2, 5, 3]
        # print("last action u tensor shape:{}".format(last_action_tensor.shape))    #[2, 5, 3]

        # print("nominal pos:{}".format(nominal_pos_tensor))
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        
        # Get feature dimensions
        if self.has_laser == True:
            laser_dim = laser_tensor.shape[-1]
            forward_opening_dim = forward_opening_tensor.shape[0]
        pos_dim = relative_pos_tensor.shape[-1]
        nominal_dim = nominal_pos_diff_tensor.shape[-1]
        leader_vel_dim = leader_vel_tensor.shape[-1]
        leader_ang_vel_dim = leader_ang_vel_tensor.shape[-1]
        last_action_dim = last_action_tensor.shape[-1]

        # Reshape tensors to combine batch and agent dimensions for concatenation
        # [batch_size, n_agents, feature_dim] -> [batch_size * n_agents, feature_dim]
        # print("forward_opening tensor hsape:{}".format(forward_opening_tensor.shape))

        if self.has_laser == True:
            laser_flat = laser_tensor.reshape(-1, laser_dim)
            reshaped_tensor = forward_opening_tensor.reshape(2, -1) # -1 infers the dimension size, which will be 5*20=100
            forward_opening_flat = reshaped_tensor.transpose(0, 1)
        # print("last_action_dim:{}".format(last_action_dim))

        last_action_flat = last_action_tensor.reshape(-1, last_action_dim)
        relative_pos_flat = relative_pos_tensor.reshape(-1, pos_dim)
        nominal_pos_flat = nominal_pos_tensor.reshape(-1, nominal_dim)
        leader_vel_flat = leader_vel_tensor.reshape(-1, leader_vel_dim)
        leader_ang_vel_flat = leader_ang_vel_tensor.reshape(-1, leader_ang_vel_dim)
        # Concatenate features along feature dimension
        # [batch_size * n_agents, combined_dim]
        # combined = torch.cat([laser_flat, relative_pos_flat, leader_vel_flat, leader_ang_vel_flat, nominal_pos_flat], dim=1)
        
        # combined = torch.cat([relative_pos_flat, leader_vel_flat, leader_ang_vel_flat, nominal_pos_flat], dim=1)
        # print("last_action_flat shape:{}".format(last_action_flat.shape))
        # print("nominal_pos flat shape:{}".format(nominal_pos_flat.shape))

        # print("forward_opening hsape:{}".format(forward_opening_flat.shape))
        # print("relative_pos_flat shape:{}".format(relative_pos_flat.shape))
        if self.has_laser == True:
            combined = torch.cat([laser_flat, relative_pos_flat, nominal_pos_flat, last_action_flat], dim=1)
            # combined = torch.cat([laser_flat, forward_opening_flat, relative_pos_flat, nominal_pos_flat], dim=1)

        else:
        # combined = torch.cat([relative_pos_flat, leader_vel_flat, leader_ang_vel_flat, nominal_pos_flat], dim=1)
            combined = torch.cat([relative_pos_flat, nominal_pos_flat, last_action_flat], dim=1)


        # Reshape back to [batch_size, n_agents, combined_dim]
        # combined_dim = laser_dim + pos_dim + nominal_dim + leader_vel_dim + leader_ang_vel_dim
        # combined_dim =   pos_dim + nominal_dim + leader_vel_dim + leader_ang_vel_dim
        if self.has_laser == True:
            # combined_dim = pos_dim + nominal_dim + laser_dim
            # combined_dim = laser_dim + forward_opening_dim + pos_dim + nominal_dim
            combined_dim = laser_dim + pos_dim + nominal_dim + last_action_dim

        else:
            combined_dim =   pos_dim + nominal_dim
        combined_x = combined.reshape(batch_size, n_agents, combined_dim)
        # print("x:{}".format(combined_x))
        # print(f"Final tensor shape: {combined_x.shape}")  # Shape: [batch_num, agent_num, combined_dim]
        # Initialize edge index and edge attributes
        edge_index = []
        edge_attr = []

        
        
        # Connect each pair of nominal formation agents
        graph_list = []
        for d in range(batch_size):
            # if dones[d] == True:
            #     graph_list.append([])
            #     continue
            x = combined_x[d, :, :]
            edge_index = []
            edge_attr = []
            batch_nominal_pos_tensor = nominal_pos_tensor[d, :]
            # print("batch_nominal_pos sape:{}".format(batch_nominal_pos_tensor.shape))
            # print("x:{}".format(x))
            # input("1")
            # Create a tensor for the nominal formation
            # nominal_formation_tensor = x[:, :-2]  # Assuming the first two dimensions are the 
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    # Add edges between agents

                    agent_pos_i = batch_nominal_pos_tensor[i, :2]  # Get the position part
                    agent_pos_j = batch_nominal_pos_tensor[j, :2]  # Get the position part

                    distance = torch.norm(agent_pos_j - agent_pos_i)
                    if distance <= self.max_connection_distance:  # Check if within threshold distance
                        # Add edges from agent to obstacle
                        # print("add edges index")
                        edge_index.append([i,  j])  # Agent to obstacle
                        edge_index.append([j, i])  # Obstacle to agent
                        edge_attr.append([distance.item()])  # Edge type 1 for agent-obstacle
                        edge_attr.append([distance.item()])
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # [num_edges, 1]
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr = torch.empty((0, 1), dtype=torch.float, device=self.device)

            # print("x:{}".format(x))
            # input("1")
            # Create the PyTorch Geometric data object
            # print("edge_index:{}".format(edge_index))
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graph_list.append(data)
        # graph_list = Batch.from_data_list(graph_list)
        # print("graph_list:{}".format(graph_list))
        # print("graph_list shape:{}".format(graph_list.shape))
        # input("1")
        return graph_list

    def step(self, actions, done_override=None):
        # actions: [num_envs, n_agents, action_dim]
        # print("actions:{}".format(actions))
        # done_override: [num_envs] tensor indicating if done should be set
        # actions_list = [actions[:, i, :] for i in range(self.n_agents)]  # List of tensors per agent
        obs, rewards, dones, infos = self.env.step(actions)




        # print("obs:{}".format(obs))
        # print("obs[0] laser shape:{}".format(obs[0]['laser'].shape))
        # print("obs[0] relative pos shape:{}".format(obs[0]['relative_pos'].shape))
        # print("obs combine shaoe:{}".format(torch.cat((obs[0]['laser'], obs[0]['relative_pos']), dim=1).shape))
        # for d in range(self.num_envs):
        obs = self.get_graph_from_obs( obs, dones)
        # obs = obs[0]
        rewards = torch.stack(rewards, dim=1).to(self.device)  # [num_envs, n_agents]
        # dones = torch.stack(dones, dim=1).to(self.device)  # [num_envs, n_agents]
        # print("rewards:{}".format(rewards))
        # Sum rewards across agents
        summed_rewards = rewards.sum(dim=1)  # [num_envs]
        # print("summed reward:{}".format(summed_rewards))
        # If done_override is provided, set done flags accordingly
        # print("done override shape:{}".format(done_override.shape))
        if done_override is not None:
            dones = dones | done_override  # Broadcast to [num_envs, n_agents]
        # print("dones shape:{}".format(dones.shape))
        # print("dones:{}".format(dones))

        return obs, summed_rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=False,
                    )


class GATActor(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents, x_limit, y_limit, theta_limit):
        super(GATActor, self).__init__()
        self.num_agents = num_agents


        gnn_input_channels = in_channels 
        # GAT layers
        self.conv1 = GATConv(gnn_input_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

        



        limits_tensor = torch.tensor([x_limit, y_limit, theta_limit], dtype=torch.float32)
        self.action_limits = limits_tensor.view(1, action_dim)
        # self.register_buffer('action_limits', limits_tensor.view(1, self.action_dim))
        # Global pooling layer
        self.pool = global_mean_pool

        # Actor network (policy head)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16, hidden_channels * 4),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_channels * 4, action_dim * 2)
        # self.log_std = nn.Parameter(torch.zeros(1, 1, action_dim))
    def forward(self, data):
        x_all_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # print("x_all_features shape:{}".format(x_all_features.shape))
        # nominal_pos_diff = x[:, 3:]

        
        

        
        x = self.conv1(x_all_features, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  # Shape: [batch_size, hidden_channels * 8]

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, data.batch, data.num_graphs)

        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)
        # print("agent_embedding shape:{}".format(agent_embeddings.shape))
        # print("nominal_pos_diff shape:{}".format(nominal_pos_diff.shape))
        # print("graph embedding shape:{}".format(graph_embedding_repeated.shape))
        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)
        # print("combined shape:{}".format(combined.shape))
        # Actor head
        actor_hidden = self.fc1(combined)

        mean_and_std_params = self.fc2(actor_hidden)
        
        # Split into raw mean and raw std parameters
        # Each will have shape: [batch_size * num_agents, action_dim]
        action_mean_raw, action_std_raw_params = torch.chunk(mean_and_std_params, 2, dim=-1)
        
        # Process action_mean_raw: apply tanh and scale by limits
        action_mean_tanh = torch.tanh(action_mean_raw)
        action_mean_scaled = action_mean_tanh * self.action_limits # self.action_limits is broadcasted
        
        min_std_val = 0.01
        max_std_val = 0.45
        std_output_scaled = torch.sigmoid(action_std_raw_params) # scales to (0, 1)
        action_std_processed = min_std_val + std_output_scaled * (max_std_val - min_std_val)
        action_std_processed = action_std_processed + 1e-5 # Epsilon for stability
        # Process action_std_raw_params: apply softplus to ensure positivity
        # action_std_processed = F.softplus(action_std_raw_params)
        # Optional: Add a small epsilon for numerical stability if std can become too close to zero
        # action_std_processed = F.softplus(action_std_raw_params) + 1e-6 
        # --- END MODIFICATION ---

        # Reshape to (num_graphs, num_agents, action_dim)
        action_mean = action_mean_scaled.view(data.num_graphs, self.num_agents, -1)
        action_std = action_std_processed.view(data.num_graphs, self.num_agents, -1)

        return action_mean, action_std

    def extract_agent_embeddings(self, x, batch, batch_size):
        agent_node_indices = []
        for graph_idx in range(batch_size):
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)

        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]
        return agent_embeddings


class GATActorWithLaser(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents, x_limit, y_limit, theta_limit):
        super(GATActorWithLaser, self).__init__()
        self.num_agents = num_agents
        self.lidar_dim = 20
        self.last_action_dim = 3
        # The dimension of features that are NOT LiDAR or last_action
        # self.non_lidar_feature_dim = in_channels - self.lidar_dim - self.last_action_dim

        # --- 1. LiDAR Encoder (pre-processes LiDAR before GNN) ---
        self.lidar_cnn_hidden_channels_c1 = 16
        self.lidar_cnn_hidden_channels_c2 = 32
        self.lidar_embedding_dim = 10 # A slightly larger embedding can be beneficial
        self.lidar_encoder_cnn = nn.Sequential(
            nn.Conv1d(1, self.lidar_cnn_hidden_channels_c1, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(self.lidar_cnn_hidden_channels_c1, self.lidar_cnn_hidden_channels_c2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(self.lidar_cnn_hidden_channels_c2, self.lidar_embedding_dim), nn.ReLU()
        )
        # Autoencoder Decoder for self-supervision
        self.lidar_decoder_mlp = nn.Sequential(
            nn.Linear(self.lidar_embedding_dim, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, self.lidar_dim)
        )
        
        # --- 2. GAT Layers (now take fused features as input) ---
        # The input to the GNN is the original non-lidar features + the processed LiDAR embedding + last action
        gnn_input_channels = in_channels + self.lidar_embedding_dim + self.last_action_dim
        self.conv1 = GATConv(gnn_input_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        # self.gnn_output_dim = hidden_channels * 8
        self.gnn_output_dim = hidden_channels * 8

        self.pool = global_mean_pool

        # --- 3. Actor Head (processes GNN output) ---
        # Input is now simpler: just the GNN's agent embedding and the global graph embedding
        actor_head_input_dim = self.gnn_output_dim * 2 # Agent GNN emb + Global GNN emb
        self.fc1 = nn.Sequential(
            nn.Linear(actor_head_input_dim, hidden_channels * 4),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_channels * 4, action_dim * 2)

        # Action limits
        limits_tensor = torch.tensor([x_limit, y_limit, theta_limit], dtype=torch.float32)
        self.register_buffer('action_limits', limits_tensor.view(1, action_dim))

    def forward(self, data):
        x_all_features, edge_index, edge_attr, batch_map = data.x, data.edge_index, data.edge_attr, data.batch
        
        # --- Step A: Split Raw Features ---
        raw_lidar = x_all_features[:, :self.lidar_dim]
        non_lidar_features = x_all_features[:, self.lidar_dim : self.lidar_dim + 6]
        last_action = x_all_features[:, self.lidar_dim + 6:]

        # --- Step B: Process LiDAR stream to get embedding ---
        processed_lidar_features = self.lidar_encoder_cnn(raw_lidar.unsqueeze(1))
        # Also get reconstruction for the auxiliary loss
        reconstructed_lidar = self.lidar_decoder_mlp(processed_lidar_features)

        # --- Step C: Early Fusion ---
        # Concatenate all node features to create the input for the GNN
        x_for_gnn = torch.cat([non_lidar_features, processed_lidar_features.detach(), last_action], dim=1)

        # --- Step D: Process Fused Features through GNN ---
        x_gnn = self.conv1(x_for_gnn, edge_index, edge_attr.squeeze(dim=1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr)
        x_gnn = F.relu(x_gnn)
        x_gnn = F.relu(self.conv2(x_gnn, edge_index, edge_attr if edge_attr is not None else None))
        x_gnn = self.conv3(x_gnn, edge_index, edge_attr if edge_attr is not None else None)

        node_gnn_embeddings = F.relu(x_gnn)
        
        # Global graph embedding from the GNN
        global_gnn_embedding = self.pool(node_gnn_embeddings, batch_map)

        # --- Step E: Actor Head ---
        agent_gnn_embeddings = self.extract_agent_embeddings(node_gnn_embeddings, batch_map, data.num_graphs)
        graph_embedding_repeated = global_gnn_embedding.repeat_interleave(self.num_agents, dim=0)
        
        combined_for_actor_head = torch.cat([agent_gnn_embeddings, graph_embedding_repeated], dim=1)
        
        actor_hidden = self.fc1(combined_for_actor_head)
        mean_and_std_params = self.fc2(actor_hidden)
        
        # ... (rest of the action generation logic is the same)
        action_mean_raw, action_std_raw_params = torch.chunk(mean_and_std_params, 2, dim=-1)
        action_mean_tanh = torch.tanh(action_mean_raw)
        action_mean_scaled = action_mean_tanh * self.action_limits
        min_std_val, max_std_val = 0.01, 0.45
        std_output_scaled = torch.sigmoid(action_std_raw_params)
        action_std_processed = min_std_val + std_output_scaled * (max_std_val - min_std_val) + 1e-5
        action_mean = action_mean_scaled.view(data.num_graphs, self.num_agents, -1)
        action_std = action_std_processed.view(data.num_graphs, self.num_agents, -1)
        
        # Get original and reconstructed lidar for agent nodes for the loss function
        raw_lidar_for_agents = self.extract_agent_embeddings(raw_lidar, batch_map, data.num_graphs)
        reconstructed_lidar_for_agents = self.extract_agent_embeddings(reconstructed_lidar, batch_map, data.num_graphs)

        return action_mean, action_std, raw_lidar_for_agents, reconstructed_lidar_for_agents

    def extract_agent_embeddings(self, x, batch, batch_size):
        # Your existing implementation
        agent_node_indices = []
        for graph_idx in range(batch_size):
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)
        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        return x[agent_node_indices]

class GATActorWithoutLaser(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents, x_limit, y_limit, theta_limit):
        super(GATActorWithoutLaser, self).__init__()
        self.num_agents = num_agents


        self.lidar_cnn_hidden_channels_c1 = 16 # Hidden channels for first CNN layer (e.g., 16)
        self.lidar_cnn_hidden_channels_c2 = 32 # Hidden channels for second CNN layer (e.g., 32)
        self.lidar_embedding_dim = 5
        self.lidar_dim = 20
        self.forward_opening_dim = 2
        self.lidar_encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.lidar_cnn_hidden_channels_c1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Halves sequence length if lidar_dim is even
            nn.Conv1d(in_channels=self.lidar_cnn_hidden_channels_c1, out_channels=self.lidar_cnn_hidden_channels_c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Reduces sequence length to 1: Output [N, lidar_cnn_hidden_channels_c2, 1]
            nn.Flatten(),            # Output: [N_total_nodes, lidar_cnn_hidden_channels_c2]
            nn.Linear(self.lidar_cnn_hidden_channels_c2, self.lidar_embedding_dim),
            nn.ReLU()
        )
        # Output of lidar_encoder_cnn: [N_total_nodes, lidar_embedding_dim]

        # 2. LiDAR Decoder (MLP to reconstruct raw LiDAR from embedding)
        self.lidar_decoder_mlp = nn.Sequential(
            nn.Linear(self.lidar_embedding_dim, hidden_channels), # Use GNN hidden for consistency
            nn.ReLU(),
            nn.Linear(hidden_channels, self.lidar_dim) # Output raw lidar_dim
            # No final activation if raw LiDAR values are not bounded like [0,1].
            # If LiDAR inputs are normalized to [0,1], add nn.Sigmoid() here.
        )

        self.forward_opening_mlp = nn.Sequential(
            nn.Linear(self.lidar_embedding_dim, hidden_channels), # Use GNN hidden for consistency
            nn.ReLU(),
            nn.Linear(hidden_channels, self.forward_opening_dim) # Output raw lidar_dim
            # No final activation if raw LiDAR values are not bounded like [0,1].
            # If LiDAR inputs are normalized to [0,1], add nn.Sigmoid() here.
        )

        gnn_input_channels = in_channels 
        # GAT layers
        self.conv1 = GATConv(gnn_input_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

        



        limits_tensor = torch.tensor([x_limit, y_limit, theta_limit], dtype=torch.float32)
        self.action_limits = limits_tensor.view(1, action_dim)
        # self.register_buffer('action_limits', limits_tensor.view(1, self.action_dim))
        # Global pooling layer
        self.pool = global_mean_pool

        # Actor network (policy head)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16 + 3, hidden_channels * 4),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_channels * 4, action_dim * 2)
        # self.log_std = nn.Parameter(torch.zeros(1, 1, action_dim))
    def forward(self, data):
        x_all_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # print("x_all_features shape:{}".format(x_all_features.shape))
        # nominal_pos_diff = x[:, 3:]

        original_node_features = x_all_features[:, self.lidar_dim: self.lidar_dim + 6 ] # All columns except the last lidar_dim ones
        
        raw_lidar_scans_all_nodes = x_all_features[:, :self.lidar_dim]  # The first lidar_dim columns
        last_action_features = x_all_features[:, self.lidar_dim + 6:]

        processed_lidar_features = self.lidar_encoder_cnn(raw_lidar_scans_all_nodes.unsqueeze(1))
        # print("processed_lidar_features:{}".format(processed_lidar_features))
        # 2. Decode LiDAR for reconstruction loss (using the same processed features)
        reconstructed_lidar_all_nodes = self.lidar_decoder_mlp(processed_lidar_features)

        # forward_opening = self.forward_opening_mlp(processed_lidar_features)
        # unique_rows_tensor = forward_opening[::5]
        # 3. Early Fusion: Concatenate processed LiDAR features with original non-LiDAR node features
        # x_for_gnn = torch.cat([original_node_features, processed_lidar_features], dim=1)
        
        x = self.conv1(original_node_features, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  # Shape: [batch_size, hidden_channels * 8]

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, data.batch, data.num_graphs)


        agent_lidar_embeddings = self.extract_agent_embeddings(processed_lidar_features, data.batch, data.num_graphs)

        last_action_embeddings = self.extract_agent_embeddings(last_action_features, data.batch, data.num_graphs)
        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)
        # print("agent_embedding shape:{}".format(agent_embeddings.shape))
        # print("nominal_pos_diff shape:{}".format(nominal_pos_diff.shape))
        # print("graph embedding shape:{}".format(graph_embedding_repeated.shape))
        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated, last_action_embeddings], dim=1)
        # print("combined shape:{}".format(combined.shape))
        # Actor head
        actor_hidden = self.fc1(combined)

        mean_and_std_params = self.fc2(actor_hidden)
        
        # Split into raw mean and raw std parameters
        # Each will have shape: [batch_size * num_agents, action_dim]
        action_mean_raw, action_std_raw_params = torch.chunk(mean_and_std_params, 2, dim=-1)
        
        # Process action_mean_raw: apply tanh and scale by limits
        action_mean_tanh = torch.tanh(action_mean_raw)
        action_mean_scaled = action_mean_tanh * self.action_limits # self.action_limits is broadcasted
        
        min_std_val = 0.01
        max_std_val = 0.3
        std_output_scaled = torch.sigmoid(action_std_raw_params) # scales to (0, 1)
        action_std_processed = min_std_val + std_output_scaled * (max_std_val - min_std_val)
        action_std_processed = action_std_processed + 1e-5 # Epsilon for stability
        # Process action_std_raw_params: apply softplus to ensure positivity
        # action_std_processed = F.softplus(action_std_raw_params)
        # Optional: Add a small epsilon for numerical stability if std can become too close to zero
        # action_std_processed = F.softplus(action_std_raw_params) + 1e-6 
        # --- END MODIFICATION ---

        # Reshape to (num_graphs, num_agents, action_dim)
        action_mean = action_mean_scaled.view(data.num_graphs, self.num_agents, -1)
        action_std = action_std_processed.view(data.num_graphs, self.num_agents, -1)

        return action_mean, action_std, raw_lidar_scans_all_nodes, reconstructed_lidar_all_nodes

    def extract_agent_embeddings(self, x, batch, batch_size):
        agent_node_indices = []
        for graph_idx in range(batch_size):
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)

        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]
        return agent_embeddings


class GATCriticWithLaser(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_agents):
        """
        Initializes the Critic model with LiDAR processing capabilities.
        Args:
            in_channels (int): Total dimension of the raw node features (including LiDAR).
            hidden_channels (int): Dimension for hidden layers in GNN and MLPs.
            num_agents (int): Number of agents in the simulation.
        """
        super(GATCriticWithLaser, self).__init__()
        self.num_agents = num_agents
        self.lidar_dim = 20 # As defined in your actor
        self.last_action_dim = 3 # As defined in your actor
        self.non_lidar_feature_dim = in_channels - self.lidar_dim - self.last_action_dim

        # --- Feature Encoders (Mirrors the Actor's Encoders) ---

        # 1. 1D CNN LiDAR Encoder
        lidar_cnn_hidden_channels_c1 = 16
        lidar_cnn_hidden_channels_c2 = 32
        lidar_embedding_dim = 5 # Should match the actor's embedding dim
        self.lidar_encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=lidar_cnn_hidden_channels_c1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=lidar_cnn_hidden_channels_c1, out_channels=lidar_cnn_hidden_channels_c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(lidar_cnn_hidden_channels_c2, lidar_embedding_dim),
            nn.ReLU()
        )

        # 2. GAT Layers for non-LiDAR relational features
        self.conv1 = GATConv(self.non_lidar_feature_dim, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.gnn_output_dim = hidden_channels * 8

        # 3. Global Pooling Layer
        self.pool = global_mean_pool

        # --- Critic Head (Value Estimation) ---
        # Input to the critic head is the concatenation of globally pooled features
        # from the GNN, LiDAR encoder, and last actions.
        critic_head_input_dim = self.gnn_output_dim + lidar_embedding_dim + self.last_action_dim
        self.critic_fc1 = nn.Linear(critic_head_input_dim, hidden_channels * 4)
        self.critic_fc2 = nn.Linear(hidden_channels * 4, 1)  # Outputs a single scalar state value

    def forward(self, data: Batch):
        x_all_features, edge_index, edge_attr, batch_map = data.x, data.edge_index, data.edge_attr, data.batch

        # Split the input features just like in the actor
        raw_lidar_scans_all_nodes = x_all_features[:, :self.lidar_dim]
        original_node_features = x_all_features[:, self.lidar_dim : self.lidar_dim + self.non_lidar_feature_dim]
        last_action_features = x_all_features[:, self.lidar_dim + self.non_lidar_feature_dim:]

        # --- 1. Process LiDAR Features ---
        processed_lidar_features = self.lidar_encoder_cnn(raw_lidar_scans_all_nodes.unsqueeze(1))
        # Globally pool the processed LiDAR features to get a single vector per graph
        global_lidar_embedding = self.pool(processed_lidar_features, batch_map)

        # --- 2. Process Relational Features with GNN ---
        x_gnn = self.conv1(original_node_features, edge_index, edge_attr.squeeze(dim=1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr)
        x_gnn = torch.relu(x_gnn)
        x_gnn = self.conv2(x_gnn, edge_index, edge_attr if edge_attr is not None else None)
        x_gnn = torch.relu(x_gnn)
        x_gnn = self.conv3(x_gnn, edge_index, edge_attr if edge_attr is not None else None)
        node_gnn_embeddings = torch.relu(x_gnn)
        # Globally pool the GNN output features
        global_gnn_embedding = self.pool(node_gnn_embeddings, batch_map)

        # --- 3. Process Last Action Features ---
        # Globally pool the last action features to get an average action representation
        global_last_action_embedding = self.pool(last_action_features, batch_map)
        
        # --- 4. Fusion and Value Estimation ---
        # Concatenate all global embeddings to form a comprehensive state representation
        combined_global_features = torch.cat([
            global_gnn_embedding, 
            global_lidar_embedding, 
            global_last_action_embedding
        ], dim=1)

        # Pass through the critic's MLP head to get the state value
        critic_hidden = F.relu(self.critic_fc1(combined_global_features))
        state_value = self.critic_fc2(critic_hidden)
        
        return state_value


class GATCritic(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_agents):
        super(GATCritic, self).__init__()
        self.num_agents = num_agents
        self.lidar_dim = 20
        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

        # Global pooling layer
        self.pool = global_mean_pool

        # Critic network (value head)
        self.critic_fc1 = nn.Linear(hidden_channels * 8, hidden_channels * 4)
        self.critic_fc2 = nn.Linear(hidden_channels * 4, 1)  # Outputs a scalar value for each agent

    def forward(self, data):
        x_all_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # print("x_all_features shape:{}".format(x_all_features.shape))
        # nominal_pos_diff = x[:, 3:]

        original_node_features = x_all_features[:, self.lidar_dim: self.lidar_dim + 6]

        # 3. Early Fusion: Concatenate processed LiDAR features with original non-LiDAR node features
        # x_for_gnn = torch.cat([original_node_features, processed_lidar_features], dim=1)
        
        x = self.conv1(original_node_features, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  

        # Critic head
        critic_hidden = torch.relu(self.critic_fc1(graph_embedding))
        state_value = self.critic_fc2(critic_hidden)
        return state_value


class ExpertGateNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_experts):
        super(ExpertGateNet, self).__init__()
        self.num_experts = num_experts

        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

        # Global pooling layer
        self.pool = global_mean_pool

        # Gating network (classification head)
        self.fc1 = nn.Linear(hidden_channels * 8, hidden_channels * 4)
        self.fc2 = nn.Linear(hidden_channels * 4, self.num_experts)  # Outputs logits for each expert

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1) if edge_attr is not None else None)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr if edge_attr is not None else None)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr if edge_attr is not None else None)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  # Shape: [batch_size, hidden_channels * 8]

        # Gating head
        gating_hidden = torch.relu(self.fc1(graph_embedding))
        expert_logits = self.fc2(gating_hidden)  # Shape: [batch_size, num_experts]

        return expert_logits  # Raw logits, to be used with CrossEntropyLoss
    




def main(args):
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Arguments: {args}")
    experiment_name = args.experiment_name
    set_seed(0)
    train_map_directory = args.train_map_directory
    device = torch.device("cpu")
    from datetime import datetime
    output_policy_filename = args.output_policy_filename
    output_critic_filename = args.output_critic_filename
    # TensorBoard writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", experiment_name, current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    swanlab.init(
        # 设置项目名
        project="multi_robot_navigation",
        experiment_name=experiment_name,
        # 设置超参数
        config={
            "train_env_type": args.train_env_type,
            "has_laser": args.has_laser,
            "env_nums": args.num_envs,
            "use_leader_laser_only": args.use_leader_laser_only,
        }
    )

    

    # Model output directory
    model_save_dir = os.path.join("models", experiment_name)
    os.makedirs(model_save_dir, exist_ok=True)
    output_policy_path = os.path.join(model_save_dir, output_policy_filename)    
    train_env_type = args.train_env_type 
    steps_per_epoch = args.steps_per_epoch


    curriculum_transition_return = []
    # Initialize the model
    num_agents = 5
    if args.has_laser:
        in_channels = 6  # Adjust based on your observation space
    else:
        in_channels = 6  # Adjust based on your observation space

    hidden_dim = 64
    action_dim = 3  # Adjust based on your action space
    x_limit = args.action_x_limit
    y_limit = args.action_y_limit
    theta_limit = args.action_theta_limit
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # Apply weight initialization to the critic layers only

    num_experts = 2  # Number of expert classes
    expert_gate_net = ExpertGateNet(in_channels=in_channels, hidden_channels=hidden_dim, num_experts=num_experts).to(device)

    # Initialize the models
    if args.has_laser:
        clutter_actor_model = GATActorWithLaser(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)
        empty_actor_model = GATActorWithLaser(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)
        tunnel_transform_actor_model = GATActorWithoutLaser(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)
        tunnel_actor_model = GATActorWithLaser(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)
        # clutter_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)
    
    else:
        clutter_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)

    

    if args.empty_policy_filename != "":
        policy_pretrained_weights = torch.load(args.empty_policy_filename, map_location=device)
        policy_pretrained_weights = {k: v for k, v in policy_pretrained_weights.items() if k in empty_actor_model.state_dict()}
        empty_actor_model.load_state_dict(policy_pretrained_weights)
        print("load policy from {}".format(args.empty_policy_filename))

    else:
        empty_actor_model.apply(initialize_weights)

    if args.clutter_policy_filename != "":
        policy_pretrained_weights = torch.load(args.clutter_policy_filename, map_location=device)
        policy_pretrained_weights = {k: v for k, v in policy_pretrained_weights.items() if k in clutter_actor_model.state_dict()}
        clutter_actor_model.load_state_dict(policy_pretrained_weights)
        print("load policy from {}".format(args.clutter_policy_filename))

    else:
        clutter_actor_model.apply(initialize_weights)

    if args.clutter_to_tunnel_policy_filename != "":
        policy_pretrained_weights = torch.load(args.clutter_to_tunnel_policy_filename, map_location=device)
        policy_pretrained_weights = {k: v for k, v in policy_pretrained_weights.items() if k in tunnel_transform_actor_model.state_dict()}
        tunnel_transform_actor_model.load_state_dict(policy_pretrained_weights)
        print("load policy from {}".format(args.clutter_to_tunnel_policy_filename))

    else:
        tunnel_transform_actor_model.apply(initialize_weights)
    
    if args.tunnel_policy_filename != "":
        policy_pretrained_weights = torch.load(args.tunnel_policy_filename, map_location=device)
        policy_pretrained_weights = {k: v for k, v in policy_pretrained_weights.items() if k in tunnel_actor_model.state_dict()}
        tunnel_actor_model.load_state_dict(policy_pretrained_weights)
        print("load policy from {}".format(args.tunnel_policy_filename))

    else:
        tunnel_actor_model.apply(initialize_weights)
    # model = GATActorCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
    from datetime import datetime

    # Create a unique log directory with a timestamp
    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = f'runs/ppo_training_{current_time}'

    # # Initialize the SummaryWriter with the unique log directory
    # writer = SummaryWriter(log_dir=log_dir)

    # print("clutter_actor para:{}".format(clutter_actor_model.state_dict()))
    # input("1")
    # model.load_state_dict(model_state_dict)
    
        
        # PPO update
        # print("num_total_transitions :{}".format(num_total_transitions))
        
        
        
        
        
        
    from vmas.simulator.utils import save_video
    
    clutter_actor_model.eval()
    tunnel_actor_model.eval()
    eval_rewards_all_episodes = []
    eval_epoch_restart_num = 20 # Number of different evaluation scenarios
    eval_num_envs = 1 # Evaluate one environment at a time for clear video/metrics
    eval_steps_per_episode = 700
    forward_env_type = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    # For simplicity, collision/connection metrics are not re-implemented here
    # but would follow a similar pattern to the training loop's info handling if needed.
    print("eval :train_map_directory:{}".format(train_map_directory))
    for eval_idx in range(eval_epoch_restart_num):
        eval_env = VMASWrapper(
            scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap2", # Or a specific eval scenario
            num_envs=eval_num_envs,
            device=device,
            continuous_actions=True,
            n_agents=num_agents,
            env_type=train_env_type, # Or a dedicated eval type
            is_evaluation_mode=True,
            working_mode="RL",
            evaluation_index=1000 + eval_idx, # Use different eval indices
            has_laser = args.has_laser,
            train_map_directory = train_map_directory,
            use_leader_laser_only = args.use_leader_laser_only, 
        )
        eval_obs_list = eval_env.get_obs() # List of Data objects (len=eval_num_envs)
        
        current_episode_frames = []
        current_episode_reward_sum = 0

        for eval_step_idx in range(eval_steps_per_episode):
            with torch.no_grad():
                eval_batched_obs = Batch.from_data_list(eval_obs_list).to(device)
                # Use deterministic actions (mean) for evaluation
                # forward_env_type = eval_env.get_forward_env_type()

                if args.has_laser == True:
                    if forward_env_type[0].item() == 0:
                        eval_action_mean, _, _, _ = empty_actor_model(eval_batched_obs)
                    elif forward_env_type[0].item() == 1:
                        eval_action_mean, _, _, _ = clutter_actor_model(eval_batched_obs)
                    elif forward_env_type[0].item() == 2:
                        eval_action_mean, _, _, _ = tunnel_transform_actor_model(eval_batched_obs)
                    elif forward_env_type[0].item() == 3:
                        eval_action_mean, _, _, _ = tunnel_actor_model(eval_batched_obs)    
                    # eval_action_mean, _= clutter_actor_model(eval_batched_obs)

                else:
                    eval_action_mean, _= clutter_actor_model(eval_batched_obs)
                # eval_action_mean shape: [eval_num_envs, num_agents, action_dim]
            
            eval_action_for_env = [eval_action_mean[:, i, :] for i in range(num_agents)]
            
            eval_next_obs_list, eval_rewards, eval_dones, eval_info = eval_env.step(eval_action_for_env)
            # eval_rewards, eval_dones are [eval_num_envs]
            forward_env_type = eval_info[0]["env_type"]
            if eval_num_envs == 1: # If rendering a single env
                frame = eval_env.render()
                if frame is not None:
                    current_episode_frames.append(frame)
            
            current_episode_reward_sum += eval_rewards[0].item() # Assuming eval_num_envs=1
            eval_obs_list = eval_next_obs_list

            # if eval_dones[0]: # Assuming eval_num_envs=1
            #     break 
        
        eval_rewards_all_episodes.append(current_episode_reward_sum)
        if current_episode_frames: # Save video of the first eval episode
            save_video(f"{log_dir}/eval_R{eval_idx}", current_episode_frames, fps=15)
        # eval_env.close()

    avg_eval_reward = np.mean(eval_rewards_all_episodes) if eval_rewards_all_episodes else 0
    swanlab.log({'Evaluation/AverageReward': avg_eval_reward}, step=epoch)
    curriculum_transition_return.append(avg_eval_reward)
    writer.add_scalar('Evaluation/AverageReward', avg_eval_reward, epoch)
    print(f'Epoch {epoch + 1} Evaluation: Avg Reward: {avg_eval_reward:.3f}')

    train_map_level = 0
    if train_map_directory == "train_maps_0_clutter":
        train_map_level = 0
    if train_map_directory == "train_maps_1_clutter":
        train_map_level = 1
    if train_map_directory == "train_maps_2_clutter":
        train_map_level = 2
    if train_map_directory == "train_maps_3_clutter":
        train_map_level = 3
    if train_map_directory == "train_maps_4_clutter":
        train_map_level = 4
    if train_map_directory == "train_maps_5_clutter":
        train_map_level = 5
    swanlab.log({'Evaluation/train_map_level': train_map_level}, step=epoch)
    

    if len(curriculum_transition_return) == 5:
        print("curriculum_transition_return:{}".format(curriculum_transition_return))
        curriculum_transition_return_mean = np.mean(curriculum_transition_return)
        print("curriculum transition return mean:{}".format(curriculum_transition_return_mean))
        curriculum_transition_return = []
    else:
        curriculum_transition_return_mean = 0.0
    if avg_eval_reward > best_evaluation_reward:
        best_evaluation_reward = avg_eval_reward
        torch.save(clutter_actor_model.state_dict(), output_policy_filename)
        torch.save(critic_model.state_dict(), output_critic_filename)

        print(f'New best evaluation model saved with avg_reward: {best_evaluation_reward:.4f}')
        if eval_idx == 0 and current_episode_frames : # Save best video again if it's also the first
            save_video(f"{log_dir}/BEST_eval_E{epoch+1}", current_episode_frames, fps=15)



    if train_env_type == "bitmap_tunnel":
        if curriculum_transition_return_mean > -200000.0 and train_map_directory == "train_tunnel_maps_0":
            train_map_directory = "train_tunnel_maps_1"
        elif curriculum_transition_return_mean > 20000.0 and train_map_directory == "train_tunnel_maps_1":
            train_map_directory = "train_tunnel_maps_2"
    elif train_env_type == "bitmap":
        print("change map :{}".format(curriculum_transition_return_mean))
        if curriculum_transition_return_mean > 13000.0 and train_map_directory == "train_maps_0_clutter":
            train_map_directory = "train_maps_1_clutter"
            print("change map to train_maos_1_clutter")
        elif curriculum_transition_return_mean > 10000.0 and train_map_directory == "train_maps_1_clutter":
            train_map_directory = "train_maps_2_clutter"
        elif curriculum_transition_return_mean > 10000.0 and train_map_directory == "train_maps_2_clutter":
            train_map_directory = "train_maps_3_clutter"
        # elif curriculum_transition_return_mean > 10000.0 and train_map_directory == "train_maps_3_clutter":
        #     train_map_directory = "train_maps_4_clutter"   
        # elif curriculum_transition_return_mean > 10000.0 and train_map_directory == "train_maps_4_clutter":
        #     train_map_directory = "train_maps_5_clutter"   
        else:
            print("why?")




    writer.close()
    print("Training finished.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script with GNN for VMAS")

    # Environment and Paths
    parser.add_argument("--experiment_name", type=str, default="ppo_experiment", help="Unique name for the experiment run")
    parser.add_argument("--train_env_type", type=str, required=True, help="Type of training environment (e.g., clutter, door_and_narrow, tunnel, bitmap)")
    parser.add_argument("--policy_filename", type=str, default="", help="Path to pre-trained policy to load (optional)")
    parser.add_argument("--empty_policy_filename", type=str, default="", help="Path to pre-trained policy to load (optional)")
    parser.add_argument("--clutter_policy_filename", type=str, default="", help="Path to pre-trained policy to load (optional)")
    parser.add_argument("--tunnel_policy_filename", type=str, default="", help="Path to pre-trained policy to load (optional)")
    parser.add_argument("--clutter_to_tunnel_policy_filename", type=str, default="", help="Path to pre-trained policy to load (optional)")
    
    
    parser.add_argument("--critic_filename", type=str, default="", help="Path to pre-trained critic to load (optional)")
    parser.add_argument("--output_policy_filename", type=str, default="ppo_policy.pth", help="Suffix for the output policy filename")
    parser.add_argument("--output_critic_filename", type=str, default="ppo_critic.pth", help="Suffix for the output policy filename")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., cpu, cuda, cuda:0)")
    parser.add_argument("--train_map_directory", type=str, default="train_maps_0_clutter", help="train map")
    parser.add_argument("--use_leader_laser_only", action="store_true", help="whether there is laser in the environment")


    # Training Hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=500, help="Number of steps per epoch (data collection steps)")
    parser.add_argument("--num_envs", type=int, default=20, help="Number of parallel environments (batch size for envs)")
    
    parser.add_argument("--action_x_limit", type=float, default=0.3, help="limit x for actions, local target_pose")
    parser.add_argument("--action_y_limit", type=float, default=0.3, help="limit y for actions, local target_pose")
    parser.add_argument("--action_theta_limit", type=float, default=0.3, help="limit theta for actions, local target_pose")

    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--action_std_init", type=float, default=0.6, help="Initial standard deviation for continuous actions (if applicable)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")

    # Reward Weights (example - you'll need to integrate these into your reward function)
    parser.add_argument("--reward_weight_goal", type=float, default=1.0, help="Weight for goal achievement reward")
    parser.add_argument("--reward_weight_collision", type=float, default=-0.5, help="Weight (penalty) for collision")

    parser.add_argument("--has_laser", action="store_true", help="whether there is laser in the environment")
    # Add more reward weights as needed:
    # parser.add_argument("--reward_weight_time_penalty", type=float, default=-0.01, help="Penalty per step")


    # Parse arguments
    args = parser.parse_args()
    
    # Call main training function
    main(args)
