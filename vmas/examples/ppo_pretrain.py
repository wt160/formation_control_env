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

    def set_obstacle_reward_scale(self, scale):
        print("vmaswrapper set scale to {}".format(scale))
        return self.env.set_obstacle_reward_scale(scale)

    def reset(self):
        obs = self.env.reset()
        # print("obs length:{}".format(len(obs[0])))
        # obs = obs[0]
        # obs is a list of observations per agent
        # Stack observations to shape [num_envs, n_agents, obs_dim]
        # obs = torch.stack(obs, dim=1).to(self.device)
        return self.get_obs()

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
            combined_dim =   pos_dim + nominal_dim + last_action_dim
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
        return graph_list


    def step(self, actions, done_override=None):
        # actions: [num_envs, n_agents, action_dim]
        obs, rewards, dones, infos = self.env.step(actions)

        # Process observations into graph format
        obs = self.get_graph_from_obs(obs, dones)
        
        # --- MODIFICATION START ---
        
        # `rewards` is a list of tensors, one for each agent.
        # Stack them into a single tensor of shape [num_envs, n_agents]
        per_agent_rewards = torch.stack(rewards, dim=1).to(self.device)
        
        # Calculate the summed reward for logging and general performance tracking
        summed_rewards = per_agent_rewards.sum(dim=1)  # Shape: [num_envs]
        
        # --- END MODIFICATION ---

        # Handle done override if provided
        if done_override is not None:
            dones = dones | done_override
            
        # --- MODIFICATION: Update return statement ---
        # Return both the summed reward and the per-agent reward tensor
        return obs, summed_rewards, per_agent_rewards, dones, infos
    # def step(self, actions, done_override=None):
    #     # actions: [num_envs, n_agents, action_dim]
    #     # print("actions:{}".format(actions))
    #     # done_override: [num_envs] tensor indicating if done should be set
    #     # actions_list = [actions[:, i, :] for i in range(self.n_agents)]  # List of tensors per agent
    #     obs, rewards, dones, infos = self.env.step(actions)




    #     # print("obs:{}".format(obs))
    #     # print("obs[0] laser shape:{}".format(obs[0]['laser'].shape))
    #     # print("obs[0] relative pos shape:{}".format(obs[0]['relative_pos'].shape))
    #     # print("obs combine shaoe:{}".format(torch.cat((obs[0]['laser'], obs[0]['relative_pos']), dim=1).shape))
    #     # for d in range(self.num_envs):
    #     obs = self.get_graph_from_obs( obs, dones)
    #     # obs = obs[0]
    #     rewards = torch.stack(rewards, dim=1).to(self.device)  # [num_envs, n_agents]
    #     # dones = torch.stack(dones, dim=1).to(self.device)  # [num_envs, n_agents]
    #     # print("rewards:{}".format(rewards))
    #     # Sum rewards across agents
    #     summed_rewards = rewards.sum(dim=1)  # [num_envs]
    #     # print("summed reward:{}".format(summed_rewards))
    #     # If done_override is provided, set done flags accordingly
    #     # print("done override shape:{}".format(done_override.shape))
    #     if done_override is not None:
    #         dones = dones | done_override  # Broadcast to [num_envs, n_agents]
    #     # print("dones shape:{}".format(dones.shape))
    #     # print("dones:{}".format(dones))

    #     return obs, summed_rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self, env_index = 0):
        return self.env.render(
                        mode="rgb_array",
                        env_index = env_index,
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=False,
                    )




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


class GATCriticWithLaser(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_agents):
        super(GATCriticWithLaser, self).__init__()
        self.num_agents = num_agents
        self.lidar_dim = 20
        self.last_action_dim = 3
        self.non_lidar_feature_dim = in_channels - self.lidar_dim - self.last_action_dim

        # --- 1. LiDAR Encoder (Mirrors the Actor's Encoder) ---
        self.lidar_cnn_hidden_channels_c1 = 16
        self.lidar_cnn_hidden_channels_c2 = 32
        self.lidar_embedding_dim = 10 # Must match actor
        self.lidar_encoder_cnn = nn.Sequential(
            nn.Conv1d(1, self.lidar_cnn_hidden_channels_c1, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(self.lidar_cnn_hidden_channels_c1, self.lidar_cnn_hidden_channels_c2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(self.lidar_cnn_hidden_channels_c2, self.lidar_embedding_dim), nn.ReLU()
        )

        # --- 2. GAT Layers (Mirrors the Actor's GNN) ---
        gnn_input_channels = self.non_lidar_feature_dim + self.lidar_embedding_dim + self.last_action_dim
        self.conv1 = GATConv(gnn_input_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.gnn_output_dim = hidden_channels * 8

        self.pool = global_mean_pool

        # --- 3. Critic Head (Value Estimation) ---
        # Input is the globally pooled output of the GNN
        critic_head_input_dim = self.gnn_output_dim
        self.critic_fc1 = nn.Linear(critic_head_input_dim, hidden_channels * 4)
        self.critic_fc2 = nn.Linear(hidden_channels * 4, num_agents)

    def forward(self, data: Batch):
        x_all_features, edge_index, edge_attr, batch_map = data.x, data.edge_index, data.edge_attr, data.batch

        # --- Step A: Split Raw Features ---
        raw_lidar = x_all_features[:, :self.lidar_dim]
        non_lidar_features = x_all_features[:, self.lidar_dim : self.lidar_dim + 6]
        last_action = x_all_features[:, self.lidar_dim + 6:]

        # --- Step B: Process LiDAR stream ---
        processed_lidar_features = self.lidar_encoder_cnn(raw_lidar.unsqueeze(1))

        # --- Step C: Early Fusion ---
        x_for_gnn = torch.cat([non_lidar_features, processed_lidar_features.detach(), last_action], dim=1)

        # --- Step D: Process Fused Features through GNN ---
        x_gnn = self.conv1(x_for_gnn, edge_index, edge_attr.squeeze(dim=1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr)
        x_gnn = F.relu(x_gnn)
        x_gnn = F.relu(self.conv2(x_gnn, edge_index, edge_attr if edge_attr is not None else None))
        x_gnn = self.conv3(x_gnn, edge_index, edge_attr if edge_attr is not None else None)

        node_gnn_embeddings = F.relu(x_gnn)
        
        # --- Step E: Global Pooling and Value Estimation ---
        global_gnn_embedding = self.pool(node_gnn_embeddings, batch_map)
        
        critic_hidden = F.relu(self.critic_fc1(global_gnn_embedding))
        state_values_per_agent = self.critic_fc2(critic_hidden) # Shape: [batch_dim, num_agents]
        
        return state_values_per_agent




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

    smoothed_avg_connection_rew = -10.0  # Initial assumption of low performance
    EMA_ALPHA = 0.05

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

    current_obstacle_reward_scale = 0.4

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
        in_channels = 9  # Adjust based on your observation space

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

        # clutter_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)
    
   

    if args.has_laser:
        critic_model = GATCriticWithLaser(in_channels + 20 + 3, hidden_dim, num_agents).to(device)
    


    if args.policy_filename:
        # Priority 1: Load a full pre-trained actor policy for fine-tuning.
        try:
            policy_pretrained_weights = torch.load(args.policy_filename, map_location=device)
            clutter_actor_model.load_state_dict(policy_pretrained_weights)
            print(f"Successfully loaded full actor policy from: {args.policy_filename}")
        except Exception as e:
            print(f"Error loading full actor policy: {e}. Starting from scratch.")
            clutter_actor_model.apply(initialize_weights)

    elif args.lidar_encoder_filename:
        # Priority 2: Initialize with a pre-trained LiDAR encoder.
        print(f"Initializing actor with pre-trained LiDAR encoder from: {args.lidar_encoder_filename}")
        
        # First, randomly initialize the entire actor model
        clutter_actor_model.apply(initialize_weights)
        
        try:
            # Load the state dict from the saved autoencoder
            autoencoder_weights = torch.load(args.lidar_encoder_filename, map_location=device)
            
            # Create a new state dict for the actor's encoder part by renaming keys
            encoder_weights_for_actor = {}
            for key, value in autoencoder_weights.items():
                if key.startswith("encoder."):
                    # Rename 'encoder.0.weight' to 'lidar_encoder_cnn.0.weight'
                    new_key = key.replace("encoder.", "lidar_encoder_cnn.")
                    encoder_weights_for_actor[new_key] = value
            
            # Load the renamed weights into the actor model.
            # `strict=False` is crucial as we are only loading a subset of the model.
            clutter_actor_model.load_state_dict(encoder_weights_for_actor, strict=False)
            print("  - Successfully loaded pre-trained LiDAR encoder weights into actor.")
        except Exception as e:
            print(f"  - Error loading LiDAR encoder weights for actor: {e}. Using random initialization.")

    else:
        # Priority 3: Train from scratch.
        print("Training actor from scratch with random initialization.")
        clutter_actor_model.apply(initialize_weights)

    # --- Critic Loading (follows the same logic) ---
    if args.critic_filename:
        # Priority 1: Load a full pre-trained critic.
        try:
            critic_pretrained_weights = torch.load(args.critic_filename, map_location=device)
            critic_model.load_state_dict(critic_pretrained_weights)
            print(f"Successfully loaded full critic from: {args.critic_filename}")
        except Exception as e:
            print(f"Error loading full critic: {e}. Starting from scratch.")
            critic_model.apply(initialize_weights)

    elif args.lidar_encoder_filename and args.has_laser:
        # Priority 2: Initialize with a pre-trained LiDAR encoder.
        print(f"Initializing critic with pre-trained LiDAR encoder from: {args.lidar_encoder_filename}")
        
        critic_model.apply(initialize_weights) # Start with random weights
        
        try:
            autoencoder_weights = torch.load(args.lidar_encoder_filename, map_location=device)
            encoder_weights_for_critic = {}
            for key, value in autoencoder_weights.items():
                if key.startswith("encoder."):
                    new_key = key.replace("encoder.", "lidar_encoder_cnn.")
                    encoder_weights_for_critic[new_key] = value
            
            critic_model.load_state_dict(encoder_weights_for_critic, strict=False)
            print("  - Successfully loaded pre-trained LiDAR encoder weights into critic.")
        except Exception as e:
            print(f"  - Error loading LiDAR encoder weights for critic: {e}. Using random initialization.")

    else:
        # Priority 3: Train from scratch.
        print("Training critic from scratch with random initialization.")
        critic_model.apply(initialize_weights)


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

    autoencoder_params = list(clutter_actor_model.lidar_encoder_cnn.parameters()) + \
                     list(clutter_actor_model.lidar_decoder_mlp.parameters())
    autoencoder_optimizer = optim.Adam(autoencoder_params, lr=1e-3) # Often uses a higher LR

    # Optimizer for the rest of the Actor (the PPO part)
    # This includes the GNN and the final MLP heads
    ppo_actor_params = [p for name, p in clutter_actor_model.named_parameters() if 'lidar' not in name]
    actor_optimizer = optim.Adam(ppo_actor_params, lr=3e-4) # Your existing PPO LR

    # The critic optimizer remains separate as before
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=3e-4)


    # actor_optimizer = optim.Adam(clutter_actor_model.parameters(), lr=3e-4)
    # critic_optimizer = optim.Adam(critic_model.parameters(), lr=3e-4)

    # PPO Hyperparameters
    num_epochs = 3000
    num_agents = 5
    # steps_per_epoch = 300
    epoch_restart_num = 1
    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    ppo_epochs = 10
    mini_batch_size_graphs = 256


    def compute_returns_and_advantages(rewards, masks, values, gamma, lam, device):
        # This function now works with tensors of shape [..., num_agents]
        print("rewards shape:{}".format(rewards.shape))
        num_steps, num_envs, num_agents = rewards.shape
        advantages = torch.zeros(num_steps, num_envs, num_agents, device=device)
        gae = torch.zeros(num_envs, num_agents, device=device)

        for step in reversed(range(num_steps)):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lam * masks[step] * gae
            advantages[step] = gae
        
        returns = advantages + values[:-1]
        return returns, advantages
    

    # def compute_returns_and_advantages(rewards, masks, values, gamma, lam, device):
    #     # rewards: [num_steps, num_envs]
    #     # masks: [num_steps, num_envs] (1 if not done, 0 if done)
    #     # values: [num_steps + 1, num_envs] (values[t] is V(s_t), values[num_steps] is V(s_T))
    #     num_steps, num_envs = rewards.shape
    #     advantages = torch.zeros(num_steps, num_envs, device=device)
    #     returns = torch.zeros(num_steps, num_envs, device=device)
    #     gae = torch.zeros(num_envs, device=device)

    #     for step in reversed(range(num_steps)):
    #         delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
    #         gae = delta + gamma * lam * masks[step] * gae
    #         advantages[step] = gae
    #         returns[step] = gae + values[step]
    #     return returns, advantages
    # def compute_returns_and_advantages(rewards, masks, values, gamma, lam):
    #     advantages = torch.zeros_like(rewards).to(device)
    #     returns = torch.zeros_like(rewards).to(device)
    #     gae = 0
    #     # print("advantage shaoe:{}".format(advantages.shape))
    #     for step in reversed(range(len(rewards))):
    #         # print("values[step+1] shape:{}".format(values[step+1].shape))
    #         # print("mask[step] shape:{}".format(masks[step].shape))
    #         delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
    #         gae = delta + gamma * lam * masks[step] * gae
    #         advantages[step] = gae
    #         returns[step] = advantages[step] + values[step]
    #     return returns, advantages

    warm_up_epochs=0
    train_num_envs = args.num_envs
    ep_rewards = []
    best_avg_reward = float('-inf')
    best_evaluation_reward = float('-inf')
    for epoch in range(num_epochs):
        clutter_actor_model.train()
        critic_model.train()

        # [num_envs]
        max_steps_per_episode = 200  # Adjust as needed
        # Initialize storage
        epoch_actions_list = []
        epoch_log_probs_list = []
        epoch_values_list = []
        epoch_rewards_list = []
        epoch_dones_list = []
        epoch_obs_data_list = [] # This will be a flat list of Data objects
        epoch_forward_opening_list = []
        epoch_run_rewards_log = [] # For logging mean re



        obs_storage = []
        actions_storage = []
        log_probs_storage = []
        rewards_storage = []
        dones_storage = []
        values_storage = []
        epoch_rewards = []
        masks_storage = [] 
        epoch_group_center_rewards = []
        epoch_agent_collision_obstacle_rewards = []
        epoch_agent_collision_rewards = []
        epoch_agent_connection_rewards = []
        epoch_agent_action_diff_rewards = []
        epoch_agent_target_collision_rewards = []
        epoch_agent_pos_rewards = []



        # obs = env.reset()  # [num_envs, n_agents, obs_dim]
        for epoch_restart_idx in range(epoch_restart_num):
            env = VMASWrapper(
                scenario_name=args.scenario_name,
                num_envs=train_num_envs,
                device=device,
                continuous_actions=True,
                n_agents=num_agents,
                env_type=train_env_type,
                is_imitation=False,
                working_mode="RL",
                has_laser = args.has_laser,
                train_map_directory = train_map_directory,
                use_leader_laser_only = args.use_leader_laser_only, 

            )
            env.set_obstacle_reward_scale(current_obstacle_reward_scale)
            obs_list_of_data = env.get_obs()
            restart_actions = []
            restart_log_probs = []
            restart_values = []
            restart_rewards = []
            restart_dones = []
            restart_obs_indices_start = len(epoch_obs_data_list)
            # env.render()
            # input("1")
            current_step_counters = torch.zeros(train_num_envs, device=device)
            time_start = time.time()
            for step_idx in range(steps_per_epoch):
                # print("obs:{}".format(obs))
                # print("obs[0]:{}".format(obs[0].x))
                n_agents = num_agents
                # obs_dim = obs.shape[2]
                print("step:{}".format(step_idx))
                # Prepare observations for GNN

                # print("obs_list_of_data:{}".format(obs_list_of_data))
                batched_obs_for_gnn = Batch.from_data_list(obs_list_of_data).to(device)
                # Forward pass through the policy
                with torch.no_grad():
                    # print("obs list size:{}".format(len(obs)))
                    # print("obs:{}".format(obs[0]))
                    # print("obs edge_attr edvice:{}".format(obs[0].edge_attr.device))
                    # print("obs edge_index deviuce:{}".format(obs[0].edge_index.device))
                    # print()
                    # print("batch_obs device:{}".format(batch_obs))
                    # batch_obs = batch_obs.to(device)
                    if args.has_laser == True:
                        # action_mean, action_std = clutter_actor_model(batched_obs_for_gnn) 

                        action_mean, action_std ,_, _ = clutter_actor_model(batched_obs_for_gnn)  # Now returns action_std
                    else:
                        action_mean, action_std = clutter_actor_model(batched_obs_for_gnn) 
                    # print("action_mean:{}".format(action_mean))
                    # print("action std:{}".format(action_std))
                    # input("1")
                    dist = torch.distributions.Normal(action_mean, action_std)
                    
                    env.set_action_mean(action_mean)
                    action_sampled_per_env = dist.sample() # [train_num_envs, num_agents, action_dim]
                    log_prob_per_env = dist.log_prob(action_sampled_per_env).sum(dim=-1) # [train_num_envs, num_agents]
                    # print("action_sample:{}".format(action_sampled_per_env))
                    # print("batch_obs:{}".format(batched_obs_for_gnn))
                    # action_mean, state_value = model(batch_obs)
                    # action_mean = actor_model(batch_obs)
                    state_values_per_env_per_agent = critic_model(batched_obs_for_gnn) # Shape: [train_num_envs, num_agents][train_num_envs]
                    # print("action_mean:{}".format(action_mean))
                
                epoch_obs_data_list.extend(obs_list_of_data) # Add individual Data objects
                restart_actions.append(action_sampled_per_env)
                restart_log_probs.append(log_prob_per_env)
                # restart_values.append(state_value_per_env)
                restart_values.append(state_values_per_env_per_agent)
                # Reshape actions for the environment
                action_for_env_step = [action_sampled_per_env[:, i, :] for i in range(num_agents)]
                # with torch.no_grad():
                #     expert_logits = expert_gate_net(batch_obs)  # [batch_size, num_experts]
                #     expert_probs = torch.softmax(expert_logits, dim=1)  # [batch_size, num_experts]
                #     selected_experts = torch.argmax(expert_probs, dim=1)  # [batch_size]
                # # print("selected_experts:{}".format(selected_experts))
                # # Define desired environment type (e.g., class 1: clutter)
                # desired_class = 1
                # desired_mask = (selected_experts == desired_class)  # [batch_size] bool tensor
                
                # Increment step counters
                current_step_counters += 1  # [num_envs]
                # step_counters += desired_mask.float()
                
                # Determine if max steps reached
                done_override_max_steps = current_step_counters >= max_steps_per_episode
                # Combine done signals: terminate if not desired or max steps reached
                # done_override = max_steps_reached  # [batch_size] bool tensor
                


                # Step the environment with overridden done signals
                next_obs_list_of_data, summed_rewards_for_logging, per_agent_rewards, dones_per_env, infos = \
                    env.step(action_for_env_step, done_override=done_override_max_steps)
                
                # next_obs, rewards, dones, infos = env.step(list_of_agent_actions, done_override=done_override)

                restart_rewards.append(per_agent_rewards)
                restart_dones.append(dones_per_env)

                epoch_run_rewards_log.append(summed_rewards_for_logging.mean().item())
                # Reset step counters for environments that are done
                current_step_counters = torch.where(dones_per_env, torch.zeros_like(current_step_counters), current_step_counters)

                obs_list_of_data = next_obs_list_of_data # For next iteration



                epoch_forward_opening_list.extend(infos[0]["forward_opening"])
                
                if (step_idx + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Restart {epoch_restart_idx+1}/{epoch_restart_num}, Step {step_idx+1}/{steps_per_epoch}")
                # next_obs, rewards, dones, infos = env.step(action_env)
                # env.render()
                # rewards = rewards.to(device)
                # dones = dones.to(device)
                # next_obs = [data.to(device) for data in next_obs]

                # step_counters = torch.where(done_override, torch.zeros_like(step_counters), step_counters)
                # print("rewards device:{}".format(rewards.device))
                # mean_rewards = rewards.mean().item()
                for agent_index, agent in enumerate(env.env.agents):
                    agent_name = agent.name
                    agent_info = infos[agent_index]
                    epoch_agent_pos_rewards.append(agent_info["agent_pos_rew"].mean().cpu().item())
                    epoch_group_center_rewards.append(agent_info["group_center_rew"].mean().cpu().item()) 
                    epoch_agent_collision_obstacle_rewards.append(agent_info["collision_obstacle_rew"].mean().cpu().item()) 
                    epoch_agent_collision_rewards.append(agent_info["agent_collisions"].mean().cpu().item())
                    epoch_agent_connection_rewards.append(agent_info["agent_connection_rew"].mean().cpu().item())
                    epoch_agent_action_diff_rewards.append(agent_info["agent_diff_rew"].mean().cpu().item())
                    epoch_agent_target_collision_rewards.append(agent_info["agent_target_collision"].mean().cpu().item())



            epoch_actions_list.append(torch.stack(restart_actions, dim=0)) # Stacks along new dim 0 (steps)
            epoch_log_probs_list.append(torch.stack(restart_log_probs, dim=0))
            epoch_values_list.append(torch.stack(restart_values, dim=0))
            epoch_rewards_list.append(torch.stack(restart_rewards, dim=0))
            epoch_dones_list.append(torch.stack(restart_dones, dim=0))
                # epoch_rewards.append(mean_rewards)
            
            # env.close()
        
        if not epoch_rewards_list: # Should not happen if steps_per_epoch > 0
            print(f"Epoch {epoch+1}: No data collected. Skipping update.")
            continue

        actions_tensor = torch.cat(epoch_actions_list, dim=0)
        log_probs_tensor = torch.cat(epoch_log_probs_list, dim=0)

        rewards_tensor = torch.stack(epoch_rewards_list, dim=0).squeeze(0)
        values_tensor = torch.stack(epoch_values_list, dim=0).squeeze(0)
        # values_tensor = torch.cat(epoch_values_list, dim=0)
        # rewards_tensor = torch.cat(epoch_rewards_list, dim=0)
        dones_tensor = torch.cat(epoch_dones_list, dim=0)
                # for env_idx in range(batch_size):
                #     # if dones[env_idx] == False:
                #     # if desired_mask[env_idx]:
                #     obs_storage.append(obs[env_idx])
                #     actions_storage.append(action_env[env_idx])
                #     log_probs_storage.append(log_prob[env_idx])
                #     rewards_storage.append(rewards[env_idx])
                #     dones_storage.append(dones[env_idx])
                #     values_storage.append(state_value[env_idx])
                #     masks_storage.append(1.0)  # Mask indicating desired environment
                    # else:
                        # Skip storing data for undesired environments
                        # continue


                # obs_storage.append(obs)
                # actions_storage.append(action_env)
                # log_probs_storage.append(log_prob.view(batch_size, n_agents))
                # rewards_storage.append(rewards)
                # dones_storage.append(dones)
                # values_storage.append(state_value.squeeze(dim=1))
                # obs = next_obs
                # writer.add_scalar('Policy/std', action_std.mean().item(), epoch * steps_per_epoch*epoch_restart + steps_per_epoch*epoch_restart_num + step)
        print("collect time:{}".format(time.time() - time_start))


        avg_agent_collision_rew = np.mean(epoch_agent_collision_rewards)
        avg_group_center_rew = np.mean(epoch_group_center_rewards)
        avg_agent_collision_obstacle_rew = np.mean(epoch_agent_collision_obstacle_rewards)
        avg_agent_connection_rew = np.mean(epoch_agent_connection_rewards)
        avg_agent_action_diff_rew = np.mean(epoch_agent_action_diff_rewards)
        avg_agent_pos_rew = np.mean(epoch_agent_pos_rewards)

        # avg_target_collision_rew = np.mean(epoch_agent_target_collision_rewards)

        # ep_rewards.append(avg_reward)
        # writer.add_scalar('Reward/avg_reward', avg_reward, epoch)
        swanlab.log({'Reward/avg_group_center_rew': avg_group_center_rew, 
                     'Reward/agent_collision_rew': avg_agent_collision_rew,
                     'Reward/agent_collision_obstacle_rew': avg_agent_collision_obstacle_rew,
                     'Reward/agent_connection_rew': avg_agent_connection_rew,
                     'Reward/agent_action_diff_rew':avg_agent_action_diff_rew,
                     'Reward/agent_pos_rew':avg_agent_pos_rew}, step=epoch)
        
        writer.add_scalar('Reward/avg_group_center_rew', avg_group_center_rew, epoch)
        writer.add_scalar('Reward/agent_collision_rew',avg_agent_collision_rew, epoch )
        writer.add_scalar('Reward/agent_collision_obstacle_rew', avg_agent_collision_obstacle_rew, epoch)
        writer.add_scalar('Reward/agent_connection_rew',avg_agent_connection_rew, epoch )


        smoothed_avg_connection_rew = (EMA_ALPHA * avg_agent_connection_rew) + (1 - EMA_ALPHA) * smoothed_avg_connection_rew

        # Log this smoothed value to see the real trend
        writer.add_scalar('Reward/Smoothed_Connection_Reward', smoothed_avg_connection_rew, epoch)
        swanlab.log({'Reward/Smoothed_Connection_Reward': smoothed_avg_connection_rew}, step=epoch)
        # --- Define the bounds for your curriculum ---

        # MIN_CONNECTIVITY_FOR_CURRICULUM = -7.0 
        # # The connectivity reward at which the curriculum reaches its maximum effect.
        # MAX_CONNECTIVITY_FOR_CURRICULUM = 8.0

        # # The obstacle reward scale at the start and end of the curriculum.
        # MIN_OBSTACLE_REWARD_SCALE = 0.4
        # MAX_OBSTACLE_REWARD_SCALE = 15.0

        # # --- Improved Curriculum Logic ---

        # # 1. Clamp the input to the defined range.
        # # This prevents extreme values from having an outsized effect.
        # smoothed_rew_tensor = torch.tensor(smoothed_avg_connection_rew, device=device)

        # # 2. Clamp the input to the defined range.
        # clamped_connection_rew = torch.clamp(
        #     smoothed_rew_tensor, # <-- Use the tensor version here
        #     MIN_CONNECTIVITY_FOR_CURRICULUM, 
        #     MAX_CONNECTIVITY_FOR_CURRICULUM
        # )

        # # 3. Normalize the clamped value to a [0, 1] range.
        # progress_ratio = (clamped_connection_rew - MIN_CONNECTIVITY_FOR_CURRICULUM) / \
        #                 (MAX_CONNECTIVITY_FOR_CURRICULUM - MIN_CONNECTIVITY_FOR_CURRICULUM)

        # # 4. Linearly interpolate the obstacle reward scale based on the progress ratio.
        # current_obstacle_reward_scale = MIN_OBSTACLE_REWARD_SCALE + progress_ratio * \
        #                                 (MAX_OBSTACLE_REWARD_SCALE - MIN_OBSTACLE_REWARD_SCALE)

        
            
        current_obstacle_reward_scale = 80
        # avg_reward = np.mean(epoch_rewards)
        # avg_agent_collision_rew = np.mean(epoch_agent_collision_rewards)
        # avg_agent_connection_rew = np.mean(epoch_agent_connection_rewards)
        # avg_agent_action_diff_rew = np.mean(epoch_agent_action_diff_rewards)
        # avg_target_collision_rew = np.mean(epoch_agent_target_collision_rewards)

        # ep_rewards.append(avg_reward)
        # writer.add_scalar('Reward/avg_reward', avg_reward, epoch)
        # writer.add_scalar('Reward/agent_collision_rew',avg_agent_collision_rew, epoch )
        # writer.add_scalar('Reward/agent_connection_rew',avg_agent_connection_rew, epoch )
        # writer.add_scalar('Reward/agent_action_diff_rew',avg_agent_action_diff_rew, epoch )
        # writer.add_scalar('Reward/target_collision_rew',avg_target_collision_rew, epoch )

        # if avg_reward > best_avg_reward:
        #     best_avg_reward = avg_reward
        #     # **Save the model**
        #     torch.save(actor_model.state_dict(), 'best_ppo_model.pth')
        #     print(f'New best model saved with avg_reward: {avg_reward:.4f}')
        # Convert storage to tensors
        # obs_storage = torch.stack(obs_storage)  # [steps_per_epoch, num_envs, n_agents, obs_dim]
        # actions_storage = torch.stack(actions_storage)  # [steps_per_epoch, num_envs, n_agents, action_dim]
        # print("action_storage shape:{}".format(actions_storage.shape))
        # log_probs_storage = torch.stack(log_probs_storage)  # [steps_per_epoch, num_envs, n_agents]

        # rewards_storage = torch.stack(rewards_storage)  # [steps_per_epoch, num_envs, n_agents]
        # print("rewards_storage shape:{}".format(rewards_storage.shape))
        # dones_storage = torch.stack(dones_storage)  # [steps_per_epoch, num_envs, n_agents]
        # # print("obs list length:{}".format(len(obs_storage)))
        # # print("value list length:{}".format(len(values_storage)))
        # values_storage = torch.stack(values_storage)  # [steps_per_epoch, num_envs, n_agents]
        # print("values_storage shape:{}".format(values_storage.shape))
        # masks_storage_tensor = torch.tensor(masks_storage, dtype=torch.float, device=device)  # [num_transitions]
        # Compute returns and advantages
        with torch.no_grad():
            final_batched_obs_for_gnn = Batch.from_data_list(obs_list_of_data).to(device)
            next_values_for_gae = critic_model(final_batched_obs_for_gnn).squeeze(-1) # [train_num_envs]
            print("next_values_for gae shape:{}".format(next_values_for_gae.shape))
            # Append next_values_for_gae for GAE calculation
            # values_tensor shape: [total_steps, num_envs]
            # next_values_for_gae shape: [num_envs] -> unsqueeze to [1, num_envs]
            print("values_tensor shape:{}".format(values_tensor.shape))

            #next_values_for gae shape:torch.Size([20, 5])
            # values_tensor shape:torch.Size([1, 200, 20, 5])
            values_for_gae = torch.cat([values_tensor, next_values_for_gae.unsqueeze(0)], dim=0)
            masks_for_gae = (~dones_tensor).float().unsqueeze(-1).expand_as(rewards_tensor)# Shape: [total_steps, num_envs]

            returns_batch, advantages_batch = compute_returns_and_advantages(
                rewards_tensor, masks_for_gae, values_for_gae, gamma, lam, device
            ) # Output shapes: [steps, envs, agents]

        # Normalize advantages
        advantages_flat = advantages_batch.view(-1, num_agents) # Flatten to [total_steps * num_envs]
        advantages_flat_norm = (advantages_flat - advantages_flat.mean(dim=0)) / (advantages_flat.std(dim=0) + 1e-8)
        
        # Reshape other tensors to be flat for minibatching
        # Target shape for these: [total_transitions, ...], total_transitions = total_steps * num_envs
        returns_flat = returns_batch.view(-1, num_agents)
        actions_flat = actions_tensor.view(-1, num_agents, action_dim)
        log_probs_flat = log_probs_tensor.view(-1, num_agents)
        # epoch_obs_data_list is already flat list of Data objects, length = total_transitions

        num_total_transitions = len(epoch_obs_data_list)
        assert num_total_transitions == actions_flat.shape[0], "Mismatch in transition counts"

        
        
        
        # PPO update
        # print("num_total_transitions :{}".format(num_total_transitions))
        for _ in range(ppo_epochs):
            permutation_indices = torch.randperm(num_total_transitions, device=device)
            for start_idx in range(0, num_total_transitions, mini_batch_size_graphs):
                end_idx = min(start_idx + mini_batch_size_graphs, num_total_transitions)
                if end_idx - start_idx < 8 : # Skip very small minibatches
                    continue
                
                mb_indices = permutation_indices[start_idx:end_idx]


                # print("epoch_forward_opening length:{}".format(len(epoch_forward_opening_list)))
                # Create minibatch of Data objects
                forward_opening_list_mb = [epoch_forward_opening_list[i] for i in mb_indices.cpu().tolist()] 
                obs_data_list_mb = [epoch_obs_data_list[i] for i in mb_indices.cpu().tolist()] # Get Data objs
                obs_batched_mb_gnn = Batch.from_data_list(obs_data_list_mb).to(device)
                # num_graphs in obs_batched_mb_gnn is len(mb_indices)

                actions_mb = actions_flat[mb_indices]           # [mb_len, num_agents, action_dim]
                log_probs_old_mb = log_probs_flat[mb_indices]   # [mb_len, num_agents]
                returns_mb = returns_flat[mb_indices]           # [mb_len]
                advantages_mb = advantages_flat_norm[mb_indices]# [mb_len]

                # Actor forward pass


                if args.has_laser == True:
                    new_action_mean_mb, new_action_std_mb, agent_raw_lidar_targets_mb, agent_reconstructed_lidar_mb = clutter_actor_model(obs_batched_mb_gnn)
                    # new_action_mean_mb, new_action_std_mb  = clutter_actor_model(obs_batched_mb_gnn)

                else:
                    new_action_mean_mb, new_action_std_mb  = clutter_actor_model(obs_batched_mb_gnn)


                # Shapes: [mb_len, num_agents, action_dim]
                new_dist_mb = torch.distributions.Normal(new_action_mean_mb, new_action_std_mb)
                new_log_probs_mb = new_dist_mb.log_prob(actions_mb).sum(dim=-1) # [mb_len, num_agents]
                entropy_mb = new_dist_mb.entropy().sum(dim=-1) # [mb_len, num_agents]

                # Critic forward pass
                new_state_values_mb = critic_model(obs_batched_mb_gnn) # [mb_len]

                # PPO Ratio and Losses
                # Sum log_probs over agents dimension before calculating ratio
                ratio = torch.exp(new_log_probs_mb - log_probs_old_mb) # [mb_len]
                
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb
                actor_loss = -torch.min(surr1, surr2)
                print("actor_loss shape:{}".format(actor_loss.shape))
                actor_loss = actor_loss.mean()
                
                critic_loss = nn.functional.mse_loss(new_state_values_mb, returns_mb)
                
                entropy_bonus = entropy_mb.mean() # Average entropy over minibatch and agents



                forward_opening_tensor = torch.stack(forward_opening_list_mb, dim=0)

                # Actor update
                reconstruction_loss_coef = 0.0

                actor_optimizer.zero_grad()

                if args.has_laser == True:
                    # forward_opening_loss = F.mse_loss(forward_opening_tensor, forward_opening_output_mb)
                    lidar_reconstruction_loss = F.mse_loss(agent_reconstructed_lidar_mb, agent_raw_lidar_targets_mb)
                    actor_total_loss = actor_loss - entropy_coef * entropy_bonus + reconstruction_loss_coef * lidar_reconstruction_loss #+ 0.1 * forward_opening_loss
                    # actor_total_loss = actor_loss - entropy_coef * entropy_bonus 

                else:
                    actor_total_loss = actor_loss - entropy_coef * entropy_bonus 

                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(clutter_actor_model.parameters(), max_grad_norm)
                actor_optimizer.step()

                # Critic update
                critic_optimizer.zero_grad()
                critic_total_loss = value_loss_coef * critic_loss
                critic_total_loss.backward()
                nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)
                critic_optimizer.step()
                
        # Logging after PPO updates for the epoch
        avg_epoch_run_reward = np.mean(epoch_run_rewards_log) if epoch_run_rewards_log else 0
        
        
        
        swanlab.log({'Training/AverageReward_DuringCollection': avg_epoch_run_reward, 
                     'Loss/ActorLoss': actor_loss.item(),
                     'Loss/CriticLoss': critic_loss.item(),
                     'Loss/EntropyBonus': entropy_bonus.item()}, step=epoch)
        
        writer.add_scalar('Training/AverageReward_DuringCollection', avg_epoch_run_reward, epoch)
        writer.add_scalar('Loss/ActorLoss', actor_loss.item(), epoch) # Last minibatch loss
        writer.add_scalar('Loss/CriticLoss', critic_loss.item(), epoch) # Last minibatch loss
        writer.add_scalar('Loss/EntropyBonus', entropy_bonus.item(), epoch) # Last minibatch entropy
        # Log mean std from actor model (get from a sample batch)
        if len(epoch_obs_data_list) > 0:
            sample_obs_for_std_log = Batch.from_data_list(epoch_obs_data_list[:min(8, len(epoch_obs_data_list))]).to(device)
            with torch.no_grad():
                if args.has_laser == True:
                    _, act_std_sample, _, _ = clutter_actor_model(sample_obs_for_std_log)
                    # _, act_std_sample= clutter_actor_model(sample_obs_for_std_log)

                else:
                    _, act_std_sample = clutter_actor_model(sample_obs_for_std_log)
                writer.add_scalar('Policy/MeanActionStd', act_std_sample.mean().item(), epoch)
                swanlab.log({'Policy/MeanActionStd': act_std_sample.mean().item()}, step=epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}, Avg Reward (collection): {avg_epoch_run_reward:.3f}, ActorL: {actor_loss.item():.3f}, CriticL: {critic_loss.item():.3f}')
            
        
        
        from vmas.simulator.utils import save_video
        if (epoch) % 10 == 0: # Evaluate every 10 epochs
            clutter_actor_model.eval()
            critic_model.eval()
            eval_rewards_all_episodes = []
            eval_epoch_restart_num = 2 # Number of different evaluation scenarios
            eval_num_envs = 2 # Evaluate one environment at a time for clear video/metrics
            eval_steps_per_episode = 200

            # For simplicity, collision/connection metrics are not re-implemented here
            # but would follow a similar pattern to the training loop's info handling if needed.
            print("eval :train_map_directory:{}".format(train_map_directory))
            for eval_idx in range(eval_epoch_restart_num):
                eval_env = VMASWrapper(
                    scenario_name=args.scenario_name, # Or a specific eval scenario
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
                
                current_episode_frames = {}
                current_episode_reward_sum = 0
                for eval_env_index in range(eval_num_envs):
                    current_episode_frames[eval_env_index] = []
                for eval_step_idx in range(eval_steps_per_episode):
                    with torch.no_grad():
                        eval_batched_obs = Batch.from_data_list(eval_obs_list).to(device)
                        # Use deterministic actions (mean) for evaluation

                        if args.has_laser == True:
                            eval_action_mean, _, _, _ = clutter_actor_model(eval_batched_obs)
                            # eval_action_mean, _= clutter_actor_model(eval_batched_obs)

                        else:
                            eval_action_mean, _= clutter_actor_model(eval_batched_obs)
                        # eval_action_mean shape: [eval_num_envs, num_agents, action_dim]
                    
                    eval_action_for_env = [eval_action_mean[:, i, :] for i in range(num_agents)]
                    
                    eval_next_obs_list, eval_rewards, eval_per_agent_rewards, eval_dones, _ = eval_env.step(eval_action_for_env)
                    # eval_rewards, eval_dones are [eval_num_envs]

                    # if eval_num_envs == 1: # If rendering a single env
                    for eval_env_index in range(eval_num_envs):
                        frame = eval_env.render(env_index = eval_env_index)
                        if frame is not None:
                            current_episode_frames[eval_env_index].append(frame)
                
                    current_episode_reward_sum += eval_rewards[0].item() # Assuming eval_num_envs=1
                    eval_obs_list = eval_next_obs_list

                    # if eval_dones[0]: # Assuming eval_num_envs=1
                    #     break 
                
                eval_rewards_all_episodes.append(current_episode_reward_sum)
                for eval_env_index in range(eval_num_envs):
                    if current_episode_frames[eval_env_index]: # Save video of the first eval episode
                        save_video(f"{log_dir}/eval_E{epoch+1}_R{eval_idx}_eval{eval_env_index}", current_episode_frames[eval_env_index], fps=15)
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
            

            if len(curriculum_transition_return) == 3:
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
                elif curriculum_transition_return_mean > 5000.0 and train_map_directory == "train_tunnel_maps_1":
                    train_map_directory = "train_tunnel_maps_2"
            elif train_env_type == "bitmap":
                print("change map :{}".format(curriculum_transition_return_mean))
                if curriculum_transition_return_mean > 50000.0 and train_map_directory == "train_maps_0_clutter":
                    train_map_directory = "train_maps_1_clutter"
                    print("change map to train_maos_1_clutter")
                elif curriculum_transition_return_mean > 50000.0 and train_map_directory == "train_maps_1_clutter":
                    train_map_directory = "train_maps_2_clutter"
                elif curriculum_transition_return_mean > 50000.0 and train_map_directory == "train_maps_2_clutter":
                    train_map_directory = "train_maps_3_clutter"
                # elif curriculum_transition_return_mean > 10000.0 and train_map_directory == "train_maps_3_clutter":
                #     train_map_directory = "train_maps_4_clutter"   
                # elif curriculum_transition_return_mean > 10000.0 and train_map_directory == "train_maps_4_clutter":
                #     train_map_directory = "train_maps_5_clutter"   
                else:
                    print("why?")



            clutter_actor_model.train()
            critic_model.train()

    writer.close()
    print("Training finished.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script with GNN for VMAS")

    # Environment and Paths
    parser.add_argument("--scenario_name", type=str, default="ppo_experiment", help="Unique name for the experiment run")
    parser.add_argument("--experiment_name", type=str, default="ppo_experiment", help="Unique name for the experiment run")
    parser.add_argument("--train_env_type", type=str, required=True, help="Type of training environment (e.g., clutter, door_and_narrow, tunnel, bitmap)")
    parser.add_argument("--policy_filename", type=str, default="", help="Path to pre-trained policy to load (optional)")
    parser.add_argument("--critic_filename", type=str, default="", help="Path to pre-trained critic to load (optional)")
    parser.add_argument("--output_policy_filename", type=str, default="ppo_policy.pth", help="Suffix for the output policy filename")
    parser.add_argument("--output_critic_filename", type=str, default="ppo_critic.pth", help="Suffix for the output policy filename")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., cpu, cuda, cuda:0)")
    parser.add_argument("--train_map_directory", type=str, default="train_maps_0_clutter", help="train map")
    parser.add_argument("--use_leader_laser_only", action="store_true", help="whether there is laser in the environment")
    parser.add_argument("--lidar_encoder_filename", type=str, default="", help="Path to pre-trained policy to load (optional)")



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
