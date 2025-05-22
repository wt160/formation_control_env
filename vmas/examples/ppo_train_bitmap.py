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

import sys

train_env_type = sys.argv[1]
policy_filename = sys.argv[2]
output_policy_filename = sys.argv[3]
steps_per_epoch = int(sys.argv[4])
# Set device



device = sys.argv[5]
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
seed = 0  # Set the seed you want
set_seed(seed)
# Define your environment wrapper
class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents, env_type=None, is_evaluation_mode=False, is_imitation=False, working_mode="imitation", evaluation_index=0, has_laser = True, train_map_directory = "train_maps_0_clutter"):
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
        self.has_laser = has_laser
        self.max_connection_distance = 1.8
        self.device = device
        self.n_agents = n_agents
        self.num_envs = num_envs

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
    
    
    def get_graph_from_obs(self, obs, dones):
        batch_size = obs[0]['nominal_pos'].shape[0]
        n_agents = len(obs)
        
        # Pre-allocate tensors for each agent's observations
        laser_obs = []
        relative_pos_obs = []
        nominal_pos_obs_diff = []
        leader_vel = []
        leader_ang_vel = []
        nominal_pos_obs = []
        # Collect observations by type
        for agent_index in range(n_agents):
            if self.has_laser == True:
                laser_obs.append(obs[agent_index]['laser'])
            relative_pos_obs.append(obs[agent_index]['relative_pos'])
            nominal_pos_obs_diff.append(obs[agent_index]['nominal_pos_diff'])
            leader_vel.append(obs[agent_index]['leader_vel'])
            leader_ang_vel.append(obs[agent_index]['leader_ang_vel'])
            nominal_pos_obs.append(obs[agent_index]['nominal_pos'])
        # Stack observations along agent dimension (dim=1)
        # laser_tensor shape: [batch_size, n_agents, laser_dim]
        if self.has_laser == True:
            laser_tensor = torch.stack(laser_obs, dim=1)
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("laser:{}".format(laser_tensor))
        # relative_pos_tensor shape: [batch_size, n_agents, pos_dim]
        relative_pos_tensor = torch.stack(relative_pos_obs, dim=1)
        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        # print("relative pos:{}".format(relative_pos_tensor))
        nominal_pos_diff_tensor = torch.stack(nominal_pos_obs_diff, dim=1)
        nominal_pos_tensor = torch.stack(nominal_pos_obs, dim=1)
        leader_vel_tensor = torch.stack(leader_vel, dim=1)
        leader_ang_vel_tensor = torch.stack(leader_ang_vel, dim=1)
        # print("nominal pos tensor shape:{}".format(nominal_pos_tensor.shape))    #[2, 5, 3]
        # print("nominal pos:{}".format(nominal_pos_tensor))
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        
        # Get feature dimensions
        if self.has_laser == True:
            laser_dim = laser_tensor.shape[-1]
        pos_dim = relative_pos_tensor.shape[-1]
        nominal_dim = nominal_pos_diff_tensor.shape[-1]
        leader_vel_dim = leader_vel_tensor.shape[-1]
        leader_ang_vel_dim = leader_ang_vel_tensor.shape[-1]
        # Reshape tensors to combine batch and agent dimensions for concatenation
        # [batch_size, n_agents, feature_dim] -> [batch_size * n_agents, feature_dim]
        if self.has_laser == True:
            laser_flat = laser_tensor.reshape(-1, laser_dim)
        relative_pos_flat = relative_pos_tensor.reshape(-1, pos_dim)
        nominal_pos_flat = nominal_pos_tensor.reshape(-1, nominal_dim)
        leader_vel_flat = leader_vel_tensor.reshape(-1, leader_vel_dim)
        leader_ang_vel_flat = leader_ang_vel_tensor.reshape(-1, leader_ang_vel_dim)
        # Concatenate features along feature dimension
        # [batch_size * n_agents, combined_dim]
        if self.has_laser == True:
            combined = torch.cat([relative_pos_flat, nominal_pos_flat, laser_flat], dim=1)
        else:
        # combined = torch.cat([relative_pos_flat, leader_vel_flat, leader_ang_vel_flat, nominal_pos_flat], dim=1)
            combined = torch.cat([relative_pos_flat, nominal_pos_flat], dim=1)


        # Reshape back to [batch_size, n_agents, combined_dim]
        # combined_dim = laser_dim + pos_dim + nominal_dim + leader_vel_dim + leader_ang_vel_dim
        # combined_dim =   pos_dim + nominal_dim + leader_vel_dim + leader_ang_vel_dim
        if self.has_laser == True:
            combined_dim = pos_dim + nominal_dim + laser_dim
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


        self.lidar_cnn_hidden_channels_c1 = 16 # Hidden channels for first CNN layer (e.g., 16)
        self.lidar_cnn_hidden_channels_c2 = 32 # Hidden channels for second CNN layer (e.g., 32)
        self.lidar_embedding_dim = 14
        self.lidar_dim = 20
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


        gnn_input_channels = in_channels - self.lidar_dim + self.lidar_embedding_dim
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

        original_node_features = x_all_features[:, :-self.lidar_dim] # All columns except the last lidar_dim ones
        raw_lidar_scans_all_nodes = x_all_features[:, -self.lidar_dim:]  # The last lidar_dim columns
        
        processed_lidar_features = self.lidar_encoder_cnn(raw_lidar_scans_all_nodes.unsqueeze(1))
        
        # 2. Decode LiDAR for reconstruction loss (using the same processed features)
        reconstructed_lidar_all_nodes = self.lidar_decoder_mlp(processed_lidar_features)

        # 3. Early Fusion: Concatenate processed LiDAR features with original non-LiDAR node features
        x_for_gnn = torch.cat([original_node_features, processed_lidar_features], dim=1)
        
        x = self.conv1(x_for_gnn, edge_index, edge_attr.squeeze(dim=1))
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


class GATCritic(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_agents):
        super(GATCritic, self).__init__()
        self.num_agents = num_agents

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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  # Shape: [batch_size, hidden_channels * 8]

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
# Initialize the model
num_agents = 5
has_laser = True

if has_laser:
    in_channels = 26  # Adjust based on your observation space
else:
    in_channels = 6  
hidden_dim = 64
action_dim = 3  # Adjust based on your action space
x_limit = 0.3
y_limit = 0.3
theta_limit = 0.3
if train_env_type == "bitmap_tunnel":
    train_map_directory = "train_tunnel_maps_0"
elif train_env_type == "bitmap":
    train_map_directory = "train_maps_1_obstacle"
def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# Apply weight initialization to the critic layers only

num_experts = 2  # Number of expert classes
expert_gate_net = ExpertGateNet(in_channels=in_channels, hidden_channels=hidden_dim, num_experts=num_experts).to(device)

# Initialize the models
clutter_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents, x_limit, y_limit, theta_limit).to(device)
critic_model = GATCritic(in_channels, hidden_dim, num_agents).to(device)

# free_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
# Load the pre-trained actor network weights
# pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
# actor_model.load_state_dict(pretrained_weights)
# expert_choose_net_weights = torch.load('best_expert_gate_net_free_clutter.pth')
# pretrained_weights = torch.load('best_imitation_model_clutter.pth', map_location=device)
clutter_pretrained_weights = torch.load(policy_filename, map_location=device)
# free_pretrained_weights = torch.load('best_imitation_model_empty_noise_1.pth', map_location=device)


# pretrained_weights = torch.load('best_ppo_model.pth', map_location=device)

# Filter out keys not present in the actor_model
# free_pretrained_weights = {k: v for k, v in free_pretrained_weights.items() if k in free_actor_model.state_dict()}

# clutter_pretrained_weights = {k: v for k, v in clutter_pretrained_weights.items() if k in clutter_actor_model.state_dict()}
# actor_model.load_state_dict(pretrained_weights)
# pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
# free_actor_model.load_state_dict(free_pretrained_weights, strict=False)

clutter_actor_model.load_state_dict(clutter_pretrained_weights, strict=False)
# expert_gate_net.load_state_dict(expert_choose_net_weights, strict=False)
# Initialize the critic network
# clutter_actor_model.apply(initialize_weights)
critic_model.apply(initialize_weights)
# model = GATActorCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
from datetime import datetime

# Create a unique log directory with a timestamp
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/ppo_training_{current_time}'

# Initialize the SummaryWriter with the unique log directory
writer = SummaryWriter(log_dir=log_dir)

# print("clutter_actor para:{}".format(clutter_actor_model.state_dict()))
# input("1")
# model.load_state_dict(model_state_dict)
actor_optimizer = optim.Adam(clutter_actor_model.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic_model.parameters(), lr=3e-4)

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
    # rewards: [num_steps, num_envs]
    # masks: [num_steps, num_envs] (1 if not done, 0 if done)
    # values: [num_steps + 1, num_envs] (values[t] is V(s_t), values[num_steps] is V(s_T))
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros(num_steps, num_envs, device=device)
    returns = torch.zeros(num_steps, num_envs, device=device)
    gae = torch.zeros(num_envs, device=device)

    for step in reversed(range(num_steps)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages[step] = gae
        returns[step] = gae + values[step]
    return returns, advantages
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
train_num_envs = 20
ep_rewards = []
best_avg_reward = float('-inf')
best_evaluation_reward = float('-inf')
for epoch in range(num_epochs):
    clutter_actor_model.train()
    critic_model.train()

      # [num_envs]
    max_steps_per_episode = 250  # Adjust as needed
    # Initialize storage
    epoch_actions_list = []
    epoch_log_probs_list = []
    epoch_values_list = []
    epoch_rewards_list = []
    epoch_dones_list = []
    epoch_obs_data_list = [] # This will be a flat list of Data objects

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
    # obs = env.reset()  # [num_envs, n_agents, obs_dim]
    for epoch_restart_idx in range(epoch_restart_num):
        env = VMASWrapper(
            scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap2",
            num_envs=train_num_envs,
            device=device,
            continuous_actions=True,
            n_agents=num_agents,
            env_type=train_env_type,
            is_imitation=False,
            working_mode="RL",
            has_laser = has_laser,
            train_map_directory = train_map_directory,
        )
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
                if has_laser == True:
                    action_mean, action_std ,_, _ = clutter_actor_model(batched_obs_for_gnn)  # Now returns action_std
                else:
                    action_mean, action_std = clutter_actor_model(batched_obs_for_gnn)  # Now returns action_std

                # print("action_mean:{}".format(action_mean))
                # print("action std:{}".format(action_std))
                # input("1")
                dist = torch.distributions.Normal(action_mean, action_std)
                
                action_sampled_per_env = dist.sample() # [train_num_envs, num_agents, action_dim]
                log_prob_per_env = dist.log_prob(action_sampled_per_env).sum(dim=-1) # [train_num_envs, num_agents]
                # print("action_sample:{}".format(action_sampled_per_env))
                # print("batch_obs:{}".format(batch_obs))
                # action_mean, state_value = model(batch_obs)
                # action_mean = actor_model(batch_obs)
                state_value_per_env = critic_model(batched_obs_for_gnn).squeeze(-1) # [train_num_envs]
                # print("action_mean:{}".format(action_mean))
            
            epoch_obs_data_list.extend(obs_list_of_data) # Add individual Data objects
            restart_actions.append(action_sampled_per_env)
            restart_log_probs.append(log_prob_per_env)
            restart_values.append(state_value_per_env)
            
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
            next_obs_list_of_data, rewards_per_env, dones_per_env, infos = \
                env.step(action_for_env_step, done_override=done_override_max_steps)
            
            # next_obs, rewards, dones, infos = env.step(list_of_agent_actions, done_override=done_override)

            restart_rewards.append(rewards_per_env)
            restart_dones.append(dones_per_env)

            epoch_run_rewards_log.append(rewards_per_env.mean().item())
            # Reset step counters for environments that are done
            current_step_counters = torch.where(dones_per_env, torch.zeros_like(current_step_counters), current_step_counters)

            obs_list_of_data = next_obs_list_of_data # For next iteration

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
    values_tensor = torch.cat(epoch_values_list, dim=0)
    rewards_tensor = torch.cat(epoch_rewards_list, dim=0)
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

    # avg_reward = np.mean(epoch_rewards)
    avg_agent_collision_rew = np.mean(epoch_agent_collision_rewards)
    avg_group_center_rew = np.mean(epoch_group_center_rewards)
    avg_agent_collision_obstacle_rew = np.mean(epoch_agent_collision_obstacle_rewards)
    avg_agent_connection_rew = np.mean(epoch_agent_connection_rewards)
    # avg_agent_action_diff_rew = np.mean(epoch_agent_action_diff_rewards)
    # avg_target_collision_rew = np.mean(epoch_agent_target_collision_rewards)

    # ep_rewards.append(avg_reward)
    # writer.add_scalar('Reward/avg_reward', avg_reward, epoch)
    writer.add_scalar('Reward/avg_group_center_rew', avg_group_center_rew, epoch)
    writer.add_scalar('Reward/agent_collision_rew',avg_agent_collision_rew, epoch )
    writer.add_scalar('Reward/agent_collision_obstacle_rew', avg_agent_collision_obstacle_rew, epoch)
    writer.add_scalar('Reward/agent_connection_rew',avg_agent_connection_rew, epoch )
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

        # Append next_values_for_gae for GAE calculation
        # values_tensor shape: [total_steps, num_envs]
        # next_values_for_gae shape: [num_envs] -> unsqueeze to [1, num_envs]
        values_for_gae = torch.cat([values_tensor, next_values_for_gae.unsqueeze(0)], dim=0)
        masks_for_gae = (~dones_tensor).float() # Shape: [total_steps, num_envs]

        returns_batch_per_env, advantages_batch_per_env = compute_returns_and_advantages(
            rewards_tensor, masks_for_gae, values_for_gae, gamma, lam, device
        ) # Output shapes: [total_steps, num_envs]

    # Normalize advantages
    advantages_flat = advantages_batch_per_env.view(-1) # Flatten to [total_steps * num_envs]
    advantages_flat_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
    
    # Reshape other tensors to be flat for minibatching
    # Target shape for these: [total_transitions, ...], total_transitions = total_steps * num_envs
    actions_flat = actions_tensor.reshape(-1, num_agents, action_dim)
    log_probs_flat = log_probs_tensor.reshape(-1, num_agents)
    returns_flat = returns_batch_per_env.reshape(-1)
    # epoch_obs_data_list is already flat list of Data objects, length = total_transitions

    num_total_transitions = len(epoch_obs_data_list)
    assert num_total_transitions == actions_flat.shape[0], "Mismatch in transition counts"

    
    
    
    # PPO update
    train_start_time = time.time()
    print("num_total_transitions :{}".format(num_total_transitions))
    for _ in range(ppo_epochs):
        permutation_indices = torch.randperm(num_total_transitions, device=device)
        for start_idx in range(0, num_total_transitions, mini_batch_size_graphs):
            end_idx = min(start_idx + mini_batch_size_graphs, num_total_transitions)
            if end_idx - start_idx < 8 : # Skip very small minibatches
                continue
            
            mb_indices = permutation_indices[start_idx:end_idx]

            # Create minibatch of Data objects
            obs_data_list_mb = [epoch_obs_data_list[i] for i in mb_indices.cpu().tolist()] # Get Data objs
            obs_batched_mb_gnn = Batch.from_data_list(obs_data_list_mb).to(device)
            # num_graphs in obs_batched_mb_gnn is len(mb_indices)

            actions_mb = actions_flat[mb_indices]           # [mb_len, num_agents, action_dim]
            log_probs_old_mb = log_probs_flat[mb_indices]   # [mb_len, num_agents]
            returns_mb = returns_flat[mb_indices]           # [mb_len]
            advantages_mb = advantages_flat_norm[mb_indices]# [mb_len]

            # Actor forward pass
            new_action_mean_mb, new_action_std_mb, agent_raw_lidar_targets_mb, agent_reconstructed_lidar_mb = clutter_actor_model(obs_batched_mb_gnn)
            # Shapes: [mb_len, num_agents, action_dim]
            new_dist_mb = torch.distributions.Normal(new_action_mean_mb, new_action_std_mb)
            new_log_probs_mb = new_dist_mb.log_prob(actions_mb).sum(dim=-1) # [mb_len, num_agents]
            entropy_mb = new_dist_mb.entropy().sum(dim=-1) # [mb_len, num_agents]

            # Critic forward pass
            new_state_values_mb = critic_model(obs_batched_mb_gnn).squeeze(-1) # [mb_len]

            # PPO Ratio and Losses
            # Sum log_probs over agents dimension before calculating ratio
            ratio = torch.exp(new_log_probs_mb.sum(dim=1) - log_probs_old_mb.sum(dim=1)) # [mb_len]
            
            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.functional.mse_loss(new_state_values_mb, returns_mb)
            
            entropy_bonus = entropy_mb.mean() # Average entropy over minibatch and agents

            # Actor update
            reconstruction_loss_coef = 0.05 
            actor_optimizer.zero_grad()
            lidar_reconstruction_loss = F.mse_loss(agent_reconstructed_lidar_mb, agent_raw_lidar_targets_mb)
            actor_total_loss = actor_loss - entropy_coef * entropy_bonus + reconstruction_loss_coef * lidar_reconstruction_loss
            actor_total_loss.backward()
            nn.utils.clip_grad_norm_(clutter_actor_model.parameters(), max_grad_norm)
            actor_optimizer.step()

            # Critic update
            critic_optimizer.zero_grad()
            critic_total_loss = value_loss_coef * critic_loss
            critic_total_loss.backward()
            nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)
            critic_optimizer.step()
               
    
    
    
    print("training time:{}".format(time.time() - train_start_time))
    input("1")
    # Logging after PPO updates for the epoch
    avg_epoch_run_reward = np.mean(epoch_run_rewards_log) if epoch_run_rewards_log else 0
    writer.add_scalar('Training/AverageReward_DuringCollection', avg_epoch_run_reward, epoch)
    writer.add_scalar('Loss/ActorLoss', actor_loss.item(), epoch) # Last minibatch loss
    writer.add_scalar('Loss/CriticLoss', critic_loss.item(), epoch) # Last minibatch loss
    writer.add_scalar('Loss/EntropyBonus', entropy_bonus.item(), epoch) # Last minibatch entropy
    # Log mean std from actor model (get from a sample batch)
    if len(epoch_obs_data_list) > 0:
        sample_obs_for_std_log = Batch.from_data_list(epoch_obs_data_list[:min(8, len(epoch_obs_data_list))]).to(device)
        with torch.no_grad():
            _, act_std_sample, _, _ = clutter_actor_model(sample_obs_for_std_log)
            writer.add_scalar('Policy/MeanActionStd', act_std_sample.mean().item(), epoch)


    print(f'Epoch {epoch + 1}/{num_epochs}, Avg Reward (collection): {avg_epoch_run_reward:.3f}, ActorL: {actor_loss.item():.3f}, CriticL: {critic_loss.item():.3f}')
          
    
    
    from vmas.simulator.utils import save_video
    if (epoch) % 5 == 0: # Evaluate every 10 epochs
        clutter_actor_model.eval()
        critic_model.eval()
        eval_rewards_all_episodes = []
        eval_epoch_restart_num = 2 # Number of different evaluation scenarios
        eval_num_envs = 1 # Evaluate one environment at a time for clear video/metrics
        eval_steps_per_episode = 500

        # For simplicity, collision/connection metrics are not re-implemented here
        # but would follow a similar pattern to the training loop's info handling if needed.

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
                has_laser = has_laser,
                train_map_directory = train_map_directory,
            )
            eval_obs_list = eval_env.get_obs() # List of Data objects (len=eval_num_envs)
            
            current_episode_frames = []
            current_episode_reward_sum = 0

            for eval_step_idx in range(eval_steps_per_episode):
                with torch.no_grad():
                    eval_batched_obs = Batch.from_data_list(eval_obs_list).to(device)
                    # Use deterministic actions (mean) for evaluation
                    eval_action_mean, _, _, _ = clutter_actor_model(eval_batched_obs)
                    # eval_action_mean shape: [eval_num_envs, num_agents, action_dim]
                
                eval_action_for_env = [eval_action_mean[:, i, :] for i in range(num_agents)]
                
                eval_next_obs_list, eval_rewards, eval_dones, _ = eval_env.step(eval_action_for_env)
                # eval_rewards, eval_dones are [eval_num_envs]

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
                save_video(f"{log_dir}/eval_E{epoch+1}_R{eval_idx}", current_episode_frames, fps=15)
            # eval_env.close()

        avg_eval_reward = np.mean(eval_rewards_all_episodes) if eval_rewards_all_episodes else 0
        writer.add_scalar('Evaluation/AverageReward', avg_eval_reward, epoch)
        print(f'Epoch {epoch + 1} Evaluation: Avg Reward: {avg_eval_reward:.3f}')

        if avg_eval_reward > best_evaluation_reward:
            best_evaluation_reward = avg_eval_reward
            torch.save(clutter_actor_model.state_dict(), output_policy_filename)
            print(f'New best evaluation model saved with avg_reward: {best_evaluation_reward:.4f}')
            if eval_idx == 0 and current_episode_frames : # Save best video again if it's also the first
                 save_video(f"{log_dir}/BEST_eval_E{epoch+1}", current_episode_frames, fps=15)


        if train_env_type == "bitmap_tunnel":
            if avg_eval_reward > 10000.0 and train_map_directory == "train_tunnel_maps_0":
                train_map_directory = "train_tunnel_maps_1"
            elif avg_eval_reward > 10000.0 and train_map_directory == "train_tunnel_maps_1":
                train_map_directory = "train_tunnel_maps_2"
        elif train_env_type == "bitmap":
            if avg_eval_reward > 10000.0 and train_map_directory == "train_maps_1_obstacle":
                train_map_directory = "train_maps_2_obstacle"
            elif avg_eval_reward > 10000.0 and train_map_directory == "train_maps_2_obstacle":
                train_map_directory = "train_maps_3_obstacle"
            elif avg_eval_reward > 10000.0 and train_map_directory == "train_maps_3_obstacle":
                train_map_directory = "train_maps_4_obstacle"
            elif avg_eval_reward > 10000.0 and train_map_directory == "train_maps_4_obstacle":
                train_map_directory = "train_maps_5_obstacle"
        clutter_actor_model.train()
        critic_model.train()

writer.close()
print("Training finished.")
