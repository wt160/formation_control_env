import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool # Assuming these are appropriate for your GNN
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from vmas import make_env # Your VMAS environment
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import sys

# --- Script Arguments ---
train_env_type = sys.argv[1]
# policy_filename = sys.argv[2] # For loading pre-trained, less relevant for OC from scratch
output_policy_filename = sys.argv[3] # Will save the OptionCriticGNN model
steps_per_epoch = int(sys.argv[4])
device_arg = sys.argv[5]
device = torch.device(device_arg if torch.cuda.is_available() else 'cpu')

# --- Seeding ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 42
set_seed(seed)

# --- Environment Wrapper (from your script) ---
class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents, env_type=None, is_evaluation_mode=False, is_imitation=False, working_mode="RL", evaluation_index=0):
        self.env = make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            dict_spaces=False,
            n_agents=n_agents,
            wrapper=None,
            seed=seed + evaluation_index if is_evaluation_mode else seed + num_envs, # Ensure different seeds
            env_type = env_type,
            is_evaluation_mode=is_evaluation_mode,
            is_imitation = is_imitation,
            working_mode=working_mode,
            evaluation_index = evaluation_index,
        )
        self.max_connection_distance = 1.8
        self.device = device
        self.n_agents = n_agents
        self.num_envs = num_envs

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def get_obs(self):
        obs_dict_list = self.env.get_obs() # This is a list of dicts per agent
        # Create a placeholder dones tensor for get_graph_from_obs, assuming all envs are active
        # obs_dict_list[0]['laser'] might not exist if num_envs=0 or after close. Handle safely.
        if not obs_dict_list or 'laser' not in obs_dict_list[0]:
             # Fallback for empty or unexpected obs_dict_list content
            print("Warning: obs_dict_list is empty or missing 'laser' key during get_obs.")
            # Return a list of empty Data objects or handle as an error
            return [Data(x=torch.empty((self.n_agents, 1), device=self.device), # Minimal placeholder
                         edge_index=torch.empty((2,0), dtype=torch.long, device=self.device))
                    for _ in range(self.num_envs)]


        batch_size_check = obs_dict_list[0]['laser'].shape[0]
        dones_placeholder = torch.zeros(batch_size_check, device=self.device, dtype=torch.bool)
        return self.get_graph_from_obs(obs_dict_list, dones_placeholder)


    def get_graph_from_obs(self, obs_dict_list, dones): # obs_dict_list is a list of dicts
        batch_size = obs_dict_list[0]['laser'].shape[0] # num_envs
        n_agents = len(obs_dict_list) # Should be self.n_agents

        laser_obs, relative_pos_obs, nominal_pos_obs, leader_vel, leader_ang_vel = [], [], [], [], []
        for agent_idx in range(n_agents):
            laser_obs.append(obs_dict_list[agent_idx]['laser'])
            relative_pos_obs.append(obs_dict_list[agent_idx]['relative_pos'])
            nominal_pos_obs.append(obs_dict_list[agent_idx]['nominal_pos'])
            leader_vel.append(obs_dict_list[agent_idx]['leader_vel'])
            leader_ang_vel.append(obs_dict_list[agent_idx]['leader_ang_vel'])

        laser_tensor = torch.stack(laser_obs, dim=1)
        relative_pos_tensor = torch.stack(relative_pos_obs, dim=1)
        nominal_pos_tensor = torch.stack(nominal_pos_obs, dim=1)
        leader_vel_tensor = torch.stack(leader_vel, dim=1)
        leader_ang_vel_tensor = torch.stack(leader_ang_vel, dim=1)

        combined_features = torch.cat([
            laser_tensor, relative_pos_tensor, nominal_pos_tensor,
            leader_vel_tensor, leader_ang_vel_tensor
        ], dim=2) # Shape: [batch_size, n_agents, combined_feature_dim]

        graph_list = []
        for i in range(batch_size): # Iterate over each environment in the batch
            if dones[i]: # If an environment is done, add placeholder or skip
                 graph_list.append(Data(x=torch.empty((n_agents, combined_features.shape[-1]), device=self.device),
                                       edge_index=torch.empty((2,0), dtype=torch.long, device=self.device)))
                 continue

            x_env = combined_features[i] # Node features for this environment's graph
            edge_index_env = []
            edge_attr_env = []
            # Simplified edge creation: fully connected within max_connection_distance for nominal positions
            # Using relative_pos for actual distances might be more accurate for dynamic graphs
            # For this example, let's assume nominal_pos for graph structure consistency
            current_nominal_pos = nominal_pos_tensor[i] # [n_agents, pos_dim]
            for j_agent in range(n_agents):
                for k_agent in range(j_agent + 1, n_agents):
                    pos_j = current_nominal_pos[j_agent, :2] # x,y
                    pos_k = current_nominal_pos[k_agent, :2]
                    distance = torch.norm(pos_k - pos_j)
                    if distance <= self.max_connection_distance:
                        edge_index_env.append([j_agent, k_agent])
                        edge_index_env.append([k_agent, j_agent])
                        edge_attr_env.append([distance.item()])
                        edge_attr_env.append([distance.item()])

            if edge_index_env:
                edge_index_tensor = torch.tensor(edge_index_env, dtype=torch.long, device=self.device).t().contiguous()
                edge_attr_tensor = torch.tensor(edge_attr_env, dtype=torch.float, device=self.device)
            else:
                edge_index_tensor = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr_tensor = torch.empty((0, 1), dtype=torch.float, device=self.device)
            graph_list.append(Data(x=x_env, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor))
        return graph_list # List of Data objects, one per environment

    def step(self, actions_list_per_env, done_override=None): # actions_list_per_env is list of tensors
        obs_dict_list, rewards_list_per_agent, dones_list_per_env, infos_list_per_agent = self.env.step(actions_list_per_env)

        next_obs_graphs = self.get_graph_from_obs(obs_dict_list, dones_list_per_env) # dones_list_per_env is already [num_envs]

        rewards_tensor = torch.stack(rewards_list_per_agent, dim=1).to(self.device) # [num_envs, n_agents]
        summed_rewards = rewards_tensor.sum(dim=1) # [num_envs]

        if done_override is not None:
            dones_final = dones_list_per_env | done_override # [num_envs]
        else:
            dones_final = dones_list_per_env

        # print(f"VMASWrapper step: summed_rewards={summed_rewards}, dones={dones_final}")
        return next_obs_graphs, summed_rewards, dones_final, infos_list_per_agent

    def close(self):
        self.env.close()

    def render(self): # Add render method
        return self.env.render(mode="rgb_array")


# --- Option-Critic Network Architecture ---
class OptionCriticGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents, num_options, x_limit, y_limit, theta_limit):
        super(OptionCriticGNN, self).__init__()
        self.num_agents = num_agents
        self.num_options = num_options
        self.action_dim = action_dim

        # GNN Body (Shared Feature Extractor)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value=0.0)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value=0.0)
        self.gnn_feature_dim_node = hidden_channels * 8 # Output dim per node after GATs
        self.gnn_feature_dim_graph = hidden_channels * 8 # Output dim for global_pool

        # For action scaling
        limits_tensor = torch.tensor([x_limit, y_limit, theta_limit], dtype=torch.float32)
        self.register_buffer('action_limits', limits_tensor.view(1, 1, action_dim)) # For broadcasting

        # 1. Policy-over-options (HLP) - Operates on global graph embedding
        self.policy_over_options_head = nn.Sequential(
            nn.Linear(self.gnn_feature_dim_graph, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, num_options) # Logits for options
        )

        # 2. Q-values for options (Critic for HLP) - Operates on global graph embedding
        self.q_options_head = nn.Sequential(
            nn.Linear(self.gnn_feature_dim_graph, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, num_options) # Q-value for each option
        )

        # For per-agent components (intra-option policies)
        # Input: concatenated agent_node_embedding and global_graph_embedding
        self.agent_plus_global_feature_dim = self.gnn_feature_dim_node + self.gnn_feature_dim_graph

        # 3. Intra-option policies (ModuleList of NUM_OPTIONS actor heads)
        self.intra_option_policy_heads = nn.ModuleList([
            self._create_actor_head(self.agent_plus_global_feature_dim, action_dim, hidden_channels)
            for _ in range(num_options)
        ])

        # 4. Termination conditions (ModuleList of NUM_OPTIONS termination heads)
        # Operates on global graph embedding of the *next* state
        self.termination_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.gnn_feature_dim_graph, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1) # Logit for termination (sigmoid applied later)
            ) for _ in range(num_options)
        ])

    def _create_actor_head(self, input_dim, action_dim, hidden_channels):
        # Replicates the actor output structure from GATActor
        return nn.Sequential(
            nn.Linear(input_dim, hidden_channels * 2), # Reduced complexity a bit
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, action_dim * 2) # raw_mean and raw_log_std
        )

    def _process_actor_head_output(self, head_output):
        # head_output shape: [total_agents_in_batch, action_dim * 2]
        mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)

        action_mean_tanh = torch.tanh(mean_raw)
        # action_limits will broadcast: [1, 1, action_dim] to [total_agents_in_batch, action_dim]
        action_mean_scaled = action_mean_tanh * self.action_limits.squeeze(0) # Squeeze for 2D if needed

        # Ensure std is positive and reasonably bounded
        action_std_processed = torch.sigmoid(log_std_raw) # Scales to (0,1)
        min_std_val = 0.05
        max_std_val = 0.5 # Increased max slightly
        action_std_processed = min_std_val + action_std_processed * (max_std_val - min_std_val)
        action_std_processed = action_std_processed + 1e-6 # Epsilon for stability
        return action_mean_scaled, action_std_processed

    def forward_gnn_features(self, data_batch):
        # data_batch is a torch_geometric.data.Batch object
        x, edge_index, edge_attr, batch_map = data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        # print(f"GNN input x: {x.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape if edge_attr is not None else None}, batch_map: {batch_map.shape}")

        # Handle case with no edges correctly for GATConv
        current_edge_attr = edge_attr.squeeze(dim=1) if edge_attr is not None and edge_attr.numel() > 0 else None
        if edge_index.numel() == 0: # No edges in the batch
            # If no edges, GATConv might behave unexpectedly or error.
            # A common approach is to just pass features through linear layers or return zeros.
            # For now, let's assume GATConv handles it or we bypass if truly no edges.
            # This part needs robust handling for edge cases.
            # A simple bypass if GATConv needs edges:
            if self.conv1. απαιτείται_άκρη and edge_index.shape[1] == 0: # Hypothetical check
                 # Create dummy node features if x is empty
                if x.numel() == 0: # If x is also empty (e.g. from a 'done' env placeholder)
                    # This should ideally be filtered out before creating Batch
                    # For robustness, return zero features if node features are empty
                    num_nodes_placeholder = batch_map.max().item() + 1 if batch_map.numel() > 0 else 1
                    # Estimate num_graphs from batch_map if possible, else assume 1 if batch_map is empty
                    num_graphs = batch_map.max().item() + 1 if batch_map.numel() > 0 else 1

                    # Create zero features based on expected dimensions
                    # This logic needs to be robust for actual empty graph cases
                    if x.shape[0] == 0: # if no nodes in the entire batch
                        # This is problematic. Let's assume at least one node per graph for pooling.
                        # For now, let's assume x always has some nodes or this case is filtered.
                        # If a graph in the batch has 0 nodes, global_mean_pool will error.
                        # The get_graph_from_obs should ensure x is not empty for non-done envs.
                        # For done envs, they might produce empty graphs.

                        # Fallback: if x is empty and we must proceed
                        placeholder_node_features = torch.zeros((num_graphs * self.num_agents, self.gnn_feature_dim_node), device=x.device)
                        placeholder_global_features = torch.zeros((num_graphs, self.gnn_feature_dim_graph), device=x.device)
                        return placeholder_node_features, placeholder_global_features, torch.empty((num_graphs*self.num_agents), dtype=torch.long, device=x.device)


            # If GATConv handles empty edge_index (by acting like MLP or using add_self_loops implicitly)
            # This behavior depends on the PyG GATConv version and its defaults.
            # It's safer to ensure edge_index is not passed if it's truly empty and not expected.
            # For now, we proceed, assuming GATConv might default to MLP-like behavior on nodes or use only self-loops if configured.
            # A fill_value of 0.0 might make sense if edge_attr is None.
            # For safety, ensure edge_attr is not None if GATConv expects it and edge_index is not empty.
            # If edge_index is empty, edge_attr should also be conceptually empty or None.
            if edge_index.numel() == 0:
                current_edge_attr = None # GATConv might not use edge_attr if no edges.
        # print(f"Initial x shape: {x.shape}")
        x = self.conv1(x, edge_index, edge_attr=current_edge_attr)
        x = F.relu(x)
        # print(f"After conv1 x shape: {x.shape}")
        # Re-check edge_attr for conv2, GATConv expects edge_dim if edge_attr is passed
        current_edge_attr_conv2 = edge_attr if edge_attr is not None and edge_attr.numel() > 0 else None
        if edge_index.numel() == 0: current_edge_attr_conv2 = None

        x = self.conv2(x, edge_index, edge_attr=current_edge_attr_conv2)
        x = F.relu(x)
        # print(f"After conv2 x shape: {x.shape}")
        # Node features after GNN layers
        node_features = x # Shape: [total_nodes_in_batch, self.gnn_feature_dim_node]

        # Global graph embedding (pooled features)
        # Handle cases where batch_map might be empty if node_features is empty.
        if node_features.numel() > 0 and batch_map.numel() > 0 :
            global_graph_embedding = global_mean_pool(node_features, batch_map) # Shape: [num_graphs_in_batch, self.gnn_feature_dim_node]
        else: # If no nodes in the batch (e.g. all envs in batch were done and returned empty Data)
            # Determine num_graphs, e.g. from expected batch size if possible, or assume 1
            num_graphs_in_batch = data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else 1
            if 'ptr' in data_batch: #ptr can also indicate num_graphs
                 num_graphs_in_batch = data_batch.ptr.numel() -1
                 if num_graphs_in_batch <=0: num_graphs_in_batch = 1 # fallback if ptr is weird

            global_graph_embedding = torch.zeros((num_graphs_in_batch, self.gnn_feature_dim_node), device=x.device)
            # node_features would also be empty or placeholder
            # This case needs robust handling based on how empty graphs are represented and batched.


        return node_features, global_graph_embedding, batch_map


    def get_policy_over_options(self, global_graph_embedding):
        pi_omega_logits = self.policy_over_options_head(global_graph_embedding)
        return Categorical(logits=pi_omega_logits)

    def get_q_options(self, global_graph_embedding):
        return self.q_options_head(global_graph_embedding)

    def get_termination(self, global_graph_embedding_next, option_idx):
        term_logit = self.termination_heads[option_idx](global_graph_embedding_next)
        return torch.sigmoid(term_logit) # Termination probability

    def get_intra_option_action(self, node_features, global_graph_embedding, batch_map, option_idx, num_graphs):
        # node_features: [total_nodes, node_feat_dim]
        # global_graph_embedding: [num_graphs, graph_feat_dim]
        # batch_map: [total_nodes] mapping nodes to graphs

        # We need to extract the first self.num_agents nodes for each graph.
        # And concatenate their features with their respective global_graph_embedding.
        all_agent_node_indices = []
        for i in range(num_graphs):
            graph_node_indices = (batch_map == i).nonzero(as_tuple=True)[0]
            # Assuming the first num_agents nodes in each graph's slice of x are the agents
            # This requires consistent node ordering in Data objects.
            # If graph_node_indices is shorter than num_agents (e.g. for a 'done' placeholder graph)
            # this will error or select fewer. This needs to be robust.
            if len(graph_node_indices) >= self.num_agents:
                 all_agent_node_indices.append(graph_node_indices[:self.num_agents])
            else: # Handle malformed/placeholder graphs for 'done' states
                 # Create placeholder indices if needed, or ensure this case is filtered out
                 # This might involve padding or careful handling of done states upstream
                 # For now, assume valid graphs are passed or this will cause issues.
                 # If we have fewer nodes than agents, this indicates an issue with data prep.
                 # Let's try to proceed, it might error if sizes don't match later.
                 # A robust solution would be to filter out these graphs or pad them.
                 all_agent_node_indices.append(graph_node_indices) # Take all available nodes

        if not all_agent_node_indices: # If batch was entirely empty/problematic
            # Return dummy distributions or handle error
            dummy_mean = torch.zeros((num_graphs, self.num_agents, self.action_dim), device=node_features.device)
            dummy_std = torch.ones((num_graphs, self.num_agents, self.action_dim), device=node_features.device) * 0.1
            return Normal(dummy_mean, dummy_std)


        # Concatenate indices carefully if they are not all same length
        # This part is tricky if num_agents is not constant per graph due to done states.
        # Assuming num_agents is constant for all *active* environments.
        # The current GATActor's extract_agent_embeddings is a good reference but works on a non-batched Data assumption.
        # For a batched `DataBatch` object:
        agent_features_list = []
        for i in range(num_graphs):
            graph_nodes_mask = (batch_map == i)
            current_graph_node_features = node_features[graph_nodes_mask][:self.num_agents] # Take first N agents
            # Pad if fewer than N agents (e.g., for a terminal state's placeholder graph)
            if current_graph_node_features.shape[0] < self.num_agents:
                padding = torch.zeros(
                    (self.num_agents - current_graph_node_features.shape[0], current_graph_node_features.shape[1]),
                    device=current_graph_node_features.device
                )
                current_graph_node_features = torch.cat([current_graph_node_features, padding], dim=0)

            agent_features_list.append(current_graph_node_features)

        # agent_node_features_selected should be [num_graphs * num_agents, node_feat_dim]
        agent_node_features_selected = torch.cat(agent_features_list, dim=0)


        global_embedding_repeated = global_graph_embedding.repeat_interleave(self.num_agents, dim=0)
        # Ensure dimensions match if some graphs had fewer than num_agents originally.
        # If agent_node_features_selected was padded, its total first dim should be num_graphs * num_agents.

        if agent_node_features_selected.shape[0] != global_embedding_repeated.shape[0]:
            # This indicates an issue with node selection or padding.
            # print(f"Shape mismatch: agent_node_features_selected {agent_node_features_selected.shape}, global_embedding_repeated {global_embedding_repeated.shape}")
            # Fallback or error
            dummy_mean = torch.zeros((num_graphs, self.num_agents, self.action_dim), device=node_features.device)
            dummy_std = torch.ones((num_graphs, self.num_agents, self.action_dim), device=node_features.device) * 0.1
            return Normal(dummy_mean, dummy_std)


        combined_agent_features = torch.cat([agent_node_features_selected, global_embedding_repeated], dim=1)

        head_output = self.intra_option_policy_heads[option_idx](combined_agent_features)
        action_mean_flat, action_std_flat = self._process_actor_head_output(head_output)

        # Reshape to [num_graphs, num_agents, action_dim]
        action_mean = action_mean_flat.view(num_graphs, self.num_agents, -1)
        action_std = action_std_flat.view(num_graphs, self.num_agents, -1)

        return Normal(action_mean, action_std)

def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, GATConv)): # Added GATConv
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# --- Hyperparameters for Option-Critic ---
NUM_OPTIONS = 2  # Example: 0 for "standard/exploratory", 1 for "narrow passage"
# From existing PPO script (can be tuned)
in_channels = 29
hidden_dim = 64 # GNN hidden dim
action_dim = 3
num_agents = 5
x_limit, y_limit, theta_limit = 0.2, 0.2, 0.3

LEARNING_RATE_HLP_POLICY = 3e-5 # For policy-over-options
LEARNING_RATE_HLP_CRITIC = 3e-4 # For Q_omega
LEARNING_RATE_OPTIONS_POLICY = 3e-5 # For intra-option policies
LEARNING_RATE_TERMINATION = 3e-5 # For termination functions
GAMMA = 0.99
TERMINATION_REG = 0.01   # Deliberation cost (encourages options to run longer)
ENTROPY_COEF_HLP = 0.01
ENTROPY_COEF_LLP = 0.01
MAX_GRAD_NORM = 0.5

# Initialize model
model = OptionCriticGNN(in_channels, hidden_dim, action_dim, num_agents, NUM_OPTIONS,
                        x_limit, y_limit, theta_limit).to(device)
model.apply(initialize_weights)

# Optimizers for different components
# Shared GNN features will get gradients from all components.
# It's often simpler to have one optimizer for the whole model,
# or group parameters carefully if using separate optimizers.
# For this example, let's try separate optimizers.
# Be cautious about parameters being in multiple optimizers if not intended.

# Optimizer for HLP (policy-over-options head + shared GNN affecting it)
optimizer_hlp_policy = optim.Adam(
    list(model.conv1.parameters()) + list(model.conv2.parameters()) + # GNN body
    list(model.policy_over_options_head.parameters()),
    lr=LEARNING_RATE_HLP_POLICY
)
# Optimizer for HLP Critic (Q_omega head + shared GNN affecting it)
optimizer_hlp_critic = optim.Adam(
    # list(model.conv1.parameters()) + list(model.conv2.parameters()) + # GNN body (already in hlp_policy)
    # If GNN is optimized by HLP policy, Q_critic might only optimize its own head
    # Or, create a combined optimizer for shared parts.
    # Let Q_critic also optimize GNN for its part, Adam handles shared params if in multiple optims (but not ideal)
    # A better way is one optimizer for shared_gnn + hlp_policy_head + hlp_critic_head
    # And then separate for option-specific heads.
    # For now, let's assume GNN gets gradients from both.
    list(model.q_options_head.parameters()), # Only Q-head, assuming GNN updated by HLP policy's loss
    lr=LEARNING_RATE_HLP_CRITIC
)
# Optimizers for each option's policy (intra_option_policy_heads do not include GNN)
optimizers_llp_policy = [
    optim.Adam(model.intra_option_policy_heads[i].parameters(), lr=LEARNING_RATE_OPTIONS_POLICY)
    for i in range(NUM_OPTIONS)
]
# Optimizers for each option's termination (termination_heads do not include GNN)
optimizers_termination = [
    optim.Adam(model.termination_heads[i].parameters(), lr=LEARNING_RATE_TERMINATION)
    for i in range(NUM_OPTIONS)
]


# --- Replay Buffer for Option-Critic (Simplified for on-policy like updates) ---
class OptionCriticBuffer:
    def __init__(self):
        self.obs_graphs = []
        self.hlp_option_choices = []
        self.hlp_log_probs = []
        self.llp_actions = [] # Primitive actions
        self.llp_log_probs = []
        self.rewards = []
        self.dones = []
        self.next_obs_graphs = []
        # We will also need values predicted at s_t for GAE-like calculations if desired
        # For Option-Critic, we primarily need Q_omega(s_t, omega_t) and V_omega(s_t)

    def add(self, obs_graph, hlp_option, hlp_log_prob, llp_action, llp_log_prob, reward, done, next_obs_graph):
        self.obs_graphs.append(obs_graph) # Should be a single Data object
        self.hlp_option_choices.append(hlp_option) # scalar
        self.hlp_log_probs.append(hlp_log_prob) # scalar tensor
        self.llp_actions.append(llp_action) # tensor [num_agents, action_dim]
        self.llp_log_probs.append(llp_log_prob) # scalar tensor (summed over agents)
        self.rewards.append(reward) # scalar
        self.dones.append(done) # bool
        self.next_obs_graphs.append(next_obs_graph) # Should be a single Data object

    def sample_and_clear(self):
        # For this on-policy style update, we'll use all data and clear
        # Convert lists of Data objects to Batch for GNN processing
        # This needs to be done carefully if individual Data objects are stored.
        # It's often easier to store tensors of features if possible, or batch right before GNN.

        # For now, let's assume we process this "batch" of trajectories directly.
        # In a more advanced setup, you'd store tensors.
        data = dict(
            obs_graphs_batch = Batch.from_data_list(self.obs_graphs).to(device),
            hlp_option_choices = torch.tensor(self.hlp_option_choices, dtype=torch.long).to(device),
            hlp_log_probs = torch.stack(self.hlp_log_probs).to(device),
            # llp_actions might need careful stacking if shapes vary slightly due to num_envs > 1 handling
            llp_actions = torch.stack(self.llp_actions).to(device), # Assuming all are [num_agents, action_dim]
            llp_log_probs = torch.stack(self.llp_log_probs).to(device),
            rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device).unsqueeze(1), # [steps, 1]
            dones = torch.tensor(self.dones, dtype=torch.bool).to(device).unsqueeze(1),       # [steps, 1]
            next_obs_graphs_batch = Batch.from_data_list(self.next_obs_graphs).to(device)
        )
        self.clear()
        return data

    def clear(self):
        self.obs_graphs.clear()
        self.hlp_option_choices.clear()
        self.hlp_log_probs.clear()
        self.llp_actions.clear()
        self.llp_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_obs_graphs.clear()

    def __len__(self):
        return len(self.rewards)

# --- Training Loop ---
log_dir_oc = f'runs/option_critic_{train_env_type}_{time.strftime("%Y%m%d-%H%M%S")}'
writer = SummaryWriter(log_dir=log_dir_oc)

num_epochs = 3000
# steps_per_epoch defined by arg
train_num_envs = 20 # Batch size for experience collection
max_steps_per_episode_in_env = 500 # Max steps for one env's episode

buffer = OptionCriticBuffer()

# Agent state for options (per environment)
current_options_env = [None] * train_num_envs
option_just_terminated_env = [True] * train_num_envs # Start by choosing an option

best_eval_reward_oc = -float('inf')

# Initialize environments
envs = [VMASWrapper(
            scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap2", # Scenario name
            num_envs=1, # Each instance in the list handles 1 env
            device=device,
            continuous_actions=True,
            n_agents=num_agents,
            env_type=train_env_type,
            is_imitation=False,
            working_mode="RL",
            evaluation_index = i # Unique seed component
        ) for i in range(train_num_envs)]

current_obs_graphs_env = [env.reset()[0] for env in envs] # Each is a Data object
current_episode_steps_env = [0] * train_num_envs
current_episode_rewards_env = [0.0] * train_num_envs


for epoch in range(num_epochs):
    model.train()
    epoch_total_reward = 0
    num_episodes_this_epoch = 0

    # Collect experiences
    for step_t in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1} Data Collection"):
        # Batch observations for model processing
        # print(f"Step {step_t}, current_obs_graphs_env: {[type(g) for g in current_obs_graphs_env]}")
        # Filter out None if any env was not reset properly (should not happen with good logic)
        valid_obs_graphs_for_batch = [g for g in current_obs_graphs_env if g is not None and g.x.numel() > 0]
        if not valid_obs_graphs_for_batch:
            # print("Warning: No valid observation graphs to batch. Resetting all envs.")
            current_obs_graphs_env = [env.reset()[0] for env in envs]
            current_options_env = [None] * train_num_envs
            option_just_terminated_env = [True] * train_num_envs
            current_episode_steps_env = [0] * train_num_envs
            current_episode_rewards_env = [0.0] * train_num_envs
            valid_obs_graphs_for_batch = current_obs_graphs_env # Should now be valid

        if not valid_obs_graphs_for_batch: # Still no valid graphs
            print("Critical Error: Still no valid graphs after reset attempt. Skipping step.")
            continue

        batched_obs_graphs_t = Batch.from_data_list(valid_obs_graphs_for_batch).to(device)
        # print(f"Batched obs: num_graphs={batched_obs_graphs_t.num_graphs}, num_nodes={batched_obs_graphs_t.num_nodes}")


        node_features_t, global_features_t, batch_map_t = model.forward_gnn_features(batched_obs_graphs_t)

        # Store data for each environment
        # This loop iterates up to train_num_envs, but GNN outputs are for valid_obs_graphs_for_batch
        # Need a mapping or process one env at a time if this becomes too complex
        # For simplicity, let's process one env at a time in the collection to avoid batching headaches here
        # This will be slower but easier to debug initially.
        # The PPO script was already batching envs, so OC should too.
        # We need to map results from batched GNN output back to per-environment decisions.

        # HLP decisions (batched)
        pi_omega_dist_t = model.get_policy_over_options(global_features_t) # [num_valid_envs, NUM_OPTIONS]
        
        actions_to_env_list = []
        
        # This loop should be over num_valid_envs, then map back
        # For now, assume valid_obs_graphs_for_batch maps 1-to-1 with env indices that are active
        active_env_indices = [i for i, g in enumerate(current_obs_graphs_env) if g is not None and g.x.numel() > 0]

        temp_hlp_choices = []
        temp_hlp_log_probs = []
        temp_llp_actions = []
        temp_llp_log_probs = []

        # Iterate through the *active* environments based on batched_obs_graphs_t.num_graphs
        for i_batch_idx in range(batched_obs_graphs_t.num_graphs):
            original_env_idx = active_env_indices[i_batch_idx] # Map batch index back to original env index

            if option_just_terminated_env[original_env_idx] or current_options_env[original_env_idx] is None:
                # Select new option using the i_batch_idx-th slice of pi_omega_dist_t
                current_options_env[original_env_idx] = pi_omega_dist_t.sample()[i_batch_idx].item()
            
            chosen_option = current_options_env[original_env_idx]
            log_prob_pi_omega = pi_omega_dist_t.log_prob(torch.tensor([chosen_option], device=device))[i_batch_idx]

            # LLP action (using the i_batch_idx-th slice of features)
            # Need to select node features for this specific graph from batched node_features_t
            current_graph_node_mask = (batch_map_t == i_batch_idx)
            current_graph_node_features = node_features_t[current_graph_node_mask]
            current_graph_global_feature = global_features_t[i_batch_idx].unsqueeze(0) # Keep batch dim

            # The intra_option_action function needs to handle single graph features
            # For now, let's assume get_intra_option_action can take features for ONE graph
            # and output actions for agents in THAT graph.
            # A simpler get_intra_option_action for single graph:
            num_agents_in_current_graph = current_graph_node_features.shape[0]

            # Replicate global embedding for each node in the current graph
            current_global_rep = current_graph_global_feature.repeat(num_agents_in_current_graph, 1)
            combined_for_llp = torch.cat([current_graph_node_features, current_global_rep], dim=1)

            llp_head_output = model.intra_option_policy_heads[chosen_option](combined_for_llp)
            llp_action_mean, llp_action_std = model._process_actor_head_output(llp_head_output)
            # llp_action_mean/std are [num_agents_in_current_graph, action_dim]

            llp_action_dist = Normal(llp_action_mean, llp_action_std)
            primitive_action = llp_action_dist.sample() # [num_agents_in_current_graph, action_dim]
            log_prob_llp_action = llp_action_dist.log_prob(primitive_action).sum(dim=-1).mean() # Mean over agents

            actions_to_env_list.append(primitive_action.cpu()) # For env.step
            
            # Store data for buffer, associated with original_env_idx
            # The buffer add needs to be called *after* env.step to get next_obs and reward
            # So, cache these per-env decisions for now
            temp_hlp_choices.append((original_env_idx, chosen_option))
            temp_hlp_log_probs.append((original_env_idx, log_prob_pi_omega))
            temp_llp_actions.append((original_env_idx, primitive_action)) # Keep on device for now
            temp_llp_log_probs.append((original_env_idx, log_prob_llp_action))


        # --- This part needs rework for batched env.step ---
        # The `envs.step` expects a list of actions, one tensor per agent, batched over environments
        # So actions_to_env_list needs to be structured as env.step expects
        # For now, let's assume actions_to_env_list contains action tensors for each active env
        # If we collected actions for all `train_num_envs`:
        # formatted_actions_for_env_step = ... # Needs careful assembly

        # Simpler: step environments one by one if batching actions_to_env_list is hard
        next_obs_graphs_env_list_temp = [None] * train_num_envs
        rewards_env_list_temp = [0.0] * train_num_envs
        dones_env_list_temp = [False] * train_num_envs

        for i_env_idx in range(train_num_envs):
            if current_obs_graphs_env[i_env_idx] is None or current_obs_graphs_env[i_env_idx].x.numel() == 0 : # If env was already done or placeholder
                # If we need to fill buffer slots for alignment, add placeholder data or skip
                next_obs_graphs_env_list_temp[i_env_idx] = current_obs_graphs_env[i_env_idx] # Keep placeholder
                # dones_env_list_temp[i_env_idx] = True # It was already done
                continue

            # Find the action for this env from temp_llp_actions
            action_for_this_env = None
            for k_idx, (orig_idx, act) in enumerate(temp_llp_actions):
                if orig_idx == i_env_idx:
                    action_for_this_env = act # This is [num_agents, action_dim]
                    # Remove from temp to avoid re-processing if logic is flawed
                    # temp_llp_actions.pop(k_idx) # Risky if indices change
                    break
            
            if action_for_this_env is None: # Should not happen if logic is correct
                # print(f"Warning: No action found for env {i_env_idx}. Skipping step for this env.")
                next_obs_graphs_env_list_temp[i_env_idx] = current_obs_graphs_env[i_env_idx]
                dones_env_list_temp[i_env_idx] = True # Treat as done to force reset
                continue

            # VMAS step expects list of agent actions: [ [env0_agent0_act, env0_agent1_act,...], [env1_agent0_act,...] ]
            # OR if num_envs=1 in wrapper, expects: [agent0_act, agent1_act, ...] tensor list
            # Our current wrapper takes actions_list_per_env
            # If env is num_envs=1 wrapper, actions_for_this_env is [num_agents, action_dim]
            # We need to convert to list of tensors [action_agent0, action_agent1, ...]
            action_list_for_single_env_step = [action_for_this_env[agent_i,:] for agent_i in range(num_agents)]

            next_obs_graph_single_env_list, reward_single_env, done_single_env, _ = envs[i_env_idx].step(action_list_for_single_env_step)
            # next_obs_graph_single_env_list is a list of Data, but for num_envs=1 it's just [Data_obj]
            next_obs_graphs_env_list_temp[i_env_idx] = next_obs_graph_single_env_list[0]
            rewards_env_list_temp[i_env_idx] = reward_single_env[0].item() # reward_single_env is [1]
            dones_env_list_temp[i_env_idx] = done_single_env[0].item() # done_single_env is [1]

            current_episode_rewards_env[i_env_idx] += rewards_env_list_temp[i_env_idx]
            current_episode_steps_env[i_env_idx] +=1

            # Add to buffer
            # Find corresponding HLP/LLP info
            chosen_opt_this_env, hlp_log_p_this_env, llp_act_this_env, llp_log_p_this_env = -1, None, None, None
            for orig_idx, data_val in temp_hlp_choices:
                if orig_idx == i_env_idx: chosen_opt_this_env = data_val; break
            for orig_idx, data_val in temp_hlp_log_probs:
                if orig_idx == i_env_idx: hlp_log_p_this_env = data_val; break
            # llp_act_this_env is action_for_this_env
            for orig_idx, data_val in temp_llp_log_probs:
                if orig_idx == i_env_idx: llp_log_p_this_env = data_val; break

            if hlp_log_p_this_env is not None and llp_log_p_this_env is not None: # Ensure data was found
                 buffer.add(current_obs_graphs_env[i_env_idx], chosen_opt_this_env, hlp_log_p_this_env,
                           action_for_this_env, llp_log_p_this_env,
                           rewards_env_list_temp[i_env_idx], dones_env_list_temp[i_env_idx],
                           next_obs_graphs_env_list_temp[i_env_idx])

            if dones_env_list_temp[i_env_idx] or current_episode_steps_env[i_env_idx] >= max_steps_per_episode_in_env:
                epoch_total_reward += current_episode_rewards_env[i_env_idx]
                num_episodes_this_epoch +=1
                writer.add_scalar("Training/EpisodeReward", current_episode_rewards_env[i_env_idx], epoch * steps_per_epoch + step_t + i_env_idx) # Approx global step
                current_obs_graphs_env[i_env_idx] = envs[i_env_idx].reset()[0]
                current_options_env[i_env_idx] = None
                option_just_terminated_env[i_env_idx] = True
                current_episode_steps_env[i_env_idx] = 0
                current_episode_rewards_env[i_env_idx] = 0.0
            else:
                current_obs_graphs_env[i_env_idx] = next_obs_graphs_env_list_temp[i_env_idx]
                # Predict termination for next step
                next_obs_graph_for_term = next_obs_graphs_env_list_temp[i_env_idx]
                if next_obs_graph_for_term.x.numel() > 0 : # If not a placeholder for already done
                    batched_next_obs_for_term = Batch.from_data_list([next_obs_graph_for_term]).to(device)
                    _, next_global_features_term, _ = model.forward_gnn_features(batched_next_obs_for_term)
                    termination_prob = model.get_termination(next_global_features_term, chosen_opt_this_env).squeeze()
                    option_just_terminated_env[i_env_idx] = (torch.rand_like(termination_prob) < termination_prob).item()
                else: # If next_obs is placeholder, terminate option
                    option_just_terminated_env[i_env_idx] = True


    # --- Option-Critic Updates ---
    if len(buffer) > 0:
        data_batch = buffer.sample_and_clear() # Returns a dict of batched tensors/graph batches

        obs_b = data_batch['obs_graphs_batch']
        hlp_option_b = data_batch['hlp_option_choices'] # [T]
        hlp_log_prob_b = data_batch['hlp_log_probs']   # [T, 1] or [T]
        llp_action_b = data_batch['llp_actions']       # [T, num_agents, action_dim]
        llp_log_prob_b = data_batch['llp_log_probs']   # [T] (mean over agents)
        reward_b = data_batch['rewards']               # [T, 1]
        done_b = data_batch['dones']                   # [T, 1]
        next_obs_b = data_batch['next_obs_graphs_batch']

        # Forward pass for s_t
        node_feat_t, global_feat_t, batch_map_t = model.forward_gnn_features(obs_b)
        pi_omega_dist_t = model.get_policy_over_options(global_feat_t)
        q_omega_values_t = model.get_q_options(global_feat_t) # [T, NUM_OPTIONS]

        # Forward pass for s_{t+1}
        node_feat_t1, global_feat_t1, batch_map_t1 = model.forward_gnn_features(next_obs_b)
        pi_omega_dist_t1 = model.get_policy_over_options(global_feat_t1) # Not strictly needed for Q target if using max
        q_omega_values_t1 = model.get_q_options(global_feat_t1).detach() # [T, NUM_OPTIONS]


        # Calculate V_omega(s_t) and V_omega(s_{t+1})
        # V(s) = sum_omega' pi_omega(omega'|s) * Q_omega(s, omega') OR max_omega' Q_omega(s,omega')
        # For stability in targets, often max is used for next state value if not using expectation over next HLP policy
        v_omega_s_t = (pi_omega_dist_t.probs * q_omega_values_t).sum(dim=1, keepdim=True).detach() # [T,1]
        v_omega_s_t1 = q_omega_values_t1.max(dim=1, keepdim=True)[0] # Using max_omega' Q(s',omega') as V(s') [T,1]

        # Get termination probabilities beta(s_{t+1}) for the chosen option hlp_option_b
        term_probs_t1_list = []
        for i in range(hlp_option_b.size(0)): # Iterate over transitions in batch
            current_chosen_option = hlp_option_b[i].item()
            # Need global_feat_t1 for the specific transition i
            # Assuming global_feat_t1 is [T, feat_dim]
            term_prob = model.get_termination(global_feat_t1[i].unsqueeze(0), current_chosen_option)
            term_probs_t1_list.append(term_prob)
        term_probs_t1 = torch.cat(term_probs_t1_list) # [T, 1]

        # --- 1. Update Q_omega(s, omega) (HLP Critic) ---
        # Target: r_t + gamma * (1-done) * [ (1-beta(s_t+1)) * Q_omega(s_t+1, omega_t) + beta(s_t+1) * V_omega(s_t+1) ]
        q_s_t_chosen_option = q_omega_values_t.gather(1, hlp_option_b.unsqueeze(1)) # [T, 1]
        q_s_t1_chosen_option = q_omega_values_t1.gather(1, hlp_option_b.unsqueeze(1)) # [T, 1] (value of *same* option in s_t+1)

        q_target = reward_b + GAMMA * (~done_b) * (
            (1 - term_probs_t1.detach()) * q_s_t1_chosen_option +
            term_probs_t1.detach() * v_omega_s_t1
        )
        loss_q_omega = F.mse_loss(q_s_t_chosen_option, q_target.detach())

        optimizer_hlp_critic.zero_grad()
        # If GNN is also part of this optimizer (e.g. combined optimizer), then don't retain graph from HLP policy backward
        # For now, q_options_head is separate.
        loss_q_omega.backward() # Grads for q_options_head
        nn.utils.clip_grad_norm_(model.q_options_head.parameters(), MAX_GRAD_NORM)
        optimizer_hlp_critic.step()


        # --- 2. Update Policy-over-options pi_Omega (HLP Actor) ---
        # Detach Q-values used for advantage calculation to stop gradients from critic loss
        advantage_hlp = (q_s_t_chosen_option.detach() - v_omega_s_t.detach())
        loss_pi_omega = -(hlp_log_prob_b.squeeze() * advantage_hlp.squeeze()).mean() # hlp_log_prob_b might be [T] or [T,1]
        entropy_hlp = pi_omega_dist_t.entropy().mean()
        total_loss_hlp_policy = loss_pi_omega - ENTROPY_COEF_HLP * entropy_hlp

        optimizer_hlp_policy.zero_grad()
        total_loss_hlp_policy.backward() # Grads for policy_over_options_head AND GNN body
        nn.utils.clip_grad_norm_(model.policy_over_options_head.parameters(), MAX_GRAD_NORM)
        nn.utils.clip_grad_norm_(model.conv1.parameters(), MAX_GRAD_NORM) # Assuming GNN params are in this optimizer
        nn.utils.clip_grad_norm_(model.conv2.parameters(), MAX_GRAD_NORM)
        optimizer_hlp_policy.step()


        # --- 3. Update Intra-option policies pi_omega (LLP Actors) ---
        # --- 4. Update Termination conditions beta_omega ---
        for opt_idx in range(NUM_OPTIONS):
            # Find transitions where this option was chosen
            mask_opt_chosen = (hlp_option_b == opt_idx).squeeze() # Boolean mask [T]
            if mask_opt_chosen.sum() == 0: continue # Skip if this option wasn't used

            # LLP Policy Update (REINFORCE-like with Q_omega as return)
            # Q_omega_for_llp_grad is Q_s_t_chosen_option where option was opt_idx
            # llp_log_prob_b is log_prob of action taken by the chosen option
            # We need to ensure these are properly aligned.
            # The Q-value q_s_t_chosen_option is already specific to the option taken at that step.
            loss_llp_policy_opt = -(llp_log_prob_b[mask_opt_chosen] * q_s_t_chosen_option[mask_opt_chosen].detach().squeeze()).mean()
            # Add entropy for LLP
            # Recompute dist for entropy:
            # Need node_features_t, global_features_t, batch_map_t specific to mask_opt_chosen
            # This is getting complex for batching. Simpler: ignore LLP entropy for now or approximate.
            # For now, just policy loss:
            optimizers_llp_policy[opt_idx].zero_grad()
            if loss_llp_policy_opt.requires_grad: # Check if there were any samples
                loss_llp_policy_opt.backward(retain_graph=True) # Retain graph if GNN features affect LLP head directly
                                                                # LLP head does not use GNN directly, uses combined features
                                                                # The computation of combined features from GNN for LLP needs to be part of graph
                nn.utils.clip_grad_norm_(model.intra_option_policy_heads[opt_idx].parameters(), MAX_GRAD_NORM)
                optimizers_llp_policy[opt_idx].step()

            # Termination Update
            # Gradient is (Q_omega(s_t+1, opt_idx) - V_omega(s_t+1)) + deliberation_cost
            # We want to increase beta if Q_omega(s_t+1, opt_idx) < V_omega(s_t+1)
            # Loss for termination: term_prob_t1[mask_opt_chosen] * (V_omega_s_t1[mask_opt_chosen] - Q_omega_values_t1[mask_opt_chosen, opt_idx].unsqueeze(1) - TERMINATION_REG)
            # This requires Q_omega_values_t1 to be for opt_idx, and term_probs_t1 to be for opt_idx
            # The term_probs_t1 was calculated for the *chosen* option at t. This alignment is complex.

            # Simpler termination loss (from Sutton's notes / Bacon paper):
            # The termination head for opt_idx should try to predict if Q(s', opt_idx) - V(s') is negative
            # grad beta_logit(s') = -(Q(s', opt_idx) - V(s') + term_reg)
            # So, loss = beta(s') * (Q(s', opt_idx) - V(s') + term_reg)
            # We need Q_omega_values_t1[mask_opt_chosen, opt_idx] for *this specific option*, not necessarily the one chosen at t+1
            q_s_t1_this_specific_option = q_omega_values_t1[mask_opt_chosen, opt_idx].unsqueeze(1)
            # term_probs_t1 was for the *chosen* option at t. We need beta(s', opt_idx)
            # Recalculate termination for this *specific* option opt_idx using global_feat_t1[mask_opt_chosen]
            current_terminations_for_opt_idx = model.get_termination(global_feat_t1[mask_opt_chosen], opt_idx)

            loss_termination_opt = (current_terminations_for_opt_idx * \
                                   (q_s_t1_this_specific_option.detach() - v_omega_s_t1[mask_opt_chosen].detach() + TERMINATION_REG)
                                  ).mean()

            optimizers_termination[opt_idx].zero_grad()
            if loss_termination_opt.requires_grad:
                loss_termination_opt.backward(retain_graph=True if opt_idx < NUM_OPTIONS -1 else False) # Avoid issues with last backward if GNN involved
                nn.utils.clip_grad_norm_(model.termination_heads[opt_idx].parameters(), MAX_GRAD_NORM)
                optimizers_termination[opt_idx].step()

        # Logging
        writer.add_scalar("Loss/HLP_Policy", total_loss_hlp_policy.item(), epoch)
        writer.add_scalar("Loss/HLP_Critic_QOmega", loss_q_omega.item(), epoch)
        if num_episodes_this_epoch > 0:
             writer.add_scalar("Training/AvgEpisodeReward_Epoch", epoch_total_reward / num_episodes_this_epoch, epoch)
        # Add more logging for option usage, termination probs, etc.

    print(f"Epoch {epoch+1} finished. Avg Reward: {epoch_total_reward / num_episodes_this_epoch if num_episodes_this_epoch > 0 else 0 :.2f}")

    # --- Evaluation Loop (Simplified from your PPO script) ---
    if (epoch + 1) % 10 == 0: # Evaluate every 10 epochs
        model.eval()
        eval_total_reward = 0
        eval_episodes = 5
        for eval_ep_idx in range(eval_episodes):
            eval_env = VMASWrapper( # Create a fresh eval env
                scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap2",
                num_envs=1, device=device, continuous_actions=True, n_agents=num_agents,
                env_type=train_env_type, is_evaluation_mode=True, working_mode="RL",
                evaluation_index=1000 + epoch + eval_ep_idx # Different scenario seed
            )
            eval_obs_graph = eval_env.reset()[0]
            eval_done = False
            eval_ep_reward = 0
            eval_ep_steps = 0
            current_eval_option = None
            eval_option_terminated = True
            frames = []

            while not eval_done and eval_ep_steps < max_steps_per_episode_in_env:
                with torch.no_grad():
                    eval_batched_obs = Batch.from_data_list([eval_obs_graph]).to(device)
                    node_f, global_f, batch_m = model.forward_gnn_features(eval_batched_obs)

                    if eval_option_terminated or current_eval_option is None:
                        pi_omega_dist_eval = model.get_policy_over_options(global_f)
                        current_eval_option = pi_omega_dist_eval.probs.argmax(dim=-1).item() # Greedy choice

                    # LLP action (deterministic = mean for eval)
                    num_g = eval_batched_obs.num_graphs # Should be 1
                    llp_action_dist_eval = model.get_intra_option_action(node_f, global_f, batch_m, current_eval_option, num_g)
                    primitive_action_eval = llp_action_dist_eval.mean # Use mean for deterministic eval

                    action_list_for_eval_env_step = [primitive_action_eval.squeeze(0)[agent_i,:] for agent_i in range(num_agents)]


                next_eval_obs_graph_list, eval_reward, eval_done_arr, _ = eval_env.step(action_list_for_eval_env_step)
                eval_next_obs_graph = next_eval_obs_graph_list[0]
                eval_reward_val = eval_reward[0].item()
                eval_done = eval_done_arr[0].item()

                eval_ep_reward += eval_reward_val
                eval_obs_graph = eval_next_obs_graph
                eval_ep_steps += 1

                if epoch % 50 == 0 and eval_ep_idx == 0 : # Render one eval episode every 50 epochs
                    frames.append(eval_env.render())


                if not eval_done : # Predict termination if not done
                    eval_batched_next_obs = Batch.from_data_list([eval_next_obs_graph]).to(device)
                    _, next_global_f_eval, _ = model.forward_gnn_features(eval_batched_next_obs)
                    term_prob_eval = model.get_termination(next_global_f_eval, current_eval_option).squeeze()
                    eval_option_terminated = (term_prob_eval.item() > 0.5) # Simple threshold
                else:
                    eval_option_terminated = True # Terminate if env is done

            eval_total_reward += eval_ep_reward
            if frames :
                 from vmas.simulator.utils import save_video # Assuming this utility
                 save_video(f"{log_dir_oc}/eval_E{epoch+1}", frames, fps=15)

            eval_env.close()

        avg_eval_reward = eval_total_reward / eval_episodes
        writer.add_scalar("Evaluation/AverageReward", avg_eval_reward, epoch)
        print(f"Epoch {epoch+1} Evaluation: Avg Reward: {avg_eval_reward:.2f}")

        if avg_eval_reward > best_eval_reward_oc:
            best_eval_reward_oc = avg_eval_reward
            torch.save(model.state_dict(), output_policy_filename)
            print(f"New best Option-Critic model saved with eval reward: {best_eval_reward_oc:.2f}")

writer.close()
print("Option-Critic training finished.")