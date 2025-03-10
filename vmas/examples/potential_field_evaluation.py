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


import sys

train_env_type = sys.argv[1]
policy_filename = sys.argv[2]
output_policy_filename = sys.argv[3]
steps_per_epoch = int(sys.argv[4])
# Set device
device = sys.argv[5]
noise_level = float(sys.argv[6])
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
seed = 42  # Set the seed you want
set_seed(seed)
# Define your environment wrapper
class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents, env_type=None, is_evaluation_mode=False, is_imitation=False, working_mode="imitation", evaluation_index=0, evaluation_noise = 0.0):
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
            evaluation_noise = evaluation_noise,
        )
        self.device = device
        self.n_agents = n_agents
        self.num_envs = num_envs

    def reset(self):
        obs = self.env.reset()
        # print("obs length:{}".format(len(obs[0])))
        obs = obs[0]
        # obs is a list of observations per agent
        # Stack observations to shape [num_envs, n_agents, obs_dim]
        # obs = torch.stack(obs, dim=1).to(self.device)
        return obs

    def get_obs(self):
        obs = self.env.get_obs()
        obs = obs[0]
        return obs

    def step(self, actions, done_override=None):
        # actions: [num_envs, n_agents, action_dim]
        # done_override: [num_envs] tensor indicating if done should be set
        actions_list = [actions[:, i, :] for i in range(self.n_agents)]  # List of tensors per agent
        obs, rewards, dones, infos = self.env.step(actions_list)
        obs = obs[0]
        rewards = torch.stack(rewards, dim=1).to(self.device)  # [num_envs, n_agents]
        # dones = torch.stack(dones, dim=1).to(self.device)  # [num_envs, n_agents]
        
        # Sum rewards across agents
        summed_rewards = rewards.sum(dim=1)  # [num_envs]
        
        # If done_override is provided, set done flags accordingly
        if done_override is not None:
            dones = dones | done_override.unsqueeze(1)  # Broadcast to [num_envs, n_agents]
        # print("dones:{}".format(dones))
        return obs, summed_rewards, dones, infos
    # def step(self, actions):
    #     # actions: [num_envs, n_agents, action_dim]
    #     # Convert actions to list per agent
    #     actions_list = [actions[:, i, :] for i in range(self.n_agents)]  # List of tensors per agent
    #     obs, rewards, dones, infos = self.env.step(actions_list)
    #     # print("obs:{}".format(obs))
    #     # Convert lists to tensors
    #     obs = obs[0]
    #     # self.env.render(
    #     #                 mode="rgb_array",
    #     #                 agent_index_focus=None,  # Can give the camera an agent index to focus on
    #     #                 visualize_when_rgb=False,
    #     #             )
    #     # # print("rewards:{}".format(rewards))
    #     # input("1")
    #     # obs = torch.stack(obs, dim=1).to(self.device)  # [num_envs, n_agents, obs_dim]
    #     rewards = torch.stack(rewards, dim=1).to(self.device)  # [num_envs, n_agents]
    #     # print("rewards shape:{}".format(rewards.shape))
    #     # dones = torch.stack(dones, dim=1).to(self.device)  # [num_envs, n_agents]
    #     summed_rewards = rewards.sum(dim=1)
    #     return obs, summed_rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=False,
                    )
class GATActor(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents):
        super(GATActor, self).__init__()
        self.num_agents = num_agents

        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

        # Global pooling layer
        self.pool = global_mean_pool

        # Actor network (policy head)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16, hidden_channels * 4),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_channels * 4, action_dim)
        # self.log_std = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.log_std = nn.Parameter(torch.ones(1, 1, action_dim) * -2.2) 
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # print("x:{}".format(x))
        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
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

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)

        # Actor head
        actor_hidden = self.fc1(combined)
        action_mean = self.fc2(actor_hidden)
        action_mean = action_mean.view(data.num_graphs, self.num_agents, -1)

        action_std = torch.exp(self.log_std).expand_as(action_mean)
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
in_channels = 7  # Adjust based on your observation space
hidden_dim = 64
action_dim = 3  # Adjust based on your action space

def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# Apply weight initialization to the critic layers only

num_experts = 2  # Number of expert classes
expert_gate_net = ExpertGateNet(in_channels=in_channels, hidden_channels=hidden_dim, num_experts=num_experts).to(device)

# Initialize the models
clutter_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
critic_model = GATCritic(in_channels, hidden_dim, num_agents).to(device)

free_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
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

clutter_pretrained_weights = {k: v for k, v in clutter_pretrained_weights.items() if k in clutter_actor_model.state_dict()}
# actor_model.load_state_dict(pretrained_weights)
# pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
# free_actor_model.load_state_dict(free_pretrained_weights, strict=False)

clutter_actor_model.load_state_dict(clutter_pretrained_weights, strict=False)
# expert_gate_net.load_state_dict(expert_choose_net_weights, strict=False)
# Initialize the critic network
critic_model.apply(initialize_weights)
# model = GATActorCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
from datetime import datetime

# Create a unique log directory with a timestamp
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/ppo_eva_{current_time}_{train_env_type}_{str(noise_level)}'

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
epoch_restart_num = 5
gamma = 0.99
lam = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_loss_coef = 0.005
max_grad_norm = 0.2
ppo_epochs = 10
mini_batch_size = 800



from vmas.simulator.utils import save_video

ep_rewards = []
best_avg_reward = float('-inf')
best_evaluation_reward = float('-inf')

eval_epoch_restart_num = 10
obs_storage = []
actions_storage = []
log_probs_storage = []
rewards_storage = []
dones_storage = []
values_storage = []
epoch_rewards = []
total_no_collision_num = 0
total_connection_num = 0
for epoch_restart in range(eval_epoch_restart_num):
    env = VMASWrapper(
        scenario_name="formation_control_teacher_graph_obs_cuda1",
        num_envs=1,
        device=device,
        continuous_actions=True,
        n_agents=num_agents,
        env_type=train_env_type,
        is_evaluation_mode=True,
        is_imitation=False,
        working_mode="potential_field",
        evaluation_index=epoch_restart,
        evaluation_noise =noise_level,
        )
    obs = env.get_obs() # [num_envs, n_agents, obs_dim]
    # env.render()
    # input("1")
    step_counters = torch.zeros(env.num_envs, device=device)
    frame_list = []
    # [num_envs, n_agents, obs_dim]

    # print("reset obs device:{}".format(obs[0].x.device))
    # Initialize storage
    
    
    for step in range(steps_per_epoch):
        # print("obs:{}".format(obs))
        # print("obs[0]:{}".format(obs[0].x))
        batch_size = len(obs)
        n_agents = num_agents
        # obs_dim = obs.shape[2]
        # print("step:{}".format(step))
        # Prepare observations for GNN


        # Forward pass through the policy
        with torch.no_grad():
            # print("obs list size:{}".format(len(obs)))
            # print("obs:{}".format(obs[0]))
            # print("obs edge_attr edvice:{}".format(obs[0].edge_attr.device))
            # print("obs edge_index deviuce:{}".format(obs[0].edge_index.device))
            batch_obs = Batch.from_data_list(obs).to(device)
            # print()
            # print("batch_obs device:{}".format(batch_obs))
            # batch_obs = batch_obs.to(device)
            action_mean, action_std = clutter_actor_model(batch_obs)  # Now returns action_std
            # dist = torch.distributions.Normal(action_mean, action_std)
            # action = dist.sample()
            # log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # print("batch_obs:{}".format(batch_obs))
            # action_mean, state_value = model(batch_obs)
            # action_mean = actor_model(batch_obs)
            # state_value = critic_model(batch_obs)
            # print("action_mean:{}".format(action_mean))

        # Reshape actions for the environment
        action_env = action_mean.view(batch_size, n_agents, -1).to(device)

        next_obs, rewards, dones, infos = env.step(action_env)
        rewards = rewards.to(device)

        # agent_collision_rew = infos[]
        dones = dones.to(device)
        next_obs = [data.to(device) for data in next_obs]
        # print("rewards device:{}".format(rewards.device))
        frame = env.render()
        # print("frammmme:{}".format(frame))
        frame_list.append(frame)
        mean_rewards = rewards.mean().item()
        
        epoch_rewards.append(mean_rewards)
    
        # obs_storage.append(obs)
        # actions_storage.append(action_env)
        # log_probs_storage.append(log_prob.view(batch_size, n_agents))
        rewards_storage.append(rewards)
        # dones_storage.append(dones)
        # values_storage.append(state_value.squeeze(dim=1))
    
        agent_info = infos[0]
        
        # 1 means in collision, 0 means no collision
        no_collision_num = torch.sum(agent_info["eva_collision_num"] == 0)
        total_no_collision_num += no_collision_num
            # 1 means connected, 0 means not connected
        connection_num = torch.sum(agent_info["eva_connection_num"] == 1)
        # print("connection_num:{}".format(connection_num))
        # if connection_num < 4:
            # print("agent_connection:{}".format(agent_info["eva_connection_num"]))
        total_connection_num += connection_num
        obs = next_obs
    # save_video("ppo_training_{}_{}".format(current_time, epoch), frame_list, fps=1 / 0.1)
    save_video("eva_video/potential_field_training_{}_{}_{}_{}".format(current_time, epoch_restart, train_env_type, noise_level), frame_list, fps=1 / 0.1)

avg_reward = np.mean(epoch_rewards)
ep_rewards.append(avg_reward)
writer.add_scalar('Evaluation Reward/avg_reward', avg_reward, 0)

total_no_collision_rate = total_no_collision_num / (eval_epoch_restart_num*(num_agents-1)*steps_per_epoch )
total_connection_rate = total_connection_num / (eval_epoch_restart_num*(num_agents-1)*steps_per_epoch )
print("{}, noise level:{}, connection:{}, no_collision:{}".format(train_env_type, noise_level, total_connection_rate, total_no_collision_rate))
print("collision num:{}, no connection:{}".format((eval_epoch_restart_num*(num_agents-1)*steps_per_epoch) - total_no_collision_num, (eval_epoch_restart_num*(num_agents-1)*steps_per_epoch ) - total_connection_num  ))
writer.add_scalar('Evaluation Metric/no_collision_rate', total_no_collision_rate, 0)
writer.add_scalar('Evaluation Metric/connection_rate', total_connection_rate, 0)

# if avg_reward > best_evaluation_reward:
#     best_evaluation_reward = avg_reward
#     # **Save the model**
#     torch.save(clutter_actor_model.state_dict(), output_policy_filename)
#     print(f'New best model saved with avg_reward: {best_evaluation_reward:.4f}')

#     save_video("best_ppo_training_{}_{}".format(current_time, 0), frame_list, fps=1 / 0.1)
# if avg_reward < -0.1:
#     save_video("ppo_training_{}_{}_bad".format(current_time, 0), frame_list, fps=1 / 0.1)
# env.close()
writer.close()