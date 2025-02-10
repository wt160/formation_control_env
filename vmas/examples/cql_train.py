import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

# Import your VMAS environment
from vmas import make_env

# Set device to use the second GPU (assuming devices are indexed starting from 0)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Define your environment wrapper
class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents):
        self.env = make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            dict_spaces=False,
            n_agents=n_agents,
            wrapper=None,
            seed=None,
        )
        self.device = device
        self.n_agents = n_agents
        self.num_envs = num_envs

    def reset(self):
        obs = self.env.reset()
        obs = obs[0]
        return obs

    def step(self, actions):
        # actions: [num_envs, n_agents, action_dim]
        actions_list = [actions[:, i, :] for i in range(self.n_agents)]  # List of tensors per agent
        obs, rewards, dones, infos = self.env.step(actions_list)
        obs = obs[0]
        rewards = torch.stack(rewards, dim=1).to(self.device)  # [num_envs, n_agents]
        summed_rewards = rewards.sum(dim=1)
        return obs, summed_rewards, dones, infos

    def close(self):
        self.env.close()

# Define the Actor network with a learnable log_std for action variance
class GATActor(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents):
        super(GATActor, self).__init__()
        self.num_agents = num_agents

        # GAT layers
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean', add_self_loops=False
        )
        self.conv2 = GATConv(
            hidden_channels * 8, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean'
        )
        self.conv3 = GATConv(
            hidden_channels * 8, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean'
        )

        # Global pooling layer
        self.pool = global_mean_pool

        # Actor network (policy head)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16, hidden_channels * 4),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_channels * 4, action_dim)
        self.fc_log_std = nn.Linear(hidden_channels * 4, action_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, data.batch, data.num_graphs)

        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)

        # Actor head
        actor_hidden = self.fc1(combined)
        action_mean = self.fc_mean(actor_hidden)
        log_std = self.fc_log_std(actor_hidden)
        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, min=-20, max=2)
        action_std = torch.exp(log_std)

        # Reshape to [batch_size, num_agents, action_dim]
        action_mean = action_mean.view(-1, self.num_agents, action_mean.size(-1))
        action_std = action_std.view(-1, self.num_agents, action_std.size(-1))

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

# Define the Critic network (Q-function)
class GATCritic(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents):
        super(GATCritic, self).__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim

        # GAT layers
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean', add_self_loops=False
        )
        self.conv2 = GATConv(
            hidden_channels * 8, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean'
        )
        self.conv3 = GATConv(
            hidden_channels * 8, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean'
        )

        # Calculate the correct input dimension
        self.hidden_dim = hidden_channels  # For clarity
        # Each agent embedding: hidden_dim * 8
        # Total agent embeddings: num_agents * hidden_dim * 8
        # Graph embedding: hidden_dim * 8
        # Actions: num_agents * action_dim
        self.input_dim = self.hidden_dim * 8 * (self.num_agents + 1) + self.num_agents * self.action_dim

        # Critic network (Q-function head)
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_channels * 4),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_channels * 4, 1)  # Outputs a scalar Q-value

    def forward(self, data, actions):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = global_mean_pool(x, data.batch)

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, data.batch, data.num_graphs)

        # Flatten agent embeddings
        agent_embeddings_flat = agent_embeddings.view(data.num_graphs, -1)  # Shape: [batch_size, num_agents * hidden_dim * 8]

        # Flatten actions
        actions_flat = actions.view(data.num_graphs, -1)  # Shape: [batch_size, num_agents * action_dim]

        # Concatenate agent embeddings, graph embedding, and actions
        combined = torch.cat([agent_embeddings_flat, graph_embedding, actions_flat], dim=1)  # Shape: [batch_size, input_dim]

        # Critic head
        critic_hidden = self.fc1(combined)
        q_value = self.fc2(critic_hidden)
        return q_value

    def extract_agent_embeddings(self, x, batch, batch_size):
        agent_node_indices = []
        for graph_idx in range(batch_size):
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)

        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]
        return agent_embeddings

# Initialize the models
num_agents = 5
in_channels = 4  # Adjust based on your observation space
hidden_dim = 64
action_dim = 3  # Adjust based on your action space

# Initialize the Actor and Critic models
actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
critic_model_1 = GATCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
critic_model_2 = GATCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)

# Initialize target networks for critics
target_critic_model_1 = GATCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
target_critic_model_2 = GATCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
target_critic_model_1.load_state_dict(critic_model_1.state_dict())
target_critic_model_2.load_state_dict(critic_model_2.state_dict())

# Load the pre-trained actor network weights (if available)
try:
    pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
    pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in actor_model.state_dict()}
    actor_model.load_state_dict(pretrained_weights, strict=False)
except FileNotFoundError:
    print("Pre-trained actor model not found. Starting from scratch.")

# Set up the optimizers
actor_optimizer = optim.Adam(actor_model.parameters(), lr=3e-4)
critic_optimizer_1 = optim.Adam(critic_model_1.parameters(), lr=3e-4)
critic_optimizer_2 = optim.Adam(critic_model_2.parameters(), lr=3e-4)

# Set up the environment
env = VMASWrapper(
    scenario_name="formation_control_teacher_graph_obs",
    num_envs=10,
    device=device,
    continuous_actions=True,
    n_agents=num_agents,
)

# CQL Hyperparameters
num_epochs = 1000
steps_per_epoch = 200
batch_size = 128  # Mini-batch size for updates
replay_buffer_size = 1000000
start_steps = 1000  # Number of steps to collect data with random policy before training
update_after = 1000  # Number of steps to start updates
update_every = 50  # Update every n steps
gamma = 0.99
tau = 0.005  # For soft update of target parameters

# CQL-specific hyperparameters
cql_alpha = 1.0  # Weight for CQL loss term
min_q_weight = 1.0  # Additional weight for conservative loss

# Automatic Entropy Adjustment
total_action_dim = action_dim * num_agents
target_entropy = -total_action_dim  # A common choice in SAC and CQL

log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=3e-4)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.storage = []
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        return map(list, zip(*batch))

replay_buffer = ReplayBuffer(replay_buffer_size)

# Initialize TensorBoard Writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/cql_training_{current_time}'
writer = SummaryWriter(log_dir=log_dir)

# Function to update target networks
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

# Training Loop
total_steps = num_epochs * steps_per_epoch
state = env.reset()
episode_reward = 0
episode_rewards = []
best_avg_reward = float('-inf')
global_step = 0

for epoch in range(num_epochs):
    actor_model.train()
    critic_model_1.train()
    critic_model_2.train()
    state = env.reset()
    episode_reward = 0

    for step in range(steps_per_epoch):
        global_step += 1
        batch_size_env = len(state)
        n_agents = num_agents

        # Prepare observations
        state_batch = Batch.from_data_list(state).to(device)

        # Select action according to policy
        with torch.no_grad():
            action_mean, action_std = actor_model(state_batch)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action = action.clamp(-1, 1)  # Assuming action space is normalized

        action_env = action.view(batch_size_env, n_agents, -1).to(device)

        # Step the environment
        next_state, reward, done, info = env.step(action_env)
        reward = reward.to(device)
        done = done.to(device)
        next_state = [data.to(device) for data in next_state]

        # Store transition in replay buffer
        replay_buffer.add((state, action_env, reward, next_state, done))

        state = next_state
        episode_reward += reward.mean().item()

        # Update CQL after collecting sufficient data
        if global_step > update_after and global_step % update_every == 0:
            for _ in range(update_every):
                # Sample a batch from replay buffer
                batch = replay_buffer.sample(batch_size)
                states_b, actions_b, rewards_b, next_states_b, dones_b = batch

                # Prepare batches
                states_b_flat = [item.to(device) for sublist in states_b for item in sublist]
                states_b_batch = Batch.from_data_list(states_b_flat).to(device)
                actions_b_tensor = torch.cat(actions_b).to(device)
                rewards_b_tensor = torch.cat(rewards_b).to(device).unsqueeze(1)
                next_states_b_flat = [item.to(device) for sublist in next_states_b for item in sublist]
                next_states_b_batch = Batch.from_data_list(next_states_b_flat).to(device)
                dones_b_tensor = torch.cat(dones_b).to(device).unsqueeze(1)

                # Compute target Q-value
                with torch.no_grad():
                    next_action_mean, next_action_std = actor_model(next_states_b_batch)
                    next_dist = torch.distributions.Normal(next_action_mean, next_action_std)
                    next_action = next_dist.sample()
                    next_action = next_action.clamp(-1, 1)

                    # Compute log probabilities
                    next_log_prob = next_dist.log_prob(next_action)
                    # Sum over action dimensions
                    next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)  # Shape: [batch_size, num_agents, 1]
                    # Sum over agents
                    next_log_prob = next_log_prob.sum(dim=1)  # Shape: [batch_size, 1]

                    # Target Q-values from target networks
                    target_q1 = target_critic_model_1(next_states_b_batch, next_action)
                    target_q2 = target_critic_model_2(next_states_b_batch, next_action)
                    target_q = torch.min(target_q1, target_q2) - log_alpha.exp() * next_log_prob
                    target_q = rewards_b_tensor + (1 - dones_b_tensor.float()) * gamma * target_q

                # Compute current Q estimates
                current_q1 = critic_model_1(states_b_batch, actions_b_tensor)
                current_q2 = critic_model_2(states_b_batch, actions_b_tensor)

                # Critic losses (MSE loss)
                critic_loss_1 = nn.functional.mse_loss(current_q1, target_q.detach())
                critic_loss_2 = nn.functional.mse_loss(current_q2, target_q.detach())

                # CQL Conservative Loss
                # Sample actions from policy (current states)
                action_mean_b, action_std_b = actor_model(states_b_batch)
                dist_b = torch.distributions.Normal(action_mean_b, action_std_b)
                sampled_actions = dist_b.rsample()
                sampled_actions = sampled_actions.clamp(-1, 1)

                # Compute log probabilities
                log_prob_b = dist_b.log_prob(sampled_actions)
                log_prob_b = log_prob_b.sum(dim=-1, keepdim=True)  # Shape: [batch_size, num_agents, 1]
                log_prob_b = log_prob_b.sum(dim=1)  # Shape: [batch_size, 1]

                # Compute Q-values for sampled actions
                current_q1_pi = critic_model_1(states_b_batch, sampled_actions)
                current_q2_pi = critic_model_2(states_b_batch, sampled_actions)

                # CQL Loss Terms
                cql_loss_1 = (current_q1_pi - log_prob_b).mean()
                cql_loss_2 = (current_q2_pi - log_prob_b).mean()

                # Total Critic Loss
                total_critic_loss_1 = critic_loss_1 + cql_alpha * (cql_loss_1 - current_q1.mean())
                total_critic_loss_2 = critic_loss_2 + cql_alpha * (cql_loss_2 - current_q2.mean())

                # Optimize critics
                critic_optimizer_1.zero_grad()
                total_critic_loss_1.backward()
                critic_optimizer_1.step()

                critic_optimizer_2.zero_grad()
                total_critic_loss_2.backward()
                critic_optimizer_2.step()

                # Delayed policy updates
                if global_step % 2 == 0:
                    # Compute actor loss
                    action_mean_b, action_std_b = actor_model(states_b_batch)
                    dist_b = torch.distributions.Normal(action_mean_b, action_std_b)
                    actions_b_sampled = dist_b.rsample()
                    actions_b_sampled = actions_b_sampled.clamp(-1, 1)

                    log_prob_b = dist_b.log_prob(actions_b_sampled)
                    log_prob_b = log_prob_b.sum(dim=-1, keepdim=True)  # Shape: [batch_size, num_agents, 1]
                    log_prob_b = log_prob_b.sum(dim=1)  # Shape: [batch_size, 1]

                    # Compute Q-values for actor update
                    q1_pi = critic_model_1(states_b_batch, actions_b_sampled)
                    q2_pi = critic_model_2(states_b_batch, actions_b_sampled)
                    min_q_pi = torch.min(q1_pi, q2_pi)

                    # Automatic Entropy Adjustment
                    alpha = log_alpha.exp()
                    actor_loss = (alpha * log_prob_b - min_q_pi).mean()

                    # Optimize actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Alpha Loss and Optimization
                    alpha_loss = -(log_alpha * (log_prob_b + target_entropy).detach()).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    # Update target networks
                    soft_update(target_critic_model_1, critic_model_1, tau)
                    soft_update(target_critic_model_2, critic_model_2, tau)

                    # Logging
                    writer.add_scalar('Loss/actor_loss', actor_loss.item(), global_step)
                    writer.add_scalar('Loss/critic_loss_1', total_critic_loss_1.item(), global_step)
                    writer.add_scalar('Loss/critic_loss_2', total_critic_loss_2.item(), global_step)
                    writer.add_scalar('Loss/alpha_loss', alpha_loss.item(), global_step)
                    writer.add_scalar('Policy/alpha', alpha.item(), global_step)
                    writer.add_scalar('Policy/entropy', -log_prob_b.mean().item(), global_step)

    # Logging average reward per epoch
    avg_reward = episode_reward / steps_per_epoch
    episode_rewards.append(avg_reward)
    writer.add_scalar('Reward/avg_reward', avg_reward, epoch)
    episode_reward = 0

    # Save best model
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(actor_model.state_dict(), 'best_cql_model.pth')
        print(f'New best model saved with avg_reward: {avg_reward:.4f}')

    print(f'Epoch {epoch + 1}/{num_epochs}, Avg Reward: {avg_reward:.4f}')

# Close environment and writer
env.close()
writer.close()
