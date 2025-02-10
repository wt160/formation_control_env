import random
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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your environment wrapper
class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents, env_type=None, is_evaluation_mode=False, is_imitation=False):
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
    def step(self, actions):
        # actions: [num_envs, n_agents, action_dim]
        # Convert actions to list per agent
        actions_list = [actions[:, i, :] for i in range(self.n_agents)]  # List of tensors per agent
        obs, rewards, dones, infos = self.env.step(actions_list)
        # print("obs:{}".format(obs))
        # Convert lists to tensors
        obs = obs[0]
        # self.env.render(
        #                 mode="rgb_array",
        #                 agent_index_focus=None,  # Can give the camera an agent index to focus on
        #                 visualize_when_rgb=False,
        #             )
        # # print("rewards:{}".format(rewards))
        # input("1")
        # obs = torch.stack(obs, dim=1).to(self.device)  # [num_envs, n_agents, obs_dim]
        rewards = torch.stack(rewards, dim=1).to(self.device)  # [num_envs, n_agents]
        # print("rewards shape:{}".format(rewards.shape))
        # dones = torch.stack(dones, dim=1).to(self.device)  # [num_envs, n_agents]
        summed_rewards = rewards.sum(dim=1)
        return obs, summed_rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=False,
                    )
class GNNBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNBackbone, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True,
                             edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True,
                             edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True,
                             edge_dim=1, fill_value='mean')

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1) if edge_attr is not None else None)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)
        return x
class GATActor(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim, num_agents):
        super(GATActor, self).__init__()
        self.num_agents = num_agents
        self.hidden_size = hidden_channels*2
        # GAT layers
        # self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        # self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        # self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.gnn = GNNBackbone(in_channels, hidden_channels)
        # Global pooling layer
        self.pool = global_mean_pool

        self.gru = nn.GRU(input_size=hidden_channels * 16, hidden_size=hidden_channels*2, num_layers=1, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_channels, action_dim)


        
        # self.log_std = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.log_std = nn.Parameter(torch.ones(1, 1, action_dim) * -1.6) 
    def forward(self, data, hidden_state):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        print("x:{}".format(x))
        x = self.gnn(x, edge_index, edge_attr)

        # x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        # x = torch.relu(x)
        # x = self.conv2(x, edge_index, edge_attr)
        # x = torch.relu(x)
        # x = self.conv3(x, edge_index, edge_attr)
        # x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  # Shape: [batch_size, hidden_channels * 8]

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, data.batch, data.num_graphs)

        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)  # Shape: [batch_size * num_agents, feature_dim]

        # Reshape for GRU: [batch_size, num_agents, feature_dim]
        combined = combined.view(data.num_graphs, self.num_agents, -1)

        # Pass through GRU
        gru_out, hidden_state = self.gru(combined, hidden_state)  # gru_out shape: [batch_size, num_agents, hidden_size]

        # Reshape for actor head: [batch_size * num_agents, hidden_size]
        gru_out = gru_out.reshape(-1, self.hidden_size)

        # Actor head
        fc1_out = self.fc1(gru_out)
        action_mean = self.fc2(fc1_out)  # Shape: [batch_size, sequence_length, num_agents, out_channels]
        # action_mean = self.fc(gru_out)
        action_mean = action_mean.view(data.num_graphs, self.num_agents, -1)
        print("action_mean without noise:{}".format(action_mean))
        # Clamp log_std
        log_std_clamped = torch.clamp(self.log_std, min=-20, max=2)
        action_std = torch.exp(log_std_clamped).expand_as(action_mean)

        return action_mean, action_std, hidden_state

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
        # self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        # self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        # self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.gnn = GNNBackbone(in_channels, hidden_channels)
        # Global pooling layer
        self.pool = global_mean_pool

        # Critic network (value head)
        self.critic_fc1 = nn.Linear(hidden_channels * 8, hidden_channels * 4)
        self.critic_fc2 = nn.Linear(hidden_channels * 4, 1)  # Outputs a scalar value for each agent

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.gnn(x, edge_index, edge_attr)
        

        # Global graph embedding
        graph_embedding = self.pool(x, data.batch)  # Shape: [batch_size, hidden_channels * 8]

        # Critic head
        critic_hidden = torch.relu(self.critic_fc1(graph_embedding))
        state_value = self.critic_fc2(critic_hidden)
        return state_value

# Initialize the model
num_agents = 5
in_channels = 4  # Adjust based on your observation space
hidden_dim = 64
action_dim = 3  # Adjust based on your action space

def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# Apply weight initialization to the critic layers only


# Initialize the models
actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
critic_model = GATCritic(in_channels, hidden_dim, num_agents).to(device)

# Load the pre-trained actor network weights
# pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
# actor_model.load_state_dict(pretrained_weights)
pretrained_weights = torch.load('best_il_model.pth', map_location=device)
# pretrained_weights = torch.load('best_ppo_model.pth', map_location=device)

print("pretrained_weights:{}".format(pretrained_weights.keys()))

print("actor_model dict:{}".format(actor_model.state_dict().keys()))
# input("1")
# Filter out keys not present in the actor_model
pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in actor_model.state_dict()}
# actor_model.load_state_dict(pretrained_weights)
# pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
print("pretrained_weights after:{}".format(pretrained_weights.keys()))

actor_model.load_state_dict(pretrained_weights, strict=False)
# Initialize the critic network
critic_model.apply(initialize_weights)
# model = GATActorCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
from datetime import datetime

# Create a unique log directory with a timestamp
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/ppo_training_{current_time}'

# Initialize the SummaryWriter with the unique log directory
writer = SummaryWriter(log_dir=log_dir)

# model.load_state_dict(model_state_dict)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic_model.parameters(), lr=3e-4)


# PPO Hyperparameters
num_epochs = 10000
num_agents = 5
steps_per_epoch = 400
gamma = 0.99
lam = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_loss_coef = 0.005
max_grad_norm = 0.5
ppo_epochs = 10
mini_batch_size = 100

def compute_returns_and_advantages(rewards, masks, values, gamma, lam):
    advantages = torch.zeros_like(rewards).to(device)
    returns = torch.zeros_like(rewards).to(device)
    gae = 0
    # print("advantage shaoe:{}".format(advantages.shape))
    for step in reversed(range(len(rewards))):
        # print("values[step+1] shape:{}".format(values[step+1].shape))
        # print("mask[step] shape:{}".format(masks[step].shape))
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages[step] = gae
        returns[step] = advantages[step] + values[step]
    return returns, advantages

warm_up_epochs=5
ep_rewards = []
best_avg_reward = float('-inf')
best_evaluation_reward = float('-inf')
for epoch in range(num_epochs):
    actor_model.train()
    critic_model.train()
    # obs = env.reset()  # [num_envs, n_agents, obs_dim]
    env = VMASWrapper(
        scenario_name="formation_control_teacher_graph_obs",
        num_envs=10,
        device=device,
        continuous_actions=True,
        n_agents=num_agents,
        env_type="narrow",
        is_imitation=False,
    )
    obs = env.get_obs()  # [num_envs, n_agents, obs_dim]
    print("obs:{}".format(obs))
    # env.render()
    # input("1")
    print("reset obs device:{}".format(obs[0].x.device))
    # Initialize storage

    hidden_state = torch.zeros(1, env.num_envs, actor_model.hidden_size).to(device)
    obs_storage = []
    actions_storage = []
    log_probs_storage = []
    rewards_storage = []
    dones_storage = []
    values_storage = []
    hidden_states_storage = []
    epoch_rewards = []

    for step in range(steps_per_epoch):
        # print("obs:{}".format(obs))
        # print("obs[0]:{}".format(obs[0].x))
        batch_size = len(obs)
        n_agents = num_agents
        # obs_dim = obs.shape[2]
        print("step:{}".format(step))
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
            # print("x:{}".format(batch_obs.x))
            # batch_obs = batch_obs.to(device)
            # print("hidden_state shape:{}".format(hidden_state.shape))
            action_mean, action_std, next_hidden_state = actor_model(batch_obs, hidden_state)
            # print("action_mean:{}".format(action_mean))
            # input("1")
            # action_mean, action_std = actor_model(batch_obs)  # Now returns action_std
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # print("batch_obs:{}".format(batch_obs))
            # action_mean, state_value = model(batch_obs)
            # action_mean = actor_model(batch_obs)
            state_value = critic_model(batch_obs)
            # print("action_mean:{}".format(action_mean))

        # Reshape actions for the environment
        action_env = action.view(batch_size, n_agents, -1).to(device)

        next_obs, rewards, dones, infos = env.step(action_env)
        # env.render()
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_obs = [data.to(device) for data in next_obs]
        # print("rewards device:{}".format(rewards.device))
        mean_rewards = rewards.mean().item()
        
        epoch_rewards.append(mean_rewards)
    
        obs_storage.append(obs)
        actions_storage.append(action_env)
        log_probs_storage.append(log_prob.view(batch_size, n_agents))
        rewards_storage.append(rewards)
        dones_storage.append(dones)
        values_storage.append(state_value.squeeze(dim=1))
        hidden_states_storage.append(hidden_state)
        obs = next_obs
        hidden_state = next_hidden_state.detach()
        writer.add_scalar('Policy/std', action_std.mean().item(), epoch * steps_per_epoch + step)


    avg_reward = np.mean(epoch_rewards)
    ep_rewards.append(avg_reward)
    writer.add_scalar('Reward/avg_reward', avg_reward, epoch)

    # if avg_reward > best_avg_reward:
    #     best_avg_reward = avg_reward
    #     # **Save the model**
    #     torch.save(actor_model.state_dict(), 'best_ppo_model.pth')
    #     print(f'New best model saved with avg_reward: {avg_reward:.4f}')
    # Convert storage to tensors
    # obs_storage = torch.stack(obs_storage)  # [steps_per_epoch, num_envs, n_agents, obs_dim]
    actions_storage = torch.stack(actions_storage)  # [steps_per_epoch, num_envs, n_agents, action_dim]
    log_probs_storage = torch.stack(log_probs_storage)  # [steps_per_epoch, num_envs, n_agents]
    rewards_storage = torch.stack(rewards_storage)  # [steps_per_epoch, num_envs, n_agents]
    dones_storage = torch.stack(dones_storage)  # [steps_per_epoch, num_envs, n_agents]
    # print("obs list length:{}".format(len(obs_storage)))
    # print("value list length:{}".format(len(values_storage)))
    values_storage = torch.stack(values_storage)  # [steps_per_epoch, num_envs, n_agents]

    # Compute returns and advantages
    with torch.no_grad():
        # Get the last value
        batch_size = len(obs)
        n_agents = num_agents

        batch_obs = Batch.from_data_list(obs).to(device)
        # batch_obs = batch_obs.to(device)
        next_value = critic_model(batch_obs)
        # print("next_value:{}".format(next_value))

        next_value = next_value.squeeze(dim=1)
        values_storage = torch.cat([values_storage, next_value.unsqueeze(0)], dim=0)  # [steps+1, num_envs, n_agents]
        # print("values storeage shape:{}".format(values_storage.shape))
        
        # print("dones_storage shape:{}".format(dones_storage.shape))
        # print("rewards shpae:{}".format(len(rewards_storage)))
        # print("dones_storage:{}".format(dones_storage))
        returns_batch, advantages_batch = compute_returns_and_advantages(
            rewards_storage,
            1 - dones_storage.float(),  # masks
            values_storage,
            gamma,
            lam
        )


    # Normalize advantages
    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
    # print("advantage_batch origin shape:{}".format(advantages_batch.shape))

    # PPO update
    num_samples = len(obs_storage)
    indices = np.arange(num_samples)

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end].tolist()

            # Prepare mini-batch
            # print("mb_idx:{}".format(mb_idx))
            obs_mb = [obs_storage[i] for i in mb_idx]
            # obs_mb = obs_storage[mb_idx]  # [batch_size, n_agents, obs_dim]
            actions_mb = actions_storage[mb_idx, :, :, :]
            # print("actions_mb shape:{}".format(actions_mb.shape))
            log_probs_old_mb = log_probs_storage[mb_idx]
            returns_mb = returns_batch[mb_idx]
            # print("advantage_batch:{}".format(advantages_batch.shape))
            advantages_mb = advantages_batch[mb_idx]
            hidden_states_mb = [hidden_states_storage[i] for i in mb_idx]
            # batch_size_mb = obs_mb.shape[0]
            # print("obs_mb:{}".format(obs_mb))
            # Prepare observations for GNN
            # obs_flat_mb = obs_mb.view(batch_size_mb * n_agents, obs_dim)

            # Create PyG data list
            # data_list_mb = []
            # for i in range(len(obs_mb)):
                # data = Data(x=obs_flat_mb[i * n_agents:(i + 1) * n_agents], edge_index=edge_index, edge_attr=edge_attr)
                # data_list_mb.append(data)
            # print("obs_mb:{}".format(obs_mb))
            # obs_mb_flat = [item for sublist in obs_mb for item in sublist]
            obs_mb_flat = [item.to(device) for sublist in obs_mb for item in sublist]

            for single in obs_mb_flat:
                single = single.to(device)
            # print(obs_mb_flat)  # This will be a single flattened list
            data_list_mb = Batch.from_data_list(obs_mb_flat).to(device)
            # data_list_mb = data_list_mb.to(device)
            # Forward pass
            hidden_states_mb = torch.stack(hidden_states_mb, dim=0)  # Shape: [batch_size, layers, num_envs, num_agents, hidden_size]
        # Reshape hidden states
            hidden_states_mb = hidden_states_mb.view(1, -1, actor_model.hidden_size)

            # action_mean_mb = actor_model(data_list_mb)
            action_mean_mb, action_std_mb ,_= actor_model(data_list_mb, hidden_states_mb)
            state_value_mb = critic_model(data_list_mb)

            # print("action_mean_mb shape:{}".format(action_mean_mb.shape))  # [30, 5, 3]
            # action_std_mb = torch.ones_like(action_mean_mb) * 0.1
            # print("log_probs_old_mb shape:{}".format(log_probs_old_mb.shape))  # [10, 3, 5]
            dist_mb = torch.distributions.Normal(action_mean_mb, action_std_mb)
            actions_mb = actions_mb.view(mini_batch_size*batch_size, num_agents, action_dim)
            log_probs_new_mb = dist_mb.log_prob(actions_mb).sum(dim=-1, keepdim=True).squeeze(dim=-1)
            entropy_mb = dist_mb.entropy().sum(dim=-1).view(-1, n_agents)
            # print("log_probs_new_mb shape:{}".format(log_probs_new_mb.shape))
            # Ratio
            log_probs_old_mb = log_probs_old_mb.view(mini_batch_size*batch_size, num_agents)
            log_probs_new_mb_sum = log_probs_new_mb.sum(dim=1)  # Shape: [batch_size, 1]
            log_probs_old_mb_sum = log_probs_old_mb.sum(dim=1)  # Shape: [batch_size, 1]
            # ratio = torch.exp(log_probs_new_mb - log_probs_old_mb)
            ratio = torch.exp(log_probs_new_mb_sum - log_probs_old_mb_sum)  # Shape: [batch_size, 1]
            # Surrogate loss
            # print("ratio shape:{}".format(ratio.shape))
            # print("advantage_mb shape:{}".format(advantages_mb.shape))
            advantages_mb = advantages_mb.view(mini_batch_size*batch_size)
            surr1 = ratio * advantages_mb.squeeze(-1)  # Remove extra dimension if necessary
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb.squeeze(-1)
            actor_loss = -torch.min(surr1, surr2).mean()
            # surr1 = ratio * advantages_mb
            # surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb
            # actor_loss = -torch.min(surr1, surr2).mean()
            # print("return_mb shape:{}".format(returns_mb.shape))
            # print("state_value_mb shape:{}".format(state_value_mb.shape))
            # Value loss
            value_loss = nn.functional.mse_loss(state_value_mb.squeeze(dim=1), returns_mb.view(mini_batch_size*batch_size))

            # Entropy loss
            entropy_loss = entropy_mb.mean()
            total_loss = actor_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss

            if epoch < warm_up_epochs:
                # Train the critic network only
                critic_optimizer.zero_grad()
                value_loss.backward()
                critic_optimizer.step()
                writer.add_scalar('Loss/critic_loss', value_loss.item(), epoch)
            else:
                # Standard PPO training
                # Combine actor and critic losses
                # loss = actor_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss.backward()
                value_loss.backward()
                # total_loss.backward()
                nn.utils.clip_grad_norm_(actor_model.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)

                actor_optimizer.step()
                critic_optimizer.step()
                writer.add_scalar('Loss/actor_loss', actor_loss.item(), epoch)
                writer.add_scalar('Loss/critic_loss', value_loss.item(), epoch)
                writer.add_scalar('Loss/entropy_loss', entropy_loss.item(), epoch)
                writer.add_scalar('Loss/total_loss', total_loss.item(), epoch)
            # Total loss
            # loss = actor_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss

            # # Backpropagation
            # optimizer.zero_grad()
            # loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # optimizer.step()

    # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    
    if epoch > warm_up_epochs:
        print(f'Epoch {epoch + 1}/{num_epochs}, actor Loss: {actor_loss.item():.4f}, Avg Reward: {avg_reward:.4f}')
    else:
        print(f'Epoch {epoch + 1}/{num_epochs}, critic Loss: {value_loss.item():.4f}, Avg Reward: {avg_reward:.4f}')
    from vmas.simulator.utils import save_video
    if epoch % 10 == 0:

        frame_list = []
        env = VMASWrapper(
        scenario_name="formation_control_teacher_graph_obs",
        num_envs=10,
        device=device,
        continuous_actions=True,
        n_agents=num_agents,
        env_type="narrow",
        is_evaluation_mode=True,
        is_imitation=False,
        )
        obs = env.get_obs()  # [num_envs, n_agents, obs_dim]

        print("reset obs device:{}".format(obs[0].x.device))
        # Initialize storage
        hidden_state = torch.zeros(1, env.num_envs, actor_model.hidden_size).to(device)
        obs_storage = []
        actions_storage = []
        log_probs_storage = []
        rewards_storage = []
        dones_storage = []
        values_storage = []
        epoch_rewards = []

        for step in range(steps_per_epoch):
            # print("obs:{}".format(obs))
            # print("obs[0]:{}".format(obs[0].x))
            batch_size = len(obs)
            n_agents = num_agents
            # obs_dim = obs.shape[2]
            print("step:{}".format(step))
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
                action_mean, action_std, next_hidden_state = actor_model(batch_obs, hidden_state)  # Now returns action_std
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
            hidden_state = next_hidden_state.detach()
            # dones_storage.append(dones)
            # values_storage.append(state_value.squeeze(dim=1))
            obs = next_obs


        avg_reward = np.mean(epoch_rewards)
        ep_rewards.append(avg_reward)
        writer.add_scalar('Evaluation Reward/avg_reward', avg_reward, epoch)

        if avg_reward > best_evaluation_reward:
            best_evaluation_reward = avg_reward
            # **Save the model**
            torch.save(actor_model.state_dict(), 'best_ppo_model_7dim.pth')
            print(f'New best model saved with avg_reward: {best_evaluation_reward:.4f}')
        
            save_video("ppo_training_{}_{}".format(current_time, epoch), frame_list, fps=1 / 0.1)
        if avg_reward < -0.1:
            save_video("ppo_training_{}_{}_bad".format(current_time, epoch), frame_list, fps=1 / 0.1)
env.close()
writer.close()