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
    


class LaserFC_Actor(nn.Module):
    def __init__(self, 
                 laser_dim=360,       # 一维激光数据维度
                 num_agents=5,        # 机器人数量（含领队）
                 relpos_dim=3,        # 相对位姿维度(x,y,theta)
                 hidden_dim=128, 
                 action_dim=3):       # 输出SE2位姿
        super(LaserFC_Actor, self).__init__()
        self.num_agents = num_agents
        
        # 激光处理分支 (1D CNN)
        self.laser_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # 自适应池化统一维度
            nn.Flatten()
        )
        
        # 自动计算激光分支输出维度
        with torch.no_grad():
            dummy_laser = torch.zeros(1, 1, laser_dim)
            self.conv_out_dim = self.laser_conv(dummy_laser).shape[1]
        
        # 相对位置处理分支
        self.pos_encoder = nn.Sequential(
            nn.Linear(relpos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 联合特征处理
        self.joint_net = nn.Sequential(
            nn.Linear(self.conv_out_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出层
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # 可训练的标准差参数
        self.log_std = nn.Parameter(torch.ones(1, num_agents, action_dim) * -2.2)

    def forward(self, laser_input, relative_pos):
        """
        Args:
            laser_input: (batch_size, 1, laser_dim) 的一维激光数据
            relative_pos: (batch_size, num_agents, 3) 的机器人相对位姿
        """
        batch_size = laser_input.size(0)
        
        # 处理激光输入
        laser_feat = self.laser_conv(laser_input)  # (batch_size, conv_out_dim)
        # print("relative_pos shape:{}".format(relative_pos.shape))
        # 处理相对位置
        pos_feat = self.pos_encoder(relative_pos.view(-1, 3))  # (batch*num_agents, 64)
        pos_feat = pos_feat.view(batch_size, self.num_agents , -1)  # (batch, num_agents, 64)
        # print("pos_feat shape:{}".format(pos_feat.shape))
        
        # 扩展激光特征并与位置特征拼接
        laser_feat = laser_feat.unsqueeze(1).expand(-1, self.num_agents, -1)  # (batch, num_agents, conv_out_dim)
        # print("laser_feat shape:{}".format(laser_feat.shape))
        combined = torch.cat([laser_feat, pos_feat], dim=-1)  # (batch, num_agents, conv_out_dim+64)
        
        # 联合特征处理
        hidden = self.joint_net(combined)
        action_mean = self.action_head(hidden)  # (batch, num_agents, action_dim)
        
        # 计算标准差
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        return action_mean, action_std


class LaserFC_Critic(nn.Module):
    def __init__(self,
                 laser_dim=360,
                 num_agents=5,
                 relpos_dim=3,
                 hidden_dim=128):
        super(LaserFC_Critic, self).__init__()
        self.num_agents = num_agents

        # Shared feature extractors with actor
        self.laser_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )

        # Automatically compute conv output dimension
        with torch.no_grad():
            dummy_laser = torch.zeros(1, 1, laser_dim)
            self.conv_out_dim = self.laser_conv(dummy_laser).shape[1]

        # Position encoder with same architecture
        self.pos_encoder = nn.Sequential(
            nn.Linear(relpos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Joint processing network
        self.joint_net = nn.Sequential(
            nn.Linear(self.conv_out_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value head with agent-wise attention aggregation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Simple attention layer for agent aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, laser_input, relative_pos):
        batch_size = laser_input.size(0)
        
        # Shared feature extraction
        laser_feat = self.laser_conv(laser_input)
        pos_feat = self.pos_encoder(relative_pos.view(-1, 3))
        pos_feat = pos_feat.view(batch_size, self.num_agents, -1)
        
        # Feature combination
        laser_feat = laser_feat.unsqueeze(1).expand(-1, self.num_agents, -1)
        combined = torch.cat([laser_feat, pos_feat], dim=-1)
        
        # Joint processing
        hidden = self.joint_net(combined)
        
        # Agent-wise attention aggregation
        attention_weights = self.attention(hidden)  # (batch, num_agents, 1)
        context = torch.sum(attention_weights * hidden, dim=1)  # (batch, hidden_dim)
        
        # Final value estimation
        value = self.value_head(context).squeeze(-1)  # (batch,)
        
        return value




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
laser_dim = 90    # 假设激光是360维的1D数组（模拟360度激光扫描）
num_agents = 5     # 1个领队 + 4个跟随者

# 初始化网络
conv_actor_net = LaserFC_Actor(laser_dim=laser_dim, num_agents=num_agents).to(device)
conv_critic_net = LaserFC_Critic(laser_dim=laser_dim, num_agents=num_agents).to(device)

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

clutter_pretrained_weights = {k: v for k, v in clutter_pretrained_weights.items() if k in conv_actor_net.state_dict()}
# actor_model.load_state_dict(pretrained_weights)
# pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
# free_actor_model.load_state_dict(free_pretrained_weights, strict=False)

conv_actor_net.load_state_dict(clutter_pretrained_weights, strict=False)
# expert_gate_net.load_state_dict(expert_choose_net_weights, strict=False)
# Initialize the critic network
conv_critic_net.apply(initialize_weights)
# model = GATActorCritic(in_channels, hidden_dim, action_dim, num_agents).to(device)
from datetime import datetime

# Create a unique log directory with a timestamp
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/ppo_evaluation_conv_{current_time}_{train_env_type}_{str(noise_level)}'

# Initialize the SummaryWriter with the unique log directory
writer = SummaryWriter(log_dir=log_dir)

# print("clutter_actor para:{}".format(clutter_actor_model.state_dict()))
# input("1")
# model.load_state_dict(model_state_dict)
actor_optimizer = optim.Adam(conv_actor_net.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(conv_critic_net.parameters(), lr=3e-4)


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
        scenario_name="formation_control_teacher_graph_obs_cuda1_conv",
        num_envs=1,
        device=device,
        continuous_actions=True,
        n_agents=num_agents,
        env_type=train_env_type,
        is_evaluation_mode=True,
        is_imitation=False,
        working_mode="RL",
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
        batch_size = 1
        n_agents = num_agents
        # print("batch_size:{}".format(batch_size))
        # obs_dim = obs.shape[2]
        # print("step:{}".format(step))
        # Prepare observations for GNN


        # Forward pass through the policy
        with torch.no_grad():
            # print("obs list size:{}".format(len(obs)))
            # print("obs:{}".format(obs[0]))
            # print("obs edge_attr edvice:{}".format(obs[0].edge_attr.device))
            # print("obs edge_index deviuce:{}".format(obs[0].edge_index.device))
            action_mean, action_std = conv_actor_net(obs['laser'], obs['relative_pos']) 
            # print()

            dist = torch.distributions.Normal(action_mean, action_std)
            # print("batch_obs device:{}".format(batch_obs))
            # batch_obs = batch_obs.to(device)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # print("batch_obs:{}".format(batch_obs))
            # action_mean, state_value = model(batch_obs)
            # action_mean = actor_model(batch_obs)
            state_value = conv_critic_net(obs['laser'], obs['relative_pos'])
            # dist = torch.distributions.Normal(action_mean, action_std)
            # action = dist.sample()
            # log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # print("batch_obs:{}".format(batch_obs))
            # action_mean, state_value = model(batch_obs)
            # action_mean = actor_model(batch_obs)
            # state_value = critic_model(batch_obs)
            # print("action_mean:{}".format(action_mean))

        # Reshape actions for the environment
        action_env = action.view(batch_size, n_agents, -1).to(device)

        next_obs, rewards, dones, infos = env.step(action_env)
        rewards = rewards.to(device)

        # agent_collision_rew = infos[]
        dones = dones.to(device)
        # next_obs = [data.to(device) for data in next_obs]
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
    save_video("eva_video/ppo_evaluation_conv_{}_{}_{}_{}".format(current_time, epoch_restart, train_env_type, noise_level), frame_list, fps=1 / 0.1)

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