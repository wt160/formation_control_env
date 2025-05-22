import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # ✅ 正确导入

# 你的数据集处理应使用常规Dataset
from torch.utils.data import Dataset
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
import math
# Load the collected data


import sys

data_filename = sys.argv[1]
policy_filename = sys.argv[2]

# data_filename = "collected_data_narrow_6.pkl"
# policy_filename = "best_imitation_model_narrow_noise_6.pth"

with open(data_filename, 'rb') as f:
    collected_data = pickle.load(f)
# print("collected_data:{}".format(collected_data))
# print("collected tensor shape:{}".format(collected_data[0]['graph_tensor'].shape))
# Process the data
dataset = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LASER_RANGE = 3.5

noise_total_num = 75
data_length = len(collected_data)

for data_index, data_point in enumerate(collected_data):
    if data_index > data_length / 3:
        break
    laser_obs = data_point['laser_obs']  #shape [B, 1, laser_bin_num]
    relative_pos = data_point['relative_pos']  # List of graphs with length batch_dim
    optimized_target_pos = data_point['optimized_target_pos']  # Dict of agent positions with batch_dim
    nominal_pos = data_point['nominal_pos']
    agent_vel = data_point['agent_vel']
    agent_ang_vel = data_point['agent_ang_vel']
    # for agent_name, agent_velocity in agent_vel.items():
        # print("agent_vel shape:{}".format(agent_velocity.shape))
    # for agent_name, agent_ang_velocity in agent_ang_vel.items():
        # print("agent_ang_vel shape:{}".format(agent_ang_velocity.shape))

    agent_names = sorted(optimized_target_pos.keys())
    # optimized_target_pos: dict with keys as agent names and values as tensors of shape [batch_dim, 2]

    batch_dim = laser_obs.shape[0]
    #add code to put data into dataset
    for batch_idx in range(batch_dim):
        # 提取当前batch的观测数据
        current_laser = laser_obs[batch_idx]  # shape [1, laser_bin_num]
        current_relative = relative_pos[batch_idx]  # 假设是形状 [num_agents, 3] 的张量
        current_nominal_pos = nominal_pos[batch_idx]
        current_relative = current_relative.squeeze(dim = -1)
        # 提取目标位置并排序对齐
        target_positions = []
        for agent_name in agent_names:
            target_positions.append(optimized_target_pos[agent_name][batch_idx])
        target_tensor = torch.tensor(np.stack(target_positions))  # shape [num_agents, 2]

        target_velocity = []
        for agent_name, agent_velocity in agent_vel.items():
            target_velocity.append(agent_velocity[batch_idx, :])
        target_vel_tensor = torch.tensor(np.stack(target_velocity))

        target_ang_velocity = []
        for agent_name, agent_ang_velocity in agent_ang_vel.items():
            target_ang_velocity.append(agent_ang_velocity[batch_idx, :])
        target_ang_vel_tensor = torch.tensor(np.stack(target_ang_velocity))

        # print("target_vel_tensor:{}".format(target_vel_tensor.shape))
        # print("target_ang_vel tensor:{}".format(target_ang_vel_tensor.shape))
        target_vel = torch.cat((target_vel_tensor, target_ang_vel_tensor), dim=1)
        print("target_vel:{}".format(target_vel))
        # print("target_vel shape:{}".format(target_vel.shape))
        # if target_vel.shape[1] == 4:
            # input("1")
        # print("target_tensor:{}".format(target_tensor.shape))

        # print("current_relative shape:{}".format(current_relative.shape))
        # print("current_relative:{}".format(current_relative.shape))
        # 添加原始数据样本
        dataset.append((
            current_laser.to(device),
            current_relative.to(device),
            target_tensor.to(device),
            current_nominal_pos.to(device),
            target_vel.to(device),
        ))
        
        # 添加带噪声的增强数据样本
        # for _ in range(noise_total_num):
        #     # 添加高斯噪声到激光数据
        #     noisy_laser = current_laser + torch.randn_like(current_laser) * 0.1 * MAX_LASER_RANGE
        #     noisy_laser = torch.clamp(noisy_laser, 0, MAX_LASER_RANGE)
            
        #     # 添加噪声到相对位置（位置+朝向）
        #     pos_noise = torch.randn_like(current_relative) * torch.tensor([0.1, 0.1, 0.05])
        #     # print("pos_noise shape:{}".format(pos_noise.shape))
        #     noisy_relative = current_relative + pos_noise
            
        #     # 添加噪声到目标位置
        #     target_noise = torch.randn_like(target_tensor) * 0.1
        #     noisy_target = target_tensor + target_noise
        #     # print("noise target_tensor:{}".format(noisy_target.shape))
        #     # print("noise_relative shape:{}".format(noisy_relative.shape))
        #     # if noisy_target.shape[1] == 4:
        #         # input("1")
        #     # if noisy_relative.shape[1] == 4:
        #         # input("2")
        #     # print("noise current_relative:{}".format(noisy_relative.shape))
        #     dataset.append((
        #         noisy_laser.to(device),
        #         noisy_relative.to(device),
        #         noisy_target.to(device),
        #         current_nominal_pos.to(device),
        #         target_vel.to(device),
        #     ))

# Shuffle and split dataset
np.random.shuffle(dataset)
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


class LaserFC_Actor(nn.Module):
    def __init__(self, 
                 laser_dim=360,       # 一维激光数据维度
                 num_agents=5,        # 机器人数量（含领队）
                 relpos_dim=3,        # 相对位姿维度(x,y,theta)
                 nominal_pos_dim=3,
                 hidden_dim=128, 
                 action_dim=3):       # 输出SE2位姿
        super(LaserFC_Actor, self).__init__()
        self.num_agents = num_agents
        
        # 激光处理分支 (1D CNN)
        self.laser_conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2),  # 输入通道改为3
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # 自适应池化统一维度
            nn.Flatten()
        )
        
        # 自动计算激光分支输出维度
        with torch.no_grad():
            dummy_laser = torch.zeros(1, 3, laser_dim)
            self.conv_out_dim = self.laser_conv(dummy_laser).shape[1]
        
        # 相对位置处理分支
        self.pos_encoder = nn.Sequential(
            nn.Linear(relpos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        

        self.nominal_pos_encoder = nn.Sequential(
            nn.Linear(nominal_pos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
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

    def forward(self, laser_input, relative_pos, nominal_pos):
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
        
        nominal_pos_feat = self.nominal_pos_encoder(nominal_pos.view(-1, 3))
        nominal_pos_feat = nominal_pos_feat.view(batch_size, self.num_agents, -1)

        # 扩展激光特征并与位置特征拼接
        laser_feat = laser_feat.unsqueeze(1).expand(-1, self.num_agents, -1)  # (batch, num_agents, conv_out_dim)
        # print("laser_feat shape:{}".format(laser_feat.shape))
        combined = torch.cat([laser_feat, pos_feat, nominal_pos_feat], dim=-1)  # (batch, num_agents, conv_out_dim+64)
        
        # 联合特征处理
        hidden = self.joint_net(combined)
        action_mean = self.action_head(hidden)  # (batch, num_agents, action_dim)
        
        # 计算标准差
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        return action_mean











laser_dim = 90    # 假设激光是360维的1D数组（模拟360度激光扫描）
num_agents = 5     # 1个领队 + 4个跟随者

# 初始化网络
conv_actor_net = LaserFC_Actor(laser_dim=laser_dim, num_agents=num_agents).to(device)

# 模拟输入
batch_size = 32
laser_data = torch.randn(batch_size, 1, laser_dim)
rel_pos = torch.randn(batch_size, num_agents, 3)  # 每个机器人的相对位姿

# 前向计算
# mean = conv_actor_net(laser_data, rel_pos)




num_agents=5
sample_graph = dataset[0]
output_dim = 3  # Assuming 2D positions

# Initialize the model, loss function, and optimizer


        



criterion = nn.MSELoss()
optimizer = optim.Adam(conv_actor_net.parameters(), lr=0.001)

# Training and validation
num_epochs = 4
best_val_loss = float('inf')

for epoch in range(num_epochs):
    conv_actor_net.train()
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
        # print("data:{}".format(data))
        laser, relative, target, nominal_pos, target_vel = data
        # print("relative shape:{}".format(relative.shape))
        predicted_velocity = conv_actor_net(laser, relative, nominal_pos)  # Shape: [batch_size, num_agents, 2]
        
        # print("target shape:{}".format(target_positions.shape))
        loss = criterion(predicted_velocity, target_vel)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Average Training Loss: {avg_loss:.4f}')
    
    # Validation
    conv_actor_net.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            # graph_tensor = data['graph_tensor']
            # target_positions = data['optimized_target_pos']
            laser, relative, target, nominal_pos, target_vel = data
            predicted_velocity = conv_actor_net(laser, relative, nominal_pos)  # Shape: [batch_size, num_agents, 2]
            # predicted_positions = gnn_actor_net(graph_tensor)
            # target_positions = data.y  # Shape: [batch_size * num_agents, 2]
            # target_positions = target_positions.view(-1, num_agents, 3)  # Shape: [batch_size, num_agents, 2]
            loss = criterion(predicted_velocity, target_vel)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(conv_actor_net.state_dict(), policy_filename)
        print('Best model saved.')

print('Training complete.')