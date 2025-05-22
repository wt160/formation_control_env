import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from collections import deque
import numpy as np
from copy import deepcopy
import copy
import random
import time
from torch_geometric.nn import GATConv, global_mean_pool

import torch
from torch_geometric.data import Data, Batch
from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video
import pickle
import copy
import math
import os
from PIL import Image
import time
import typing
from typing import Callable, Dict, List
import numpy as np
import pickle
from collections import deque
import torch
import torch.optim as optim
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, TransformerEncoder, TransformerEncoderLayer
from vmas.make_env import make_env
from scipy.optimize import linear_sum_assignment
from vmas import render_interactively
class MuZeroNetwork(nn.Module):
    """MuZero核心网络组件"""
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 表示网络
        self.representation = GNNEncoder(input_dim, hidden_dim)
        
        # 动态网络（输入维度修正）
        self.dynamics = DynamicsModel(hidden_dim * 8 + action_dim, hidden_dim * 8)  # 8 heads
        
        # 预测网络（输入维度修正）
        self.prediction = PredictionHead(hidden_dim * 8, action_dim)  # 8 heads

        # 奖励预测头（维度修正）
        self.reward_head = nn.Linear(hidden_dim * 8, 1)  # 8 heads
        
    def forward(self, graph, action=None):
        # 编码当前状态
        h = self.representation(graph)
        if action is None:
            policy, value = self.prediction(h)
            return policy, value
        
        # 状态转移
        h_next = self.dynamics(torch.cat([h, action], dim=1))
        reward = self.reward_head(h_next)
        policy, value = self.prediction(h_next)
        return h_next, reward, policy, value

class GNNEncoder(nn.Module):
    """图神经网络编码器（保持维度不变）"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, 
                           heads=8, concat=True, 
                           edge_dim=1, add_self_loops=False)
        
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, 
                           heads=8, concat=True, 
                           edge_dim=1)
        
    def forward(self, graph):
        # 输入graph.x形状: [num_nodes, input_dim]
        x = self.conv1(graph.x, graph.edge_index, edge_attr=graph.edge_attr)
        x = F.relu(x)  # 形状: [num_nodes, hidden_dim * 8]
        
        x = self.conv2(x, graph.edge_index, edge_attr=graph.edge_attr)  # 形状: [num_nodes, hidden_dim * 8]
        
        # 全局平均池化
        return torch.mean(x, dim=0)  # 输出形状: [hidden_dim * 8]

class DynamicsModel(nn.Module):
    """动态预测模型（维度修正）"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),  # 输入维度: hidden_dim*8 + action_dim
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # 输出维度: hidden_dim*8
        )
        
    def forward(self, x):
        return self.mlp(x)

class PredictionHead(nn.Module):
    """策略-价值预测头（维度修正）"""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        # 策略头
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 512),  # 输入维度: hidden_dim*8
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # 价值头
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        policy = F.softmax(self.policy_net(x), dim=-1)
        value = self.value_net(x)
        return policy, value
class MCTSNode:
    def __init__(self, h, prior, reward=0.0, done=False):
        self.h = h            # 隐藏状态 (Tensor)
        self.prior = prior    # 先验概率
        self.reward = reward  # 即时奖励
        self.done = done      # 终止状态标记
        self.children = {}    # 子节点字典 {action_idx: node}
        self.visit_count = 0
        self.value_sum = 0.0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0


class MuZeroAgent:
    """多机器人MuZero智能体"""
    def __init__(self, config, device):
        self.config = config
        self.network = MuZeroNetwork(config.input_dim, config.action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        self.device = device
        # 离散动作空间定义
        self.action_space = torch.tensor(
            [(x,y,t) for x in [-0.5,0,0.5,1] for y in [-0.5,0,0.5] for t in [-0.5,0,0.5]]
        ).float().to(device)

    def _get_action_distribution(self, node):
        """计算动作的访问概率分布"""
        visit_counts = [
            child.visit_count for child in node.children.values()
        ]
        total_visits = sum(visit_counts)
        return [count / total_visits for count in visit_counts]





   
    def mcts_simulation(self, root_obs, num_simulations=50):
        # 初始化根节点（保持不变）
        with torch.no_grad():
            root_h = self.network.representation(root_obs.obs[0])
        root_node = MCTSNode(root_h, prior=1.0)

        for _ in range(num_simulations):
            node = root_node
            search_path = [node]
            accumulated_reward = 0.0

            # 选择阶段：仅在已扩展节点继续

            while node.expanded() and not node.done:
                action, child_node = self._select_child_continuous(node)
                search_path.append(child_node)
                accumulated_reward += child_node.reward * (self.config.gamma ** len(search_path))
                node = child_node

            # 扩展阶段：当节点未扩展时执行
            if not node.done and not node.expanded():
                with torch.no_grad():
                    action_dist, value = self.network.prediction(node.h)
                    candidate_actions = action_dist.sample((5,))  # 采样5个候选动作

                    # 批量动态预测
                    dynamics_inputs = torch.cat([
                        node.h.unsqueeze(0).repeat(5, 1),
                        candidate_actions
                    ], dim=1)
                    h_nexts, rewards = self.network.dynamics(dynamics_inputs)

                # 创建子节点
                for action, h_next, reward in zip(candidate_actions, h_nexts, rewards):
                    new_node = MCTSNode(
                        h=h_next.squeeze(0),
                        prior=action_dist.log_prob(action).exp().item(),
                        reward=reward.item()
                    )
                    node.children[_hash_tensor(action)] = new_node

                # 反向传播
                total_value = accumulated_reward + (self.config.gamma ** len(search_path)) * value.item()
                self._backpropagate(search_path, total_value)

        return self._select_continuous_action(root_node), root_node.value()

    def _select_child_continuous(self, node):
        """修正后的连续动作子节点选择"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # 此时node.children一定非空（因为node.expanded()为True）
        for action_hash, child in node.children.items():
            action = _recover_tensor(action_hash)
            
            # 连续版PUCT公式（考虑动作相似度）
            similarity = torch.cosine_similarity(node.h, child.h, dim=0)
            exploration_term = self.config.c_puct * child.prior * np.sqrt(node.visit_count) / (child.visit_count + 1)
            score = child.value() + exploration_term * similarity.item()
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


    def _select_child(self, node):
        """UCB选择子节点"""
        actions = list(node.children.keys())
        ucb_values = [
            child.value() + self.config.ucb_c * child.prior *
            np.sqrt(node.visit_count) / (child.visit_count + 1)
            for child in node.children.values()
        ]
        action = actions[np.argmax(ucb_values)]
        return action, node.children[action]
    
    def _backpropagate(self, path, value):
        """修正后的回溯方法"""
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.gamma * value
            
    def _select_action(self, node):
        """根据访问计数选择动作"""
        visit_counts = [
            (action, child.visit_count)
            for action, child in node.children.items()
        ]
        total_visits = sum(n for _, n in visit_counts)
        probs = [n / total_visits for _, n in visit_counts]
        return visit_counts[np.argmax(probs)][0]
    

    def train_step(self, batch):
        """修正后的训练步骤"""
        states, actions, target_policies, target_values = batch
        
        # 网络预测
        policy_pred, value_pred = self.network(states)
        
        # 计算双头损失
        policy_loss = torch.mean(-torch.sum(target_policies * torch.log(policy_pred), dim=1))
        value_loss = torch.mean((value_pred - target_values)**2)
        total_loss = policy_loss + self.config.value_weight * value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    

class TrainingPipeline:
    """完整训练流程"""
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        

        
    def run(self):
        for episode in range(self.config.num_episodes):
            trajectories = []
            
            # 数据收集
            for d in range(self.config.batch_size):
                state = self.env.reset()
                done = False
                trajectory = []
                
                while not done:
                    # 运行MCTS获取策略与价值预测
                    action, mcts_probs, value_estimate = self.agent.mcts_simulation(state)
                    
                    # 环境交互
                    next_state, reward, done, _ = self.env.step(action)
                    
                    trajectory.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'mcts_probs': mcts_probs,
                        'value_estimate': value_estimate,
                        'done': done
                    })
                    
                    state = next_state
                
                # 计算n-step TD目标
                self._compute_targets(trajectory)
                trajectories.append(trajectory)
            
            # 模型更新
            batch = self._process_trajectories(trajectories)
            loss = self.agent.train_step(batch)
            
            print(f"Episode {episode}, Loss: {loss:.4f}")


    def _compute_targets(self, trajectory):
        """计算多步TD目标和策略目标"""
        gamma = self.config.discount
        n_steps = self.config.n_step
        
        for i in range(len(trajectory)):
            # 价值目标：n-step TD + bootstrap
            target_value = 0
            for j in range(n_steps):
                if i + j >= len(trajectory):
                    break
                target_value += (gamma**j) * trajectory[i+j]['reward']
            
            if i + n_steps < len(trajectory) and not trajectory[i+n_steps]['done']:
                target_value += (gamma**n_steps) * trajectory[i+n_steps]['value_estimate']
            
            trajectory[i]['target_value'] = target_value
            
            # 策略目标：MCTS访问计数分布
            trajectory[i]['target_policy'] = trajectory[i]['mcts_probs']

    def _process_trajectories(self, trajectories):
        """处理轨迹数据为训练批次"""
        batch_states = []
        batch_actions = []
        batch_target_policies = []
        batch_target_values = []
        
        for traj in trajectories:
            for transition in traj:
                batch_states.append(transition['state'])
                batch_actions.append(transition['action'])
                batch_target_policies.append(transition['target_policy'])
                batch_target_values.append(transition['target_value'])
        
        return (
            Batch.from_data_list(batch_states),
            torch.tensor(batch_actions),
            torch.tensor(batch_target_policies),
            torch.tensor(batch_target_values).unsqueeze(1)
        )         
    def _process_batch(self, states, actions, rewards):
        """处理批数据"""
        states = Batch.from_data_list(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards).unsqueeze(1)
        return states, actions, rewards


class VMASEnvState:
    def __init__(self, obs, done):
        self.done = done
        self.obs = obs

class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents, env_type=None, is_evaluation_mode=False, is_imitation=False, working_mode="imitation", evaluation_index=0):
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
        )
        self.max_connection_distance = 1.8
        self.device = device
        self.n_agents = n_agents
        self.num_envs = num_envs

    def reset(self):
        obs = self.env.reset()
        print("obs length:{}".format(len(obs)))
        print("obs:{}".format(obs))

        batch_size = obs[0]['relative_pos'].shape[0]
        dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        obs = self.get_graph_from_obs( obs, dones)

        print("after graph obs:{}".format(obs))
        # obs is a list of observations per agent
        # Stack observations to shape [num_envs, n_agents, obs_dim]
        # obs = torch.stack(obs, dim=1).to(self.device)
        state = VMASEnvState(obs, dones)
        return state

    def get_obs(self):
        obs = self.env.get_obs()
        obs = obs[0]
        batch_size = obs[0]['relative_pos'].shape[0]
        dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        obs = self.get_graph_from_obs( obs, dones)
        return obs

    def get_leader_paths(self):
        leader_paths = self.env.get_leader_paths()
        return leader_paths
    
    
    def get_graph_from_obs(self, obs, dones):
        x = None
        batch_size = obs[0]['relative_pos'].shape[0]
        n_agents = len(obs)
        
        # Pre-allocate tensors for each agent's observations
        laser_obs = []
        relative_pos_obs = []
        nominal_pos_obs = []
        # Collect observations by type
        for agent_index in range(n_agents):
            # laser_obs.append(obs[agent_index]['laser'])
            relative_pos_obs.append(obs[agent_index]['relative_pos'])
            nominal_pos_obs.append(obs[agent_index]['nominal_pos'])
        # Stack observations along agent dimension (dim=1)
        # laser_tensor shape: [batch_size, n_agents, laser_dim]
        # laser_tensor = torch.stack(laser_obs, dim=1)
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("laser:{}".format(laser_tensor))
        # relative_pos_tensor shape: [batch_size, n_agents, pos_dim]
        relative_pos_tensor = torch.stack(relative_pos_obs, dim=1)
        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        # print("relative pos:{}".format(relative_pos_tensor))
        nominal_pos_tensor = torch.stack(nominal_pos_obs, dim=1)
        # print("nominal pos tensor shape:{}".format(nominal_pos_tensor.shape))    #[2, 5, 3]
        # print("nominal pos:{}".format(nominal_pos_tensor))
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        
        # Get feature dimensions
        # laser_dim = laser_tensor.shape[-1]
        pos_dim = relative_pos_tensor.shape[-1]
        nominal_dim = nominal_pos_tensor.shape[-1]
        # Reshape tensors to combine batch and agent dimensions for concatenation
        # [batch_size, n_agents, feature_dim] -> [batch_size * n_agents, feature_dim]
        # laser_flat = laser_tensor.reshape(-1, laser_dim)
        relative_pos_flat = relative_pos_tensor.reshape(-1, pos_dim)
        nominal_pos_flat = nominal_pos_tensor.reshape(-1, nominal_dim)
        # Concatenate features along feature dimension
        # [batch_size * n_agents, combined_dim]
        # combined = torch.cat([laser_flat, relative_pos_flat, nominal_pos_flat], dim=1)
        combined = torch.cat([relative_pos_flat, nominal_pos_flat], dim=1)
        
        # Reshape back to [batch_size, n_agents, combined_dim]
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
            if dones[d] == True:
                graph_list.append([])
                continue
            x = combined_x[d, :, :]
            edge_index = []
            edge_attr = []
            # print("x:{}".format(x))
            # input("1")
            # Create a tensor for the nominal formation
            nominal_formation_tensor = x[:, :-2]  # Assuming the first two dimensions are the 
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    # Add edges between agents
                    agent_pos_i = nominal_formation_tensor[i, :2]  # Get the position part
                    agent_pos_j = nominal_formation_tensor[j, :2]  # Get the position part

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
        
        # Sum rewards across agents
        summed_rewards = rewards.sum(dim=1)  # [num_envs]
        
        # If done_override is provided, set done flags accordingly
        if done_override is not None:
            dones = dones | done_override.unsqueeze(1)  # Broadcast to [num_envs, n_agents]
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





# 配置参数
class Config:
    input_dim = 6  # 根据实际图数据维度调整
    action_dim = 15  # 离散动作数量
    num_episodes = 1000
    batch_size = 32
    ucb_c = 1.25
    discount = 0.99
    num_simulations = 200


render = False
save_render = False
num_envs = 1
n_steps = 1000
random_action = False
# device = "cpu",
device = torch.device('cpu')
scenario_name = "waterfall"
n_agents = 5
continuous_actions = True
visualize_render = True
collect_num = 30
is_imitation = True
env_type = "narrow"
working_mode = "imitation"
filename ="123.pkl"
world_model_filename ="world_model.pkl"
env = VMASWrapper(
        scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap",
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        n_agents=n_agents,
        env_type=env_type,
        is_evaluation_mode=False,
        is_imitation=is_imitation,
        working_mode=working_mode,
    )

agent = MuZeroAgent(Config(), device)
pipeline = TrainingPipeline(env, agent, Config())

# 启动训练
pipeline.run()