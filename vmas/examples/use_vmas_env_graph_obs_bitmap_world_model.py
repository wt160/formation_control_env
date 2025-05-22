#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from copy import deepcopy
import copy
import random
import time

import torch
from torch_geometric.data import Data, Batch
from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video
import pickle
from collections import defaultdict

def get_expert_action(agent: Agent, continuous: bool, env):
    action = env.scenario.get_expert_action(agent)
    return action.clone()

def get_expert_action(agent: Agent, continuous: bool, env):
    action = env.scenario.get_expert_action(agent)
    return action.clone()

def _get_deterministic_action(agent: Agent, continuous: bool, env):
    if continuous:
        action = -agent.action.u_range_tensor.expand(env.batch_dim, agent.action_size)
    else:
        action = (
            torch.tensor([1], device=env.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(env.batch_dim, 1)
        )
    return action.clone()

class FormationGenerator:
    def __init__(self, n_followers=4, grid_step=0.5):
        self.n_followers = n_followers
        self.grid_step = grid_step
        self.config_registry = defaultdict(int)
        
        # 预定义标准编队（高权重）
        self.nominal_formations = {
            'v_shape': self._create_formation([
                [0.0, 0.0], [-1.0, -1.0], [1.0, -1.0], [-0.5, -2.0], [0.5, -2.0]
            ]),
            'line': self._create_formation([
                [0.0, 0.0], [0.0, -1.0], [0.0, -2.0], [0.0, -3.0], [0.0, -4.0]
            ]),
            'parallel': self._create_formation([
                [0.0, 0.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, -2.0], [1.0, -2.0]
            ])
        }
        
        # 生成候选网格
        self.grid_points = self._generate_grid()
        self._init_config_pool()

    def _generate_grid(self):
        """生成领队后方的网格点"""
        x_range = np.arange(-2.0, 2.1, self.grid_step)
        y_range = np.arange(-4.0, -0.1, self.grid_step)
        grid = []
        for x in x_range:
            for y in y_range:
                grid.append([round(x,1), round(y,1)])
        return np.array(grid)
    
    def _create_formation(self, positions):
        """创建编队并注册"""
        formation = np.array(positions[1:])  # 排除领队
        key = tuple(formation.flatten().round(1))
        self.config_registry[key] = 100  # 初始高权重
        return formation

    def _init_config_pool(self):
        """初始化配置池"""
        # 添加标准编队
        for formation in self.nominal_formations.values():
            self._register_formation(formation)
        
        # 生成随机有效编队
        for _ in range(100):
            new_formation = self._generate_random_formation()
            self._register_formation(new_formation)

    def _generate_random_formation(self):
        """生成有效随机编队"""
        while True:
            # 随机选择不重复的网格点
            indices = np.random.choice(len(self.grid_points), self.n_followers, replace=False)
            formation = self.grid_points[indices]
            
            # 验证有效性（所有点在领队后方）
            if np.all(formation[:,1] < 0):
                return formation

    def _register_formation(self, formation):
        """注册新编队配置"""
        key = tuple(formation.flatten().round(1))
        if key not in self.config_registry:
            self.config_registry[key] = 0

    def get_balanced_formation(self):
        """获取使用次数最少的编队"""
        min_count = min(self.config_registry.values())
        candidates = [k for k, v in self.config_registry.items() if v == min_count]
        chosen_key = random.choice(candidates)
        
        # 更新使用计数
        self.config_registry[chosen_key] += 1
        
        # 将键值转换为位置数组
        positions = np.array(chosen_key).reshape(-1, 2)
        return np.vstack([[0.0, 0.0], positions])  # 添加领队位置

    def get_nominal_formation(self):
        """获取预定义标准编队"""
        formation_type = random.choice(list(self.nominal_formations.keys()))
        return np.vstack([[0.0, 0.0], self.nominal_formations[formation_type]])


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




class EnhancedDataCollector:
    def __init__(self, env, num_envs, n_agents, config_balance_interval=100):
        self.env = env
        self.num_envs = num_envs
        self.n_agents = n_agents
        
        # 初始化编队生成系统
        self.formation_generator = FormationGenerator(n_followers=n_agents-1)
        
        # 配置平衡参数
        self.config_balance_interval = config_balance_interval
        self.last_balance_step = 0
        
        # 定义离散动作空间（v_x, v_y, v_theta）
        self.action_space = {
            'v_x': torch.arange(-1.0, 1.1, 0.1).tolist(),
            'v_y': torch.arange(-1.0, 1.1, 0.1).tolist(),
            'v_theta': torch.arange(-0.5, 0.6, 0.1).tolist()
        }
        
        self.nominal_velocity_sets = [
            {'velocity': [0.0, 0.0, 0.0], 'weight': 50},  # 静止状态
            {'velocity': [0.3, 0.0, 0.0], 'weight': 30},  # 正向移动
            {'velocity': [-0.2, 0.0, 0.0], 'weight': 20}  # 逆向移动
        ]
        
        # 速度范围约束
        self.velocity_ranges = {
            'vx': (-0.5, 1.0),
            'vy': (-0.5, 0.5),
            'vtheta': (-0.5, 0.5)
        }


        # 状态跟踪
        self.current_formations = [None] * num_envs

    def _get_discrete_action(self):
        """生成符合规范的离散动作"""
        return [
            random.choice(self.action_space['v_x']),
            random.choice(self.action_space['v_y']),
            random.choice(self.action_space['v_theta'])
        ]

    def _select_formation_strategy(self):
        """编队选择策略（80%标准编队，20%平衡编队）"""
        return 'nominal' if random.random() < 0.8 else 'balanced'

    def _apply_formation(self, env_idx):
        """为指定环境实例应用编队"""
        strategy = self._select_formation_strategy()
        
        if strategy == 'nominal':
            formation = self.formation_generator.get_nominal_formation()
        else:
            formation = self.formation_generator.get_balanced_formation()
        

        # 生成速度
        velocities = self._generate_velocity()
        
        
        
        # 记录当前状态
        self.current_formations[env_idx] = {
            'positions': formation,
            'velocities': velocities
        }

        # 设置到环境
        self.env.env.set_initial_state(
            env_idx,
            positions=formation,
            velocities=velocities
        )

        # 记录当前编队配置
        # self.env.env.set_positions(formation, env_idx)

    def _validate_formation(self, obs, env_idx):
        """验证编队有效性"""
        try:
            # 检查跟随者是否在领队后方
            leader_y = obs.x[0][1].item()
            for follower in obs.x[1:]:
                if follower[1].item() > leader_y:
                    return False
            return True
        except Exception as e:
            print(f"Validation error in env {env_idx}: {str(e)}")
            return False

    def _record_configuration(self, env_idx):
        """记录编队配置使用情况"""
        config_key = tuple(self.current_formations[env_idx][1:].flatten().round(1))
        self.formation_generator.config_registry[config_key] += 1


    def _validate_velocity(self, velocities):
        """验证速度合规性"""
        for v in velocities:
            if not (self.velocity_ranges['vx'][0] <= v[0] <= self.velocity_ranges['vx'][1]):
                return False
            if not (self.velocity_ranges['vy'][0] <= v[1] <= self.velocity_ranges['vy'][1]):
                return False
            if not (self.velocity_ranges['vtheta'][0] <= v[2] <= self.velocity_ranges['vtheta'][1]):
                return False
        return True


    def _generate_velocity(self):
        """生成速度配置（带权重采样）"""
        # 按权重选择速度模式
        choices = random.choices(
            population=self.nominal_velocity_sets,
            weights=[v['weight'] for v in self.nominal_velocity_sets],
            k=1
        )
        base_velocity = choices[0]['velocity']
        
        # 添加高斯噪声
        noise_scale = 0.05  # 噪声强度
        velocities = []
        for _ in range(self.n_agents):
            vx = base_velocity[0] + np.random.normal(0, noise_scale)
            vy = base_velocity[1] + np.random.normal(0, noise_scale)
            vtheta = base_velocity[2] + np.random.normal(0, noise_scale)
            
            # 应用范围约束
            vx = np.clip(vx, *self.velocity_ranges['vx'])
            vy = np.clip(vy, *self.velocity_ranges['vy'])
            vtheta = np.clip(vtheta, *self.velocity_ranges['vtheta'])
            
            velocities.append([vx, vy, vtheta])
        
        return torch.tensor(velocities, dtype=torch.float32)


    def collect_episode_data(self, max_steps=500):
        """执行完整的数据收集流程"""
        collected_data = []
        world_model_data = []
        
        # 初始化环境编队
        for env_idx in range(self.num_envs):
            self._apply_formation(env_idx)

        current_step = 0
        while current_step < max_steps:
            # 定期平衡配置分布
            if current_step - self.last_balance_step > self.config_balance_interval:
                self._balance_configurations()
                self.last_balance_step = current_step
            
            # 生成批量动作
            batch_actions = []
            for _ in range(self.num_envs):
                env_actions = []
                for _ in range(self.n_agents):
                    env_actions.append(self._get_discrete_action())
                batch_actions.append(env_actions)
            
            # 转换为环境需要的张量格式
            action_tensors = [
                torch.tensor(actions, dtype=torch.float32) 
                for actions in batch_actions
            ]
            
            # 执行环境步进
            obs_list, rewards, dones, infos = self.env.step(action_tensors)
            
            # 处理每个环境实例的数据
            valid_data_points = []
            for env_idx, (obs, actions) in enumerate(zip(obs_list, batch_actions)):
                # 跳过无效环境实例
                if obs is None:
                    continue
                
                # 验证编队有效性
                if not self._validate_formation(obs, env_idx):
                    self._apply_formation(env_idx)  # 重置编队
                    continue
                
                # 记录配置使用情况
                self._record_configuration(env_idx)
                
                # 构建数据点
                data_point = {
                    'observation': obs,
                    'action': actions,
                    'formation': self.current_formations[env_idx].copy(),
                    'timestamp': current_step
                }
                valid_data_points.append(data_point)
            
            # 更新数据集
            collected_data.extend(valid_data_points)
            current_step += 1
            
            # 构建世界模型训练对
            if len(collected_data) >= 2:
                world_model_data += self._create_world_model_pairs(collected_data[-self.num_envs*2:])
        
        return collected_data, world_model_data

    def _balance_configurations(self):
        """平衡配置分布的重置操作"""
        print("Executing configuration balancing...")
        for env_idx in range(self.num_envs):
            try:
                # 选择最少使用的配置
                formation = self.formation_generator.get_balanced_formation()
                self.current_formations[env_idx] = formation
                self.env.env.set_positions(formation, env_idx)
            except Exception as e:
                print(f"Balancing error in env {env_idx}: {str(e)}")
                self._apply_formation(env_idx)

    def _create_world_model_pairs(self, data_buffer):
        """创建连续状态转移对"""
        wm_pairs = []
        for i in range(1, len(data_buffer)):
            prev_data = data_buffer[i-1]
            current_data = data_buffer[i]
            
            # 验证环境索引一致性
            if prev_data['timestamp'] + 1 != current_data['timestamp']:
                continue
                
            wm_pairs.append({
                'prev_observation': prev_data['observation'],
                'current_observation': current_data['observation'],
                'action': prev_data['action'],
                'formation': prev_data['formation']
            })
        return wm_pairs

    def visualize_config_distribution(self):
        """可视化当前配置分布"""
        import matplotlib.pyplot as plt
        
        config_counts = list(self.formation_generator.config_registry.values())
        plt.figure(figsize=(10, 6))
        plt.hist(config_counts, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Configuration Usage Count')
        plt.ylabel('Frequency')
        plt.title('Configuration Distribution Histogram')
        plt.grid(True)
        plt.show()















































def use_vmas_env(
    render: bool = False,
    save_render: bool = False,
    num_envs: int = 32,
    n_steps: int = 1000,
    random_action: bool = False,
    device: str = "cuda",
    scenario_name: str = "waterfall",
    n_agents: int = 5,
    continuous_actions: bool = True,
    visualize_render: bool = True,
    collect_num = 30,
    is_imitation: bool = True,
    env_type: str = "narrow",
    working_mode: str = "imitation",
    filename: str="123.pkl",
    world_model_filename: str="world_model.pkl",
):
    """Example function to use a vmas environment

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        n_agents (int): Number of agents
        scenario_name (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    dict_spaces = True  # Weather to return obs, rewards, and infos as dictionaries with agent names
    # (by default they are lists of len # of agents)
    collected_data = []  # 

    world_model_data = []
    for current_collect_num in range(collect_num):

        env = VMASWrapper(
            scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap_world_model",
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            n_agents=n_agents,
            env_type=env_type,
            is_evaluation_mode=False,
            is_imitation=is_imitation,
            working_mode=working_mode,
        )
        obs = env.get_obs()


        get_leader_path_start = time.time()
        # batched_paths = env.get_leader_paths()
        # input("get leader path time:{}".format(time.time() - get_leader_path_start))
        # for dim in range(num_envs):
        #     init_direction = env.env.compute_leader_init_direction(batched_paths[dim])
        #     env.env.reset_world_at_with_init_direction(dim, init_direction)
        last_obs = None
        frame_list = []  # For creating a gif
        init_time = time.time()
        step = 0


        print("current_collect_num:{}".format(current_collect_num))
        with torch.no_grad():
            for _ in range(n_steps):
                step += 1
                print(f"Step {step}")



                
                # VMAS actions can be either a list of tensors (one per agent)
                # or a dict of tensors (one entry per agent with its name as key)
                # Both action inputs can be used independently of what type of space its chosen
                dict_actions = random.choice([True, False])

                actions = {} if dict_actions else []
                for agent in env.env.agents:
                    # print("agent:{}".format(agent))
                    max_value = 0.5
                    min_value = -0.5
                        # action = env.get_random_action(agent)
                    action = (max_value - min_value) * torch.rand((agent.batch_dim, 3)) + min_value
                    if dict_actions:
                        actions.update({agent.name: action})
                    else:
                        actions.append(action)
                # print("actions:{}".format(actions))
                obs, rews, dones, info = env.step(actions)
                # print("obs:{}".format(obs))
                # print("info:{}".format(info))
                agent_0 = env.env.agents[0]
                agent_0_name = agent_0.name

                # graph_list = agent_0_obs.get('graph_list', [])
                optimized_target_pos = {}
                current_agent_vel = {}
              
                # print("info:{}".format(info))
                # Collect optimized_target_pos for all agents
                for agent_index, agent in enumerate(env.env.agents):
                    agent_name = agent.name
                    agent_info = info[agent_index]
                    agent_pos = agent_info['optimized_target_pos'].cpu().detach().numpy()
                    optimized_target_pos[agent_name] = agent_pos
                    agent_vel = agent_info['agent_vel']
                    agent_ang_vel = agent_info['agent_ang_vel']
                    # print("agent_ang_vel:{}".format(agent_ang_vel))
                    current_agent_vel[agent_name] = torch.cat((agent_vel, agent_ang_vel), dim = 1)
                # Build data point
                
                for idx, single_obs in enumerate(obs):
                    if single_obs == []:
                        continue

                    data_point = {
                        'graph_tensor': single_obs,
                        'current_agent_vel': {agent_name: agent_vel for agent_name, agent_vel in current_agent_vel.items()},
                    }


                    if last_obs is not None:
                        world_model_data_point = {
                            'last_graph_tensor': last_obs[idx],
                            'current_graph_tensor': single_obs,
                            'current_agent_vel': {agent_name: agent_vel for agent_name, agent_vel in current_agent_vel.items()},
                            'action': actions,
                        }
                    # Append to collected data
                        world_model_data.append(world_model_data_point)
                    collected_data.append(data_point)
                # Update last obs
                last_obs = copy.deepcopy(obs)

                # print("obs:{}".format(obs))
                # print("info:{}".format(info))
                if render:
                    frame = env.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=visualize_render,
                    )
                    if save_render:
                        frame_list.append(frame)

        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
            f"for {scenario_name} scenario."
        )
        if current_collect_num % 20 == 0:
            with open(filename, 'wb') as f:
                pickle.dump(collected_data, f)

            with open(world_model_filename, 'wb') as f:
                pickle.dump(world_model_data, f)
            print("Collected data saved to 'collected_data.pkl'.")

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.env.scenario.world.dt)





if __name__ == "__main__":


    render = False,
    save_render = False,
    num_envs = 1,
    n_steps = 1000,
    random_action = False,
    device = "cuda",
    scenario_name = "waterfall",
    n_agents = 5,
    continuous_actions = True
    visualize_render = True
    collect_num = 30
    is_imitation = True,
    env_type = "narrow",
    working_mode = "imitation",
    filename ="123.pkl",
    world_model_filename ="world_model.pkl",
    env = VMASWrapper(
            scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap_world_model",
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            n_agents=n_agents,
            env_type=env_type,
            is_evaluation_mode=False,
            is_imitation=is_imitation,
            working_mode=working_mode,
        )

    
    collector = EnhancedDataCollector(env, num_envs=num_envs, n_agents=n_agents)

    # 收集数据并监控
    collected_data, world_model_data = collector.collect_episode_data(max_steps=1000)

    # 可视化配置分布
    collector.visualize_config_distribution()








    # use_vmas_env(
    #     scenario_name="formation_control_teacher_graph_obs_cuda1",
    #     render=False,
    #     num_envs=2,
    #     n_steps=350,
    #     save_render=False,
    #     random_action=True,
    #     continuous_actions=True,
    #     collect_num=200,
    #     is_imitation=True,
    #     device="cuda:1",
    #     env_type="tunnel",
    #     working_mode="imitation",
    #     filename="collected_data_tunnel_0.pkl",
    # )


    # use_vmas_env(
    #     scenario_name="formation_control_teacher_graph_obs_cuda1",
    #     render=False,
    #     num_envs=2,
    #     n_steps=310,
    #     save_render=False,
    #     random_action=True,
    #     continuous_actions=True,
    #     collect_num=200,
    #     is_imitation=True,
    #     device="cuda:1",
    #     env_type="door",
    #     working_mode="imitation",
    #     filename="collected_data_door_0.pkl",
    # )


    # use_vmas_env(
    #     scenario_name="formation_control_teacher_graph_obs_cuda1",
    #     render=False,
    #     num_envs=2,
    #     n_steps=300,
    #     save_render=False,
    #     random_action=True,
    #     continuous_actions=True,
    #     collect_num=400,
    #     is_imitation=True,
    #     device="cuda",
    #     env_type="clutter",
    #     working_mode="imitation",
    #     filename="collected_data_clutter_11.pkl",
    # )




    # use_vmas_env(
    #     scenario_name="formation_control_teacher_graph_obs_cuda1",
    #     render=False,
    #     num_envs=2,
    #     n_steps=500,
    #     save_render=False,
    #     random_action=True,
    #     continuous_actions=True,
    #     collect_num=400,
    #     is_imitation=True,
    #     device="cuda:1",
    #     env_type="door_and_narrow",
    #     working_mode="imitation",
    #     filename="collected_data_door_narrow_0.pkl",
    # )
    t1 = time.time()
    use_vmas_env(
        scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap_world_model",
        render=True,
        num_envs=1,
        n_steps=30000,
        save_render=True,
        random_action=True,
        continuous_actions=True,
        collect_num=1,
        is_imitation=True,
        device="cpu",
        env_type="narrow",
        working_mode="world_model",
        filename="collected_data_narrow_8.pkl",
        world_model_filename="world_model_data_narrow_8.pkl",
    )
    print("used time:{}".format(time.time() - t1))
    # use_vmas_env(
    #     scenario_name="formation_control_teacher_graph_obs_cuda1_test",
    #     render=True,
    #     num_envs=1,
    #     n_steps=1000,
    #     save_render=False,
    #     random_action=True,
    #     continuous_actions=True,
    #     collect_num=200,
    #     is_imitation=True,
    #     device="cuda:1",
    #     env_type="mixed_in_distribution",
    #     working_mode="imitation",
    #     filename="collected_data_mixed.pkl",
    # )

#collect data for imitation learning
#data includes:

#input:
# leader robot pose direction as reference frame, follower robot poses and directions in leader robot's reference frame
#obstacles positions in the leader robot's frame, obstacles number varied. 


#output:
#reformated follower robot's positions and directions. 