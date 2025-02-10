#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video
import pickle

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
class VMASWrapper:
    def __init__(self, scenario_name, num_envs, device, continuous_actions, n_agents, env_type=None, is_evaluation_mode=False, is_imitation=False, working_mode="imitation"):
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
                        visualize_when_rgb=True,
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
        self.log_std = nn.Parameter(torch.ones(1, 1, action_dim) * -1.6) 
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


    # Initialize the models
    clutter_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
    narrow_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
    door_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)
    tunnel_actor_model = GATActor(in_channels, hidden_dim, action_dim, num_agents).to(device)

    
    
    
    
    
    expert_in_channels = 4  # Adjust this to match your input feature dimension
    num_experts = 5  # Number of expert classes
    expert_gate_net = ExpertGateNet(in_channels=expert_in_channels, hidden_channels=hidden_dim, num_experts=num_experts).to(device)

    expert_gate_net_weights = torch.load("best_expert_type_net.pth", map_location=device)
    expert_gate_net_weights = {k: v for k, v in expert_gate_net_weights.items() if k in expert_gate_net.state_dict()}
    expert_gate_net.load_state_dict(expert_gate_net_weights, strict=False)
    # Load the pre-trained actor network weights
    # pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
    # actor_model.load_state_dict(pretrained_weights)
    clutter_weights = torch.load('best_imitation_model_except_tunnel_env.pth', map_location=device)
    # clutter_weights = torch.load('best_imitation_model_all_env.pth', map_location=device)

    
    clutter_weights = {k: v for k, v in clutter_weights.items() if k in clutter_actor_model.state_dict()}

    narrow_weights = torch.load('best_ppo_model_narrow_7.pth', map_location=device)
    narrow_weights = {k: v for k, v in narrow_weights.items() if k in narrow_actor_model.state_dict()}

    door_weights = torch.load('best_imitation_model_door_0.pth', map_location=device)
    door_weights = {k: v for k, v in door_weights.items() if k in door_actor_model.state_dict()}

    tunnel_weights = torch.load('best_imitation_model_tunnel_0.pth', map_location=device)
    tunnel_weights = {k: v for k, v in tunnel_weights.items() if k in tunnel_actor_model.state_dict()}
    # actor_model.load_state_dict(pretrained_weights)
    # pretrained_weights = torch.load('best_gnn_model.pth', map_location=device)
    clutter_actor_model.load_state_dict(clutter_weights, strict=False)
    narrow_actor_model.load_state_dict(narrow_weights, strict=False)
    door_actor_model.load_state_dict(door_weights, strict=False)
    tunnel_actor_model.load_state_dict(tunnel_weights, strict=False)

    assert not (save_render and not render), "To save the video you have to render it"

    dict_spaces = True  # Weather to return obs, rewards, and infos as dictionaries with agent names
    # (by default they are lists of len # of agents)
    collected_data = []  # 

    for current_collect_num in range(collect_num):
        env = VMASWrapper(
            scenario_name=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            n_agents=n_agents,
            env_type=env_type,
            is_evaluation_mode=False,
            is_imitation=is_imitation,
            working_mode = working_mode,
            )
        obs = env.get_obs()  # [num_envs, n_agents, obs_dim]

        frame_list = []  # For creating a gif
        init_time = time.time()
        step = 0
        env_obs_list = []
        print("current_collect_num:{}".format(current_collect_num))
        with torch.no_grad():
            for _ in range(n_steps):
                step += 1
                print(f"Step {step}")
                batch_size = len(obs)
                print("batch size:{}".format(batch_size))
                # VMAS actions can be either a list of tensors (one per agent)
                # or a dict of tensors (one entry per agent with its name as key)
                # Both action inputs can be used independently of what type of space its chosen
                dict_actions = random.choice([True, False])
                    
                
                batch_obs = Batch.from_data_list(obs).to(device)
                if env_obs_list == []:
                    action_mean, action_std = clutter_actor_model(batch_obs)  # Now returns action_std
                
                else:
                    env_obs = Batch.from_data_list(env_obs_list).to(device)
                    env_logits = expert_gate_net(env_obs)
                    print("env_logis:{}".format(env_logits))
                    env_class = torch.argmax(env_logits, dim=1)
                    print("leader_x:{}".format(leader_x))

                    if leader_x > 0 and leader_x < 8:
                        env_class = 3
                    else:
                        env_class = 1

                    # elif leader_x > -4 and leader_x < 0:
                        # env_class = 3
                    # elif leader_x > 0 and leader_x < 7:
                        # env_class = 3
                    # elif leader_x > 7:
                        # env_class = 3
                    # env_class = 1
                    print("env_class:{}".format(env_class))
                    if env_class == 0 or env_class == 1:
                        action_mean, action_std = clutter_actor_model(batch_obs)  # Now returns action_std
                    elif env_class == 2:
                        action_mean, action_std = narrow_actor_model(batch_obs)  # Now returns action_std
                    elif env_class == 3:
                        action_mean, action_std = tunnel_actor_model(batch_obs)  # Now returns action_std
                    elif env_class == 4:
                        action_mean, action_std = door_actor_model(batch_obs)  # Now returns action_std
                action_env = action_mean.view(batch_size, n_agents, -1).to(device)

                # print("actions:{}".format(action_env))
                obs, rews, dones, info = env.step(action_env)
                # print("obs:{}".format(obs))
                # print("info:{}".format(info))
                env_obs_list = info[0]["env_observation"]
                leader_x = info[0]["leader_x"]

                print("env_obs_list:{}".format(env_obs_list))
                # graph_list = agent_0_obs.get('graph_list', [])
                optimized_target_pos = {}

                # Process graph_list to make it serializable
                # serializable_graph_list = []

                # for graph in graph_list:
                #     # Extract data from the PyG Data object
                #     x = graph.x.cpu().numpy()  # Node features
                #     edge_index = graph.edge_index.cpu().numpy()  # Edge indices
                #     if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                #         edge_attr = graph.edge_attr.cpu().numpy()  # Edge attributes
                #     else:
                #         edge_attr = None

                #     graph_data = {
                #         'x': x,
                #         'edge_index': edge_index,
                #         'edge_attr': edge_attr,
                #     }
                #     serializable_graph_list.append(graph_data)
                # Collect optimized_target_pos for all agents
                for index, agent in enumerate(env.env.agents):
                    agent_name = agent.name

                    agent_info = info[index]
                    agent_pos = agent_info['optimized_target_pos'].cpu().detach().numpy()
                    optimized_target_pos[agent_name] = agent_pos

                # Build data point
                
                data_point = {
                    'graph_tensor': obs,
                    'optimized_target_pos': {agent_name: agent_pos for agent_name, agent_pos in optimized_target_pos.items()},
                }
                # Append to collected data
                collected_data.append(data_point)

                # print("obs:{}".format(obs))
                # print("info:{}".format(info))
                if render:
                    frame = env.render()
                        
                    if save_render:
                        frame_list.append(frame)

        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
            f"for {scenario_name} scenario."
        )

        with open('simple_narrow_7dim.pkl', 'wb') as f:
            pickle.dump(collected_data, f)
        print("Collected data saved to 'collected_data.pkl'.")

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)





if __name__ == "__main__":



    use_vmas_env(
        scenario_name="formation_control_teacher_graph_obs_cuda1",
        render=True,
        num_envs=1,
        n_steps=1000,
        save_render=False,
        random_action=True,
        continuous_actions=True,
        collect_num=200,
        is_imitation=True,
        device="cuda:1",
        env_type="mixed_in_distribution",
        working_mode="RL",
    )


    # use_vmas_env(
    #     scenario_name="formation_control_teacher",
    #     render=False,
    #     num_envs=1000,
    #     n_steps=550,
    #     save_render=False,
    #     random_action=True,
    #     continuous_actions=True,
    #     collect_num=2,
    # )


#collect data for imitation learning
#data includes:

#input:
# leader robot pose direction as reference frame, follower robot poses and directions in leader robot's reference frame
#obstacles positions in the leader robot's frame, obstacles number varied. 


#output:
#reformated follower robot's positions and directions. 