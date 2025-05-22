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
        batch_size = obs[0]['laser'].shape[0]
        n_agents = len(obs)
        
        # Pre-allocate tensors for each agent's observations
        laser_obs = []
        relative_pos_obs = []
        nominal_pos_obs = []
        leader_vel = []
        leader_ang_vel = []
        # Collect observations by type
        for agent_index in range(n_agents):
            laser_obs.append(obs[agent_index]['laser'])
            relative_pos_obs.append(obs[agent_index]['relative_pos'])
            nominal_pos_obs.append(obs[agent_index]['nominal_pos'])
            leader_vel.append(obs[agent_index]['leader_vel'])
            leader_ang_vel.append(obs[agent_index]['leader_ang_vel'])

        # Stack observations along agent dimension (dim=1)
        # laser_tensor shape: [batch_size, n_agents, laser_dim]
        laser_tensor = torch.stack(laser_obs, dim=1)
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("laser:{}".format(laser_tensor))
        # relative_pos_tensor shape: [batch_size, n_agents, pos_dim]
        relative_pos_tensor = torch.stack(relative_pos_obs, dim=1)
        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        # print("relative pos:{}".format(relative_pos_tensor))
        nominal_pos_tensor = torch.stack(nominal_pos_obs, dim=1)
        leader_vel_tensor = torch.stack(leader_vel, dim=1)
        leader_ang_vel_tensor = torch.stack(leader_ang_vel, dim=1)
        # print("nominal pos tensor shape:{}".format(nominal_pos_tensor.shape))    #[2, 5, 3]
        # print("nominal pos:{}".format(nominal_pos_tensor))
        # print("laser tensor shape:{}".format(laser_tensor.shape))   #   [2, 5, 20]
        # print("relative pos tensor shape:{}".format(relative_pos_tensor.shape))    #[2, 5, 3]
        
        # Get feature dimensions
        laser_dim = laser_tensor.shape[-1]
        pos_dim = relative_pos_tensor.shape[-1]
        nominal_dim = nominal_pos_tensor.shape[-1]
        leader_vel_dim = leader_vel_tensor.shape[-1]
        leader_ang_vel_dim = leader_ang_vel_tensor.shape[-1]
        # Reshape tensors to combine batch and agent dimensions for concatenation
        # [batch_size, n_agents, feature_dim] -> [batch_size * n_agents, feature_dim]
        laser_flat = laser_tensor.reshape(-1, laser_dim)
        relative_pos_flat = relative_pos_tensor.reshape(-1, pos_dim)
        nominal_pos_flat = nominal_pos_tensor.reshape(-1, nominal_dim)
        leader_vel_flat = leader_vel_tensor.reshape(-1, leader_vel_dim)
        leader_ang_vel_flat = leader_ang_vel_tensor.reshape(-1, leader_ang_vel_dim)
        # Concatenate features along feature dimension
        # [batch_size * n_agents, combined_dim]
        combined = torch.cat([laser_flat, relative_pos_flat, nominal_pos_flat, leader_vel_flat, leader_ang_vel_flat], dim=1)
        
        # Reshape back to [batch_size, n_agents, combined_dim]
        combined_dim = laser_dim + pos_dim + nominal_dim + leader_vel_dim + leader_ang_vel_dim
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
        print("dones:{}".format(dones))
        return obs, summed_rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=False,
                    )



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
            scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap2",
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
                # dict_actions = random.choice([True, False])
                dict_actions = False
                actions = {} if dict_actions else []
                for agent in env.env.agents:
                    # print("agent:{}".format(agent))
                    
                        # action = env.get_random_action(agent)
                    action = torch.zeros((agent.batch_dim, 3))
                    if dict_actions:
                        actions.update({agent.name: action})
                    else:
                        actions.append(action)
                print("actions:{}".format(actions))
                print("actions[0] shape:{}".format(actions[0].shape))
                obs, rews, dones, info = env.step(actions)
                # print("obs:{}".format(obs))
                # print("info:{}".format(info))
                # print("dones:{}".format(dones))
                all_env_done = True
                for idx in range(num_envs):
                    if dones[idx] == False:
                        all_env_done = False
                        break
                if all_env_done:
                    break
                    # break
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
                        'current_agent_vel': {agent_name: agent_vel[idx, :] for agent_name, agent_vel in current_agent_vel.items()},
                    }


                    if last_obs is not None:
                        world_model_data_point = {
                            'last_graph_tensor': last_obs[idx],
                            'current_graph_tensor': single_obs,
                            'current_agent_vel': {agent_name: agent_vel for agent_name, agent_vel in current_agent_vel.items()},
                            'action': actions,
                        }
                    # Append to collected data
                        # world_model_data.append(world_model_data_point)
                    
                    if dones[idx] == False:
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
            f"It took: {total_time}s for {step} steps of {num_envs} parallel environments on device {device} "
            f"for {scenario_name} scenario."
        )
        if current_collect_num % 20 == 0:
            with open(filename, 'wb') as f:
                pickle.dump(collected_data, f)

            # with open(world_model_filename, 'wb') as f:
            #     pickle.dump(world_model_data, f)
            print("Collected data saved to 'collected_data.pkl'.")

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.env.scenario.world.dt)





if __name__ == "__main__":

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
        scenario_name="formation_control_teacher_graph_obs_cuda1_bitmap2",
        render=False,
        num_envs=20,
        n_steps=700,
        save_render=False,
        random_action=True,
        continuous_actions=True,
        collect_num=200,
        is_imitation=True,
        device="cpu",
        env_type="narrow",
        working_mode="imitation",
        filename="collected_data_complex_0.pkl",
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