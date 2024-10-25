#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
import time

import torch

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

    for current_collect_num in range(collect_num):
        env = make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            dict_spaces=dict_spaces,
            wrapper=None,
            seed=None,
            # Environment specific variables
            n_agents=n_agents,
        )

        frame_list = []  # For creating a gif
        init_time = time.time()
        step = 0


        print("current_collect_num:{}".format(current_collect_num))
        with torch.no_grad():
            for _ in range(n_steps):
                step += 1
                # print(f"Step {step}")

                # VMAS actions can be either a list of tensors (one per agent)
                # or a dict of tensors (one entry per agent with its name as key)
                # Both action inputs can be used independently of what type of space its chosen
                dict_actions = random.choice([True, False])

                actions = {} if dict_actions else []
                for agent in env.agents:
                    # print("agent:{}".format(agent))
                    
                        # action = env.get_random_action(agent)
                    action = torch.zeros((agent.batch_dim, 3))
                    if dict_actions:
                        actions.update({agent.name: action})
                    else:
                        actions.append(action)
                # print("actions:{}".format(actions))
                obs, rews, dones, info = env.step(actions)
                # print("obs:{}".format(obs))
                # print("info:{}".format(info))
                agent_0 = env.agents[0]
                agent_0_name = agent_0.name
                agent_0_obs = obs[agent_0_name]

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
                for agent in env.agents:
                    agent_name = agent.name
                    agent_info = info[agent_name]
                    agent_pos = agent_info['optimized_target_pos'].cpu().detach().numpy()
                    optimized_target_pos[agent_name] = agent_pos

                # Build data point
                
                data_point = {
                    'graph_tensor': agent_0_obs.detach().cpu(),
                    'optimized_target_pos': {agent_name: agent_pos for agent_name, agent_pos in optimized_target_pos.items()},
                }
                # Append to collected data
                collected_data.append(data_point)

                # print("obs:{}".format(obs))
                # print("info:{}".format(info))
                if render:
                    frame = env.render(
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

        with open('collected_data_random_500_steps_100_env.pkl', 'wb') as f:
            pickle.dump(collected_data, f)
        print("Collected data saved to 'collected_data.pkl'.")

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)





if __name__ == "__main__":

    use_vmas_env(
        scenario_name="formation_control_teacher",
        render=False,
        num_envs=100,
        n_steps=500,
        save_render=False,
        random_action=True,
        continuous_actions=True,
        collect_num=20,
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