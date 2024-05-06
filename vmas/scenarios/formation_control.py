#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import copy
import math
import time
import typing
from typing import Callable, Dict, List
import numpy as np
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class RobotPathGenerator:
    def __init__(self):
        self.direction = np.array([0, 1])  # Initialize with a default direction, can be random too

    def generate_random_path(self, x_max, y_max, num_steps, max_step_size=0.04, direction_weight=0.8):
        # Initialize the path array
        path = np.zeros((num_steps, 2))
        # Start at a random position within bounds
        path[0, :] = [np.random.uniform(-x_max+2, x_max-2), np.random.uniform(-y_max+2, y_max-2)]

        for i in range(1, num_steps):
            # Generate a new random direction
            new_angle = np.random.uniform(0, 2 * np.pi)
            new_direction = np.array([np.cos(new_angle), np.sin(new_angle)])
            
            # Blend the new random direction with the previous direction
            self.direction = direction_weight * self.direction + (1 - direction_weight) * new_direction
            self.direction /= np.linalg.norm(self.direction)  # Normalize to maintain constant step size

            # Generate a random step size within the allowed maximum
            step_size = np.random.uniform(0, max_step_size)
            step = step_size * self.direction

            # Update the path, ensuring it stays within bounds
            if i > 200:
                path[i, 0] = min(max(path[i-1, 0] + step[0], -x_max), x_max)
                path[i, 1] = min(max(path[i-1, 1] + step[1], -y_max), y_max)
            else:
                path[i, 0] = path[i-1, 0]
                path[i, 1] = path[i-1, 1]

        return path

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.n_agents = kwargs.get("n_agents", 5)
        self.collisions = kwargs.get("collisions", True)
        self.viewer_size = (700, 700)
        self.plot_grid = True
        self.grid_spacing = 0.5
        self.device =device
        # self.agents_with_same_goal = kwargs.get("agents_with_same_goal", 1)
        # self.split_goals = kwargs.get("split_goals", False)
        self.observe_all_goals = kwargs.get("observe_all_goals", True)

        self.lidar_range = kwargs.get("lidar_range", 0.35)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", False)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.2)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 3
        self.min_collision_distance = 0.005
        generator = RobotPathGenerator()

        # Parameters
        x_max, y_max = 3.0, 3.0  # bounds
        num_steps = 1000  # number of steps in the path

        # Generate the path
        self.random_path = generator.generate_random_path(x_max, y_max, num_steps)
        # assert 1 <= self.agents_with_same_goal <= self.n_agents
        # if self.agents_with_same_goal > 1:
            # assert (
                # not self.collisions
            # ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        # if self.split_goals:
            # assert (
                # self.n_agents % 2 == 0
                # and self.agents_with_same_goal == self.n_agents // 2
            # ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        world = World(batch_dim, device, substeps=2)
        # world._x_semidim = 5
        # world._y_semidim = 5
        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)


        self.formation_center = Landmark(
                name=f"formation center",
                collide=False,
                movable=True,
                rotatable=True,
                color=Color.GRAY,
            )
        
        self.formation_center_pos = torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                    device=device,
                )
        self.formation_goals = {}
        self.formation_goals_landmark = {}
        for i in range(self.n_agents):
            self.formation_goals[i] = torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                    device=device,
                )
            self.formation_goals_landmark[i] = Landmark(
                name=f"formation goal{i}",
                collide=False,
                movable=True,
                rotatable=True,
                color=Color.GRAY,
            )
            world.add_landmark(self.formation_goals_landmark[i])
            
        world.add_landmark(self.formation_center)

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)

                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=12,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            
            agent.goal = copy.deepcopy(self.formation_goals_landmark[i])

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.keep_track_time = torch.zeros(batch_dim, device=device)
        self.update_formation_assignment_time = torch.zeros(batch_dim, device=device)
        self.current_assignments = None

        return world

    def reset_world_at(self, env_index: int = None):
        self.update_formation_assignment_time[env_index] = time.time()
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )
        ScenarioUtils.spawn_entities_randomly(
            self.world.landmarks,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )
        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_semidim, self.world_semidim),
                y_bounds=(-self.world_semidim, self.world_semidim),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            # if self.split_goals:
            #     goal_index = int(i // self.agents_with_same_goal)
            # else:
            #     goal_index = 0 if i < self.agents_with_same_goal else i

            # agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )
        # if env_index is None:
        #     self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
        # else:
        #     self.t[env_index] = 0
        # if env_index is None:
            # self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
        # else:
        self.t = 0

    

    def process_action(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            
            formation_movement = "random"
            if formation_movement == "random":

                ###################  random walk ######################
                
                self.formation_center.set_pos(
                    torch.tensor(
                        [
                            self.random_path[self.t][0],
                            self.random_path[self.t][1],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=None,
                )
                self.formation_center_pos[0] = self.random_path[self.t][0]
                self.formation_center_pos[1] = self.random_path[self.t][1]
                ###################  random walk end######################
            elif formation_movement == "circle":
                ###################  circling ######################
                t = self.t / 30
                self.formation_center.set_pos(
                    torch.tensor(
                        [
                            math.cos(t),
                            math.sin(t),
                        ],
                        device=self.world.device,
                    ),
                    batch_index=None,
                )
                self.formation_center_pos[0] = math.cos(t)
                self.formation_center_pos[1] = math.sin(t)
                ###################  circling end ######################




            self.formation_center.set_rot(
                torch.tensor(
                    torch.pi,
                    device=self.world.device,
                ),
                batch_index=None,
            )
            self.formation_center_pos[2] = torch.pi
        
        angles = [-135/180.0*math.pi, -135/180.0*math.pi, 135/180.0*math.pi, 135/180.0*math.pi]
        dists = [0.5, 1, 0.5, 1]
        for i, world_agent in enumerate(self.world.agents):
            if agent == world_agent:
                if i == 0:
                    self.formation_goals[i][0] = self.formation_center_pos[0]
                    self.formation_goals[i][1] = self.formation_center_pos[1]
                    self.formation_goals[i][2] = self.formation_center_pos[2]
                else:


                    # formation_type = "ren_shape"
                    # formation_type = "line"
                    formation_type = "rectangle"
                    if formation_type == "ren_shape":
                        #大雁人字形
                        self.formation_goals[i][0] = self.formation_center_pos[0] + math.cos(self.formation_center_pos[2] + angles[i-1]) * dists[i-1]
                        self.formation_goals[i][1] = self.formation_center_pos[1] + math.sin(self.formation_center_pos[2] + angles[i-1]) * dists[i-1]
                    elif formation_type == "line":
                        #直线型
                        dists = [0.5, 1, -0.5, -1]
                        self.formation_goals[i][0] = self.formation_center_pos[0] + dists[i-1]
                        self.formation_goals[i][1] = self.formation_center_pos[1]
                    elif formation_type == "rectangle":
                        #矩形
                        displacement_x = [1, -1, 1, -1]
                        displacement_y = [1, 1, -1, -1]
                        self.formation_goals[i][0] = self.formation_center_pos[0] + displacement_x[i-1]
                        self.formation_goals[i][1] = self.formation_center_pos[1] + displacement_y[i-1]

                self.formation_goals_landmark[i].set_pos(
                torch.tensor(
                    [
                        self.formation_goals[i][0],
                        self.formation_goals[i][1],
                    ],
                    device=self.world.device,
                ),
                batch_index=None,
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        # if is_first:
        #     print("agent:{}".format(0))
        # else:
        #     print("agent else")
        # if self.shared_rew:
        #     print("shared reward")
        # else:
        #     print("not shared reward")

        if is_first:
            self.t += 1
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew
    
    def agent_reward(self, agent: Agent):
        # Extract positions
        agent_reward_start = time.time()
        agent_positions = torch.stack([a.state.pos for a in self.world.agents])  # [num_agents, 1200, 2]
        goal_positions = torch.stack([g.state.pos for g in self.formation_goals_landmark.values()])  # [num_goals, 1200, 2]
        # print("agent pos:{}".format(agent_positions))
        # print("goal_positions:{}".format(goal_positions))
        # print("agent_positions shape:{}".format(agent_positions.shape))
        # print("goal positions shape:{}".format(goal_positions.shape))
        num_agents = agent_positions.shape[0]
        num_goals = goal_positions.shape[0]
        num_envs = agent_positions.shape[1]  # 1200 environments
        distances = torch.zeros((num_agents, num_goals, num_envs), device=self.device)

        # Calculate distances
        for i in range(num_agents):
            for j in range(num_goals):
                # print("i,j:{},{}".format(i, j))
                # print("agent shape:{}".format(agent_positions[i, :, :].shape))
                # print("goal shape:{}".format(goal_positions[j, :, :].shape))
                temp = agent_positions[i, :, :] - goal_positions[j, :, :]
                # print("minus shape:{}".format(temp.shape))
                distances[i, j, :] = torch.linalg.vector_norm(temp, dim=-1)
        # print("distances:{}".format(distances))
        # Allocate storage for agent-specific rewards
        # agent_rewards = torch.zeros(num_envs)

        agent.pos_rew = torch.zeros(num_envs, device =self.device )

        # Get assignments from the batch_greedy_assignment

        assignments = self.batch_greedy_assignment(distances)
        # print("assignments:{}".format(assignments))
        # print("after assignment")
        # Get the assigned goal for the current agent in each environment
        current_agent_index = self.world.agents.index(agent)
        assigned_goals = assignments[:, current_agent_index]
        print("assigned_goals:{}".format(assigned_goals))
        # print("assigned goals shape:{}".format(assigned_goals.shape))
        # print("formation_goals_landmark:{}".format(self.formation_goals_landmark[0].state.pos))
        print("num_envs:{}".format(num_envs))
        for e in range(num_envs):
            # print("agent.goal.state.pos:{}".format(agent.goal.state.pos))
            agent.goal.state.pos[e] = copy.deepcopy(self.formation_goals_landmark[assigned_goals[e].item()].state.pos[0])

        print("agent goal shape:{}".format(agent.goal.state.pos))

        # print("assign goals:{}".format(assigned_goals))
        # # Compute rewards based on these assignments
        # for env_idx in range(num_envs):
        #     # print(env_idx)
        #     assigned_goal_index = assigned_goals[env_idx]
        #     if assigned_goal_index != -1:  # Check if an assignment was made
        #         min_distance = distances[current_agent_index, assigned_goal_index, env_idx]

        #         # Check if the agent is on its goal
        #         agent.on_goal = min_distance < agent.goal.shape.radius

        #         # Calculate position shaping reward
        #         pos_shaping = min_distance * self.pos_shaping_factor
        #         agent.pos_rew[env_idx] = agent.pos_shaping[env_idx] - pos_shaping
        #         agent.pos_shaping[env_idx] = pos_shaping
        #     else:
        #         agent.on_goal = False
        #         agent.pos_rew[env_idx] = 0  # No assignment was possible

        # # Record the time taken for computation
        # agent_reward_time = time.time() - agent_reward_start
        # # print("Agent reward computation time:{}".format(agent_reward_time))
        # return agent.pos_rew
        rows = torch.arange(num_envs)
        # print("after assignment distances:{}".format(distances))

        min_distances = distances[current_agent_index, assigned_goals, rows]
        print("min_distances:{}".format(min_distances))
        # Check which assignments were made (assigned_goal_index != -1)
        valid_assignments = assigned_goals != -1
        
        # Calculate whether the agent is on its goal for all environments
        agent_on_goal = min_distances < 0.05
        agent_on_goal = agent_on_goal & valid_assignments  # Only consider valid assignments
        
        # Calculate position shaping reward for all environments
        pos_shaping = min_distances * self.pos_shaping_factor
        pos_rew = agent.pos_shaping - pos_shaping
        pos_rew = torch.where(valid_assignments, pos_rew, torch.zeros_like(pos_rew))  # Set rewards to 0 where no assignment
        
        # Update agent's rewards and position shaping
        agent.pos_rew = pos_rew
        agent.pos_shaping = torch.where(valid_assignments, pos_shaping, agent.pos_shaping)  # Only update shaping where valid
        
        # Update on_goal status
        agent.on_goal = agent_on_goal
        agent_reward_time = time.time() - agent_reward_start

        # print("Agent reward computation time:{}".format(agent_reward_time))

        return agent.pos_rew

    # def agent_reward(self, agent: Agent):
    #     # agent_goal_dist_dict = {}
    #     distances = []

    #     # print("formation-goals:{}".format(self.formation_goals_landmark))
    #     for i, goal_landmark in self.formation_goals_landmark.items():
    #         # print(goal_landmark)
    #         # print("agent pos shape:{}".format(agent.state.pos.shape))
    #         agent_goal_dist = torch.linalg.vector_norm(
    #         agent.state.pos - goal_landmark.state.pos,
    #         dim=-1,
    #         )
    #         # print("agent_goal_dist shape:{}".format(agent_goal_dist.shape))
    #         distances.append(agent_goal_dist)
    #         # agent_goal_dist_dict[i] = agent_goal_dist
    #     # min_index, min_distance = min(agent_goal_dist_dict.items(), key=lambda item: item[1])
    #     all_distances = torch.stack(distances)
    #     min_distances, min_indices = torch.min(all_distances, dim=0)

    #     agent.on_goal = min_distances < agent.goal.shape.radius

    #     pos_shaping = min_distances * self.pos_shaping_factor
    #     agent.pos_rew = agent.pos_shaping - pos_shaping
    #     agent.pos_shaping = pos_shaping
    #     return agent.pos_rew

    #when robot and goal is one-on-one 
    # def agent_reward(self, agent: Agent):
    #     agent.distance_to_goal = torch.linalg.vector_norm(
    #         agent.state.pos - agent.goal.state.pos,
    #         dim=-1,
    #     )
    #     agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

    #     pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
    #     agent.pos_rew = agent.pos_shaping - pos_shaping
    #     agent.pos_shaping = pos_shaping
    #     return agent.pos_rew



    def batch_greedy_assignment(self, distances):
        num_agents, num_goals, num_envs = distances.shape
        working_distances = distances.clone()
        assignments = -torch.ones((num_envs, num_agents), dtype=torch.long, device=distances.device)
        available_agents = torch.ones((num_envs, num_agents), dtype=torch.bool, device=distances.device)
        available_tasks = torch.ones((num_envs, num_goals), dtype=torch.bool, device=distances.device)

        while available_agents.any():
            # print("distances shape:{}".format(distances.shape))
            # print("available_tasks shape:{}".format(available_tasks.unsqueeze(1).shape))
            # task_mask = available_tasks.unsqueeze(2).expand(-1, -1, num_agents).transpose(1, 2)
            task_mask = available_tasks.unsqueeze(1).expand(-1, num_agents, -1).transpose(0, 2)
            # Update inf_mask within the loop to reflect current task availability
            # print("task mask shape:{}".format(task_mask.shape))
            # print("task mask:{}".format(task_mask))
            inf_mask = torch.where(task_mask, torch.zeros_like(working_distances), torch.inf)
            # Apply the updated inf_mask and compute valid distances
            valid_distances = torch.where(task_mask, working_distances, inf_mask)
            # Update inf_mask within the loop to reflect current task availability
            # inf_mask = torch.where(available_tasks.unsqueeze(1), torch.zeros_like(distances), torch.inf)
            

            # print("inf_mask shape:{}".format(inf_mask.shape))
            
            # Apply inf_mask and compute valid distances
            # valid_distances = torch.where(available_tasks.unsqueeze(1), distances, inf_mask)

            # Flatten the tensor for each environment and find the minimum cost assignment per environment
            # print("distances:{}".format(distances))
            # print("distances shape:{}".format(distances.shape))
            # print("valid_distances:{}".format(valid_distances))

            transposed_distances = working_distances.permute(2, 0, 1)  # Now shape [num_envs, num_agents, num_goals]

            # Flatten the agent-goal dimensions for each environment
            flattened_distances = transposed_distances.view(num_envs, -1)

            # Find the minimum distance and corresponding index for each environment
            min_values, min_indices = flattened_distances.min(dim=1)
            # min_values, min_indices = distances.view(num_envs, -1).min(dim=1)
            agent_indices = min_indices // num_goals
            task_indices = min_indices % num_goals
            # print("agent_index:{}".format(agent_indices))
            # print("task_index:{}".format(task_indices))

            # Create masks for updating based on selected indices
            env_indices = torch.arange(num_envs, device=working_distances.device)
            update_mask = available_agents[env_indices, agent_indices] & available_tasks[env_indices, task_indices]
            # print("update_mask:{}".format(update_mask))
            # Update assignments and availability using masks
            assignments[env_indices[update_mask], agent_indices[update_mask]] = task_indices[update_mask]
            available_agents[env_indices[update_mask], agent_indices[update_mask]] = False
            available_tasks[env_indices[update_mask], task_indices[update_mask]] = False

            # Set all agents' distances to this task to infinity to avoid reassignment
            working_distances[:, task_indices, env_indices] = torch.inf
            working_distances[agent_indices, :, env_indices] = torch.inf

        return assignments




    def observation(self, agent: Agent):
        goal_poses = []
        observations = [agent.state.pos, agent.state.vel]

    # Add positions and velocities of all agents (including the current agent)
        # for a in self.world.agents:
            # observations.append(a.state.pos)
            # observations.append(a.state.vel)

        # if self.observe_all_goals:
        #     for g in self.formation_goals_landmark.values():
        #         goal_poses.append(agent.state.pos - g.state.pos.clone())
        # else:
        goal_poses.append(agent.state.pos - agent.goal.state.pos)
        
        observation_tensor = torch.cat(
            observations + goal_poses + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1
        )
        return observation_tensor

    # def observation(self, agent: Agent):
    #     goal_poses = []
    #     observations = [agent.state.pos, agent.state.vel]

    # # Add positions and velocities of all agents (including the current agent)
    #     for a in self.world.agents:
    #         observations.append(a.state.pos)
    #         observations.append(a.state.vel)

    #     if self.observe_all_goals:
    #         for g in self.formation_goals_landmark.values():
    #             goal_poses.append(agent.state.pos - g.state.pos)
    #     else:
    #         goal_poses.append(agent.state.pos - agent.goal.state.pos)
        
    #     observation_tensor = torch.cat(
    #         observations + goal_poses + (
    #             [agent.sensors[0]._max_range - agent.sensors[0].measure()]
    #             if self.collisions
    #             else []
    #         ),
    #         dim=-1
    #     )
    #     return observation_tensor
      
    def done(self):
        reached_goals = torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                ) < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

        # Update keep_track_time based on whether agents have reached their goals
        for i in range(self.world.batch_dim):
            if reached_goals[i]:
                self.keep_track_time[i] += 1
            else:
                self.keep_track_time[i] = 0

        # Check if the keep_track_time meets or exceeds the keep limit for any environment
        done_status = self.keep_track_time >= 30
        if torch.any(done_status):
            # Count the number of True entries
            num_true = torch.sum(done_status).item()  # .item() to get the value as a Python scalar
            print("Number of True entries:", num_true)
        else:
            pass
            # print("No True entries found.")
        return done_status

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.2, clf_slack=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_pos = (-1.0) * (observation[:, 4:6] - agent_pos)

        # Pre-compute tensors for the CLF and CBF constraints,
        # Lyapunov Function from: https://arxiv.org/pdf/1903.03692.pdf

        # Laypunov function
        V_value = (
            (agent_pos[:, X] - goal_pos[:, X]) ** 2
            + 0.5 * (agent_pos[:, X] - goal_pos[:, X]) * agent_vel[:, X]
            + agent_vel[:, X] ** 2
            + (agent_pos[:, Y] - goal_pos[:, Y]) ** 2
            + 0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) * agent_vel[:, Y]
            + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * (agent_pos[:, X] - goal_pos[:, X]) + agent_vel[:, X]) * (
            agent_vel[:, X]
        ) + (2 * (agent_pos[:, Y] - goal_pos[:, Y]) + agent_vel[:, Y]) * (
            agent_vel[:, Y]
        )
        LgV_vals = torch.stack(
            [
                0.5 * (agent_pos[:, X] - goal_pos[:, X]) + 2 * agent_vel[:, X],
                0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )
        # Define Quadratic Program (QP) based controller
        u = cp.Variable(2)
        V_param = cp.Parameter(1)  # Lyapunov Function: V(x): x -> R, dim: (1,1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(
            2
        )  # Lie derivative of Lyapunov Function, dim: (1, action_dim)
        clf_slack = cp.Variable(1)  # CLF constraint slack variable, dim: (1,1)

        constraints = []

        # QP Cost F = u^T @ u + clf_slack**2
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack**2)

        # control bounds between u_range
        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        # CLF constraint
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)

        # Initialize CVXPY layers
        QP_controller = CvxpyLayer(
            QP_problem,
            parameters=[V_param, lfV_param, lgV_params],
            variables=[u],
        )

        # Solve QP
        CVXpylayer_parameters = [
            V_value.unsqueeze(1),
            LfV_val.unsqueeze(1),
            LgV_vals,
        ]
        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[
            0
        ]

        return action


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
