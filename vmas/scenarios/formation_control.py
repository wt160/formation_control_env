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
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box, Line
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class RobotPathGenerator:
    def __init__(self):
        self.direction = np.array([0, 1])  # Initialize with a default direction, can be random too

    def generate_random_path(self, x_max, y_max, num_steps, max_step_size=0.04, direction_weight=0.8):
        # Initialize the path array
        path = np.zeros((num_steps, 3))
        # Start at a random position within bounds
        path[0, :] = [np.random.uniform(-x_max+2, x_max-2), np.random.uniform(-y_max+2, y_max-2), np.random.uniform(0, 2*np.pi)]

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
            if i > 50:
                path[i, 0] = min(max(path[i-1, 0] + step[0], -x_max), x_max)
                path[i, 1] = min(max(path[i-1, 1] + step[1], -y_max), y_max)
                path[i, 2] = path[i-1, 2] + 0.01
            else:
                path[i, 0] = path[i-1, 0]
                path[i, 1] = path[i-1, 1]

        return path

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.n_agents = kwargs.get("n_agents", 4)
        self.init_positions_noise_level = kwargs.get("init_position_noise_level", 0)
        self.collisions = kwargs.get("collisions", True)
        self.viewer_size = (1000, 700)
        self.plot_grid = True
        self.grid_spacing = 1
        self.device =device
        # self.agents_with_same_goal = kwargs.get("agents_with_same_goal", 1)
        # self.split_goals = kwargs.get("split_goals", False)
        self.observe_all_goals = kwargs.get("observe_all_goals", True)

        self.lidar_range = kwargs.get("lidar_range", 1.8)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", False)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.2)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -100)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 5
        self.min_collision_distance = 0.005
        generator = RobotPathGenerator()
        self.current_formation_type = "ren_shape"
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
        world._x_semidim = 7
        world._y_semidim = 5
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

        self.leader_robot = Landmark(
                name=f"leader_robot",
                collide=False,
                movable=True,
                rotatable=True,
                color=Color.RED,
            )
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
        self.formation_goals_modified = {}
        for i in range(self.n_agents):
            self.formation_goals[i] = torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                    device=device,
                )
            modified_color = (
                known_colors[i]
                if i < len(known_colors)

                else colors[i - len(known_colors)]
            )
            self.formation_goals_landmark[i] = Landmark(
                name=f"formation goal{i}",
                collide=False,
                movable=True,
                rotatable=True,
                color=Color.GRAY,
            )
            world.add_landmark(self.formation_goals_landmark[i])
            self.formation_goals_modified[i] = Landmark(
                name=f"modified_goal_{i}",
                collide=False,
                shape=Sphere(radius = 0.1),
                movable=True,
                rotatable=True,
                color=modified_color,
            )
            # world.add_landmark(self.formation_goals_modified[i])
        world.add_landmark(self.formation_center)
        world.add_landmark(self.leader_robot)
        # print("formation_center shape:{}".format(self.formation_center.state.pos.shape))
        # self.walls = []
        # for i in range(4):
        #     wall = Landmark(
        #         name=f"wall {i}",
        #         collide=True,
        #         shape=Line(length=world._x_semidim* 2 + 0.1),
        #         color=Color.BLACK,
        #     )
        #     world.add_landmark(wall)
        #     self.walls.append(wall)

        self.obstacle_pattern = 2
        self.create_obstacles(self.obstacle_pattern, world)

        def detect_obstacles(x):
            return x.name.startswith("obs_") or x.name.startswith("agent_") or x.name.startswith("wall")

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
                dynamics=HolonomicWithRotation(), 
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=20,
                            max_range=self.lidar_range,
                            entity_filter=detect_obstacles,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.angle_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            agent.target_distance = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)
            if i == 0:
                agent.set_pos(
                    torch.tensor(
                        [
                            -4,
                            0,
                        ],
                        device=device,
                    ),
                    batch_index=None,
                )
            elif i == 1:
                #-3.6464,  0.3536
                agent.set_pos(    
                    torch.tensor(
                        [
                            -3.6464,
                            0.3536,
                        ],
                        device=device,
                    ),
                    batch_index=None,
                )
            elif i == 2:
                #-3.6464, -0.3536
                agent.set_pos(    
                    torch.tensor(
                        [
                            -3.6464,
                            -0.3536,
                        ],
                        device=device,
                    ),
                    batch_index=None,
                )
            # Add goals
            
            agent.goal = self.formation_goals_landmark[i]

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.angle_rew = torch.zeros(batch_dim, device=device)
        self.formation_maintain_rew = torch.zeros(batch_dim, device=device) 
        self.final_rew = self.pos_rew.clone()
        self.keep_track_time = torch.zeros(batch_dim, device=device)
        self.update_formation_assignment_time = torch.zeros(batch_dim, device=device)
        self.current_assignments = None

        return world

    def create_obstacles(self, obstacle_pattern, world: World):
        self.obstacles = []

        if obstacle_pattern == 0:
            #random located obstalces
            self.n_boxes = 10
            self.box_width = 0.5
            for i in range(self.n_boxes):
                obs = Landmark(
                    name=f"obs_{i}",
                    collide=True,
                    movable=False,
                    shape=Box(length=self.box_width, width=self.box_width),
                    color=Color.RED,
                    collision_filter=lambda e: not isinstance(e.shape, Box),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)

        elif obstacle_pattern == 1:
            #two large boxes, short corridor
            self.n_boxes = 2
            self.corridor_width = 5.0
            self.corridor_length = 7
            for i in range(self.n_boxes):
                obs = Landmark(
                    name=f"obs_{i}",
                    collide=True,
                    movable=False,
                    shape=Box(length=self.corridor_length, width=self.corridor_width),
                    color=Color.RED,
                    collision_filter=lambda e: not isinstance(e.shape, Box),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
            pass
        elif obstacle_pattern == 2:
            #two large boxes, relatively large corridor
            self.n_boxes = 2
            self.corridor_width = 4
            self.corridor_length = 7
            for i in range(self.n_boxes):
                obs = Landmark(
                    name=f"obs_{i}",
                    collide=True,
                    movable=False,
                    shape=Box(length=self.corridor_length, width=self.corridor_width),
                    color=Color.RED,
                    collision_filter=lambda e: not isinstance(e.shape, Box),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
            pass

    def spawn_obstacles(self, obstacle_pattern, env_index):
        passage_indexes = []
        j = self.n_boxes // 2


        def get_pos(i):
            if obstacle_pattern == 0:
                #random located obstalces

                pos = torch.tensor(
                    [
                        np.random.uniform(-self.world_semidim, self.world_semidim),
                        np.random.uniform(-self.world_semidim, self.world_semidim),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ).repeat(i.shape[0], 1)
                return pos
            elif obstacle_pattern == 1:
                is_zero = (i == 0)
                is_one = (i == 1)
                is_two = (i == 2)
                if is_zero.any():
                    pos = torch.tensor(
                        [
                            1,
                            2.75,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
                elif is_one.any():
                    pos = torch.tensor(
                    [
                        1,
                        -2.75,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
                elif is_two.any():
                    pos = torch.tensor(
                    [
                        3,
                        3,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
            elif obstacle_pattern == 2:
                is_zero = (i == 0)
                is_one = (i == 1)
                is_two = (i == 2)
                if is_zero.any():
                    pos = torch.tensor(
                        [
                            0,
                            2.5,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
                elif is_one.any():
                    pos = torch.tensor(
                    [
                        0,
                        -2.5,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
                elif is_two.any():
                    pos = torch.tensor(
                    [
                        3,
                        3,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
        i = torch.zeros(
            (self.world.batch_dim,) if env_index is None else (1,),
            dtype=torch.int,
            device=self.world.device,
        )
        
        for obs in self.obstacles:
            obs.set_pos(get_pos(i), batch_index=env_index)
            
            i += 1


    def reset_world_at(self, env_index: int = None):
        self.update_formation_assignment_time[env_index] = time.time()
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim+1, -3),
            (-self.world_semidim+4, self.world_semidim-4),
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
        noise_scale = self.init_positions_noise_level
        for i, agent in enumerate(self.world.agents):
            noise = torch.randn(2) * noise_scale
            noise = noise.to(self.device)
            if i == 0:
                agent.set_pos(
                    torch.tensor(
                        [
                            -4.3536, 
                            -0.3536,
                        ],
                        device=self.world.device,
                    ) + noise,
                    batch_index=env_index,
                )
            elif i == 1:
                #-3.6464,  0.3536
                agent.set_pos(    
                    torch.tensor(
                        [
                            -4.3536,  
                            0.3536,
                        ],
                        device=self.world.device,
                    ) + noise,
                    batch_index=env_index,
                )
            elif i == 2:
                #-3.6464, -0.3536
                agent.set_pos(    
                    torch.tensor(
                        [ 
                            -4.7071,
                            -0.7071,
                        ],
                        device=self.world.device,
                    ) + noise,
                    batch_index=env_index,
                )
            elif i == 3:
                #-3.6464, -0.3536
                agent.set_pos(    
                    torch.tensor(
                        [
                            -4.7071,
                            0.7071,
                        ],
                        device=self.world.device,
                    ) + noise,
                    batch_index=env_index,
                )
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

            agent.goal.set_pos(goal_poses[i], batch_index=env_index)

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
            if env_index is None:
                agent.angle_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.rot - agent.goal.state.rot,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.angle_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.rot[env_index] - agent.goal.state.rot[env_index]
                    )
                    * self.pos_shaping_factor
                )
        for i, agent in enumerate(self.world.agents):
            current_agent_index = self.world.agents.index(agent)
            target_positions_dict = {}
            target_rotations_dict = {}
            for a_index, agent in enumerate(self.world.agents):
                target_positions_dict[a_index] = agent.state.pos
                target_rotations_dict[a_index] = agent.state.rot

            batch_size = next(iter(target_positions_dict.values())).shape[0]
            num_agents = len(target_positions_dict)

            formation_goals_positions = []
            for i in range(num_agents):
                formation_goals_positions.append(self.formation_goals_landmark[i].state.pos[0])
            formation_goals_positions = torch.stack(formation_goals_positions)

            # print("formation_goals_positions shape{},:{}".format(formation_goals_positions.shape, formation_goals_positions))
            # print("target_positions:{}".format(target_positions_dict))

            for batch_idx in range(batch_size):
                # Extract positions for the current batch
                target_positions = torch.stack([target_positions_dict[i][batch_idx] for i in range(num_agents)])

                # Calculate the cost matrix based on distances
                cost_matrix = torch.cdist(target_positions, formation_goals_positions)

                # Solve the assignment problem
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
                
                # Find the index in the assignment for the current agent
                agent_assignment_idx = row_ind.tolist().index(current_agent_index)
                assigned_goal_idx = col_ind[agent_assignment_idx]
                
                # Calculate the distance from the current agent to its assigned goal
                agent_position = target_positions[current_agent_index]
                assigned_goal_position = formation_goals_positions[assigned_goal_idx]
                distance_to_goal = torch.norm(agent_position - assigned_goal_position)
                # Normalize the distance to get the reward (assuming max reward is 1.0)
                # max_distance = torch.norm(formation_goals_positions - formation_goals_positions.mean(dim=0), dim=1).max().item()
                # reward = 1.0 - (distance_to_goal.item() / max_distance)
                # reward = max(reward, 0.0)  # Ensure the reward is non-negative
                agent.target_distance[batch_idx] = distance_to_goal

        # if env_index is None:
        #     self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
        # else:
        #     self.t[env_index] = 0
        # if env_index is None:
            # self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
        # else:
        self.spawn_obstacles(self.obstacle_pattern, env_index)
        # self.spawn_walls(env_index)
        self.t = 0

    def spawn_walls(self, env_index):
        for i, wall in enumerate(self.walls):
            wall.set_pos(
                torch.tensor(
                    [
                        (
                            0.0
                            if i % 2
                            else (
                                self.world.x_semidim + self.agent_radius
                                if i == 0
                                else -self.world.x_semidim - self.agent_radius
                            )
                        ),
                        (
                            0.0
                            if not i % 2
                            else (
                                self.world.y_semidim + self.agent_radius
                                if i == 1
                                else -self.world.y_semidim - self.agent_radius
                            )
                        ),
                    ],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            wall.set_rot(
                torch.tensor(
                    [torch.pi / 2 if not i % 2 else 0.0],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def process_action(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            formation_movement = "horizental"
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
                self.formation_center.set_rot(
                    torch.tensor(
                        [
                            self.random_path[self.t][2],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=None,
                )
                self.formation_center_pos[0] = self.random_path[self.t][0]
                self.formation_center_pos[1] = self.random_path[self.t][1]
                self.formation_center_pos[2] = self.random_path[self.t][2]

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
                self.formation_center.set_rot(
                    torch.tensor(
                        torch.pi,
                        device=self.world.device,
                    ),
                    batch_index=None,
                )
                self.formation_center_pos[2] = torch.pi
                ###################  circling end ######################
            elif formation_movement == "horizental":
                ###move from left to right, test formation's ability to cross through tunnel
                t = self.t / 30
                if self.t < 100:
                    self.leader_robot.set_pos(
                        torch.tensor(
                            [
                                 -4,
                                0,
                            ],
                            device=self.world.device,
                        ),
                        batch_index=None,
                    )
                    self.formation_center.set_pos(
                        torch.tensor(
                            [
                                 -4,
                                0,
                            ],
                            device=self.world.device,
                        ),
                        batch_index=None,
                    )
                    self.formation_center_pos[0] = -4
                    self.formation_center_pos[1] = 0
                else:
                    self.leader_robot.set_pos(
                        torch.tensor(
                            [
                                (self.t-100)/30*0.5 - 4,
                                0,
                            ],
                            device=self.world.device,
                        ),
                        batch_index=None,
                    )
                    self.formation_center.set_pos(
                        torch.tensor(
                            [
                                (self.t-100)/30*0.5 - 4,
                                0,
                            ],
                            device=self.world.device,
                        ),
                        batch_index=None,
                    )
                    self.formation_center_pos[0] = ((self.t-100)/30*0.5 - 4)
                    self.formation_center_pos[1] = 0
                self.leader_robot.set_rot(
                    torch.tensor(
                        torch.pi,
                        device=self.world.device,
                    ),
                    batch_index=None,
                )
                self.formation_center.set_rot(
                    torch.tensor(
                        torch.pi,
                        device=self.world.device,
                    ),
                    batch_index=None,
                )
                self.formation_center_pos[2] = torch.pi


            
        
        angles = [-135/180.0*math.pi, 135/180.0*math.pi, -135/180.0*math.pi,  135/180.0*math.pi]
        dists = [-0.5, -0.5, -1, -1]
        for i, world_agent in enumerate(self.world.agents):
            if agent == world_agent:

                # formation_type = "line"
                # formation_type = "rectangle"
                
                # if self.t < 200:
                #     formation_type = "rectangle"
                # else:
                #     formation_type = "ren_shape"
                if self.current_formation_type == "ren_shape":
                    #大雁人字形
                    angles = [-135/180.0*math.pi, 135/180.0*math.pi, -135/180.0*math.pi,  135/180.0*math.pi]
                    dists = [-0.5, -0.5, -1, -1]
                    self.formation_goals[i][0] = self.formation_center_pos[0] + math.cos(self.formation_center_pos[2] + angles[i]) * dists[i]
                    self.formation_goals[i][1] = self.formation_center_pos[1] + math.sin(self.formation_center_pos[2] + angles[i]) * dists[i]
                elif self.current_formation_type == "vertical_line":
                    dists = [0.5, 1, -0.5, -1]
                    self.formation_goals[i][0] = self.formation_center_pos[0] 
                    self.formation_goals[i][1] = self.formation_center_pos[1] + dists[i-1]
                
                
                elif self.current_formation_type == "line":
                    #直线型
                    dists = [-0.5, -1, -1.5, -2]
                    self.formation_goals[i][0] = self.formation_center_pos[0] + dists[i-1]
                    self.formation_goals[i][1] = self.formation_center_pos[1]
                elif self.current_formation_type == "rectangle":
                    #矩形
                    displacement_x = [1, -1, 1, -1]
                    displacement_y = [1, 1, -1, -1]
                    self.formation_goals[i][0] = self.formation_center_pos[0] + displacement_x[i-1]
                    self.formation_goals[i][1] = self.formation_center_pos[1] + displacement_y[i-1]
                self.formation_goals[i][2] = self.formation_center_pos[2]
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
                self.formation_goals_landmark[i].set_rot(
                torch.tensor(
                    [
                        self.formation_goals[i][2],
                    ],
                    device=self.world.device,
                ),
                batch_index=None,
                )
                # print("formation goal {}, {}".format(i, self.formation_goals_landmark[i].state.pos))

    def reward(self, agent: Agent):
        current_agent_index = self.world.agents.index(agent)
        is_first = agent == self.world.agents[0]
        # if is_first:
        #     print("agent:{}".format(0))
        # else:
        #     print("agent else")
        # if self.shared_rew:'
        #     print("shared reward")
        # else:
        #     print("not shared reward")
        agent.agent_collision_rew[:] = 0
        if is_first:
            self.t += 1
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            self.angle_rew[:] = 0
            self.formation_maintain_rew[:] = 0
            for i, a in enumerate(self.world.agents):
                a.agent_collision_rew[:] = 0
                a.pos_rew[:] = 0
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
        # if is_first:
            for i, a in enumerate(self.world.agents):
                for obs in self.obstacles:
                    # if self.world.collides(a, obs):
                    distance = self.world.get_distance(a, obs)
                    a.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty    
            
            # print("---------------------before----------------------")
            self.formation_maintain_rew = self.agent_reward_graph_formation_maintained()
        agent.pos_rew = self.single_agent_reward_graph_formation_maintained(agent)
                # self.angle_rew += self.agent_angle_reward(a) 
            # print("---------------------after-------------------------")

            # self.all_goal_reached = torch.all(
            #     torch.stack([a.on_goal for a in self.world.agents], dim=-1),
            #     dim=-1,
            # )

            # self.final_rew[self.all_goal_reached] = self.final_reward

            

            

        pos_reward =  agent.pos_rew
        angle_reward = self.angle_rew if self.shared_rew else agent.angle_rew
        # return 5*pos_reward + self.final_rew + agent.agent_collision_rew + angle_reward
        # return self.formation_maintain_rew + agent.agent_collision_rew
        # print("agent {} pos reward:{} collsition reward:{}".format(current_agent_index, pos_reward, agent.agent_collision_rew))
        return pos_reward + agent.agent_collision_rew + self.formation_maintain_rew
    


    def normalize_positions(self, positions):
        # Calculate relative positions to the center
        relative_positions = positions - torch.tensor(self.formation_center_pos[:2], device=self.device)
        
        # Calculate distances and angles
        dists = torch.norm(relative_positions, dim=1)
        angles = torch.atan2(relative_positions[:, 1], relative_positions[:, 0])
        
        # Normalize distances
        max_dist = dists.max()
        normalized_dists = dists / max_dist
        
        # Convert back to Cartesian coordinates
        normalized_positions = torch.stack([
            normalized_dists * torch.cos(angles),
            normalized_dists * torch.sin(angles)
        ], dim=1)
        
        return normalized_positions


    def agent_reward_graph_formation_maintained(self):
        target_positions_dict = {}
        for a_index, agent in enumerate(self.world.agents):
            target_positions_dict[a_index] = agent.state.pos

        batch_size = next(iter(target_positions_dict.values())).shape[0]
        graph_connect_rew = torch.zeros(batch_size, device=self.device)
        num_agents = len(target_positions_dict)
        
        formation_goals_positions = []
        for i in range(num_agents):
            angles = [-135 / 180.0 * math.pi, 135 / 180.0 * math.pi, -135 / 180.0 * math.pi, 135 / 180.0 * math.pi]
            dists = [-0.5, -0.5, -1, -1]
            goal_x = self.formation_center_pos[0] + math.cos(self.formation_center_pos[2] + angles[i]) * dists[i]
            goal_y = self.formation_center_pos[1] + math.sin(self.formation_center_pos[2] + angles[i]) * dists[i]
            formation_goals_positions.append(torch.tensor([goal_x, goal_y], device=self.device))
        formation_goals_positions = torch.stack(formation_goals_positions)

        for batch_idx in range(batch_size):
            target_positions = torch.stack([target_positions_dict[i][batch_idx] for i in range(num_agents)])
            
            # Normalize positions to unit formation
            normalized_target_positions = self.normalize_positions(target_positions)
            normalized_formation_goals_positions = self.normalize_positions(formation_goals_positions)
            
            # Calculate the cost matrix based on distances
            cost_matrix = torch.cdist(normalized_target_positions, normalized_formation_goals_positions)

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

            # Calculate the total distance for the optimal assignment
            total_distance = cost_matrix[row_ind, col_ind].sum()

            # Normalize the distance to get the reward (assuming max reward is 1.0)
            max_distance = (num_agents * torch.norm(normalized_formation_goals_positions - normalized_formation_goals_positions.mean(dim=0), dim=1).max()).item()
            reward = 1.0 - (total_distance.item() / max_distance)
            reward = max(reward, 0.0)  # Ensure the reward is non-negative

            graph_connect_rew[batch_idx] = reward

        return graph_connect_rew

    # def agent_reward_graph_formation_maintained(self):
    #     target_positions_dict = {}
    #     target_rotations_dict = {}
    #     for a_index, agent in enumerate(self.world.agents):
    #         target_positions_dict[a_index] = agent.state.pos
    #         target_rotations_dict[a_index] = agent.state.rot
            
    #     batch_size = next(iter(target_positions_dict.values())).shape[0]
    #     graph_connect_rew = torch.zeros(batch_size, device=self.device) 
    #     num_agents = len(target_positions_dict)

    #     formation_goals_positions = []
    #     for i in range(num_agents):
    #         formation_goals_positions.append(self.formation_goals_landmark[i].state.pos[0])
    #     formation_goals_positions = torch.stack(formation_goals_positions)

    #     def geometric_center(positions):
    #         return torch.mean(positions, dim=0)

    #     for batch_idx in range(batch_size):
    #         target_positions = torch.stack([target_positions_dict[i][batch_idx] for i in range(num_agents)])
    #         formation_center = geometric_center(formation_goals_positions)
    #         target_center = geometric_center(target_positions)
    #         # print("formation_goals_positions shape:{}".format(formation_goals_positions.shape))
    #         # print("target_positions shape:{}".format(target_positions.shape))
    #         # print("target_center shape:{}".format(target_center.shape))
    #         # Center the positions
    #         #begin for matching error reward

    #         centered_formation_positions = formation_goals_positions - formation_center
    #         centered_target_positions = target_positions - target_center
    #         # print("centered_target-positions shape:{}".format(centered_target_positions.shape))
    #         # Calculate the optimal assignment using the Hungarian algorithm (linear_sum_assignment)
    #         # print("centered_formation_positions shape:{}".format(centered_formation_positions.shape))
    #         cost_matrix = torch.cdist(centered_target_positions, centered_formation_positions)
    #         # print("cost_matrix shape:{}".format(cost_matrix.shape))
    #         # Ensure cost_matrix is 2D (num_agents x num_agents)
    #         if cost_matrix.dim() != 2:
    #             raise ValueError("Cost matrix must be 2D")
            
    #         row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            
    #         optimal_assignment_distances = cost_matrix[row_ind, col_ind]
            
    #         # Calculate the optimal scale factor
    #         optimal_scale_factor = (torch.norm(centered_target_positions[row_ind], dim=1) / torch.norm(centered_formation_positions[col_ind], dim=1)).mean()

    #         # Scale the formation positions
    #         scaled_formation_positions = centered_formation_positions * optimal_scale_factor

    #         # Calculate the rotation matrix using Singular Value Decomposition (SVD)
    #         H = torch.matmul(centered_target_positions[row_ind].T, scaled_formation_positions[col_ind])
    #         U, S, Vt = torch.linalg.svd(H)
    #         rotation_matrix = torch.matmul(U, Vt)

    #         # Rotate the formation positions
    #         rotated_formation_positions = torch.matmul(scaled_formation_positions, rotation_matrix.T)

    #         # Calculate the matching error
    #         matching_error = torch.norm(centered_target_positions[row_ind] - rotated_formation_positions[col_ind], dim=1).mean()

    #         #end for matching_error reward


            
    #         center_matching_error = torch.norm(target_center - formation_center).item()
    #         # print("matching_error:{}".format(matching_error))
    #         # print("center error:{}".format(center_matching_error))
    #         # Calculate the reward
    #         max_reward = 1.0
    #         reward = max_reward - (matching_error.item() + center_matching_error)
    #         # reward = max_reward - center_matching_error
    #         graph_connect_rew[batch_idx] = reward
    #         # Calculate the reward
    #     # print("formation_maintain reward:{}".format(graph_connect_rew))
    #     return graph_connect_rew

    def single_agent_reward_graph_formation_maintained(self, agent):
        current_agent_index = self.world.agents.index(agent)
        target_positions_dict = {}
        target_rotations_dict = {}
        for a_index, agent in enumerate(self.world.agents):
            target_positions_dict[a_index] = agent.state.pos
            target_rotations_dict[a_index] = agent.state.rot

        batch_size = next(iter(target_positions_dict.values())).shape[0]
        num_agents = len(target_positions_dict)

        formation_goals_positions = []
        for i in range(num_agents):
            formation_goals_positions.append(self.formation_goals_landmark[i].state.pos[0])
        formation_goals_positions = torch.stack(formation_goals_positions)

        # print("formation_goals_positions shape{},:{}".format(formation_goals_positions.shape, formation_goals_positions))
        # print("target_positions:{}".format(target_positions_dict))

        for batch_idx in range(batch_size):
            # Extract positions for the current batch
            target_positions = torch.stack([target_positions_dict[i][batch_idx] for i in range(num_agents)])

            # Calculate the cost matrix based on distances
            cost_matrix = torch.cdist(target_positions, formation_goals_positions)

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            
            # Find the index in the assignment for the current agent
            agent_assignment_idx = row_ind.tolist().index(current_agent_index)
            assigned_goal_idx = col_ind[agent_assignment_idx]
            
            # Calculate the distance from the current agent to its assigned goal
            agent_position = target_positions[current_agent_index]
            assigned_goal_position = formation_goals_positions[assigned_goal_idx]
            distance_to_goal = torch.norm(agent_position - assigned_goal_position)
            # Normalize the distance to get the reward (assuming max reward is 1.0)
            # max_distance = torch.norm(formation_goals_positions - formation_goals_positions.mean(dim=0), dim=1).max().item()
            # reward = 1.0 - (distance_to_goal.item() / max_distance)
            # reward = max(reward, 0.0)  # Ensure the reward is non-negative

            reward = agent.target_distance[batch_idx] - distance_to_goal
            agent.target_distance[batch_idx] = distance_to_goal

            if distance_to_goal.item() < 0.02:
                reward += 1.0
            # print("agent {} reward {}".format(current_agent_index, reward))
            agent.pos_rew[batch_idx] = reward

        return agent.pos_rew


    #when robot and goal is one-on-one 
    def agent_reward(self, agent: Agent):
        current_agent_index = self.world.agents.index(agent)

        agent_positions = torch.stack([a.state.pos for a in self.world.agents])  # [num_agents, 1200, 2]

        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        num_envs = agent_positions.shape[1]  # 1200 environments

        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        for e in range(num_envs):
            # print("agent.goal.state.pos:{}".format(agent.goal.state.pos))
            self.formation_goals_modified[current_agent_index].state.pos[e] = agent.goal.state.pos[e]
        return agent.pos_rew

    def agent_angle_reward(self, agent: Agent):
        current_agent_index = self.world.agents.index(agent)

        agent_angles = torch.stack([a.state.rot for a in self.world.agents])


        agent.angle_to_goal = torch.linalg.vector_norm(
            agent.state.rot - agent.goal.state.rot,
            dim=-1,
        )
        num_envs = agent_angles.shape[1]  # 1200 environments


        angle_shaping = agent.angle_to_goal * self.pos_shaping_factor
        agent.angle_rew = agent.angle_shaping - angle_shaping
        agent.angle_shaping = angle_shaping
        for e in range(num_envs):
            # print("agent.goal.state.pos:{}".format(agent.goal.state.pos))
            self.formation_goals_modified[current_agent_index].state.rot[e] = agent.goal.state.rot[e]
        return agent.angle_rew


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
        goal_rot_poses = []
        current_agent_index = self.world.agents.index(agent)
        target_positions_dict = {}
        for a_index, agent in enumerate(self.world.agents):
            target_positions_dict[a_index] = agent.state.pos

        batch_size = next(iter(target_positions_dict.values())).shape[0]
        num_agents = len(target_positions_dict)

        formation_goals_positions = []
        for i in range(num_agents):
            formation_goals_positions.append(self.formation_goals_landmark[i].state.pos[0])
        formation_goals_positions = torch.stack(formation_goals_positions)

        # print("formation_goals_positions shape{},:{}".format(formation_goals_positions.shape, formation_goals_positions))
        # print("target_positions:{}".format(target_positions_dict))
        relative_goal_position_list = []
        for batch_idx in range(batch_size):
            # Extract positions for the current batch
            target_positions = torch.stack([target_positions_dict[i][batch_idx] for i in range(num_agents)])

            # Calculate the cost matrix based on distances
            cost_matrix = torch.cdist(target_positions, formation_goals_positions)

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            
            # Find the index in the assignment for the current agent
            agent_assignment_idx = row_ind.tolist().index(current_agent_index)
            assigned_goal_idx = col_ind[agent_assignment_idx]
            
            # Calculate the distance from the current agent to its assigned goal
            agent_position = target_positions[current_agent_index]
            assigned_goal_position = formation_goals_positions[assigned_goal_idx]
            relative_goal_position = assigned_goal_position - agent_position
            relative_goal_position_list.append(relative_goal_position)
        relative_goal_position_tensor = torch.stack(relative_goal_position_list)
        relative_goal_position_tensor_list = []
        relative_goal_position_tensor_list.append(relative_goal_position_tensor)


        relative_to_other_robots_pose = []
        relative_to_other_formation_pose = []

        formation_goals_positions = []
        for i in range(len(self.world.agents)):
            # formation_goals_positions.append(self.formation_goals_landmark[i].state.pos[0])
            relative_to_other_formation_pose.append(self.formation_goals_landmark[i].state.pos - agent.state.pos)
        # formation_goals_positions = torch.stack(formation_goals_positions)
        # #follower robots
        

        relative_to_leader_pose = [self.leader_robot.state.pos - agent.state.pos]
        for a in self.world.agents:
            if a != agent:
                relative_to_other_robots_pose.append(a.state.pos - agent.state.pos)

        
        # print("observation:{}".format(agent.state.pos))
        # observations = [agent.state.pos, agent.state.vel]

    # Add positions and velocities of all agents (including the current agent)
        # for a in self.world.agents:
        #     if a != agent:
        #         observations.append(a.state.pos)
        # formation_type = "ren_shape"
                   
        # if formation_type == "ren_shape":
        #     #大雁人字形
        #     self.formation_goals[i][0] = self.formation_center_pos[0] + math.cos(self.formation_center_pos[2] + angles[i-1]) * dists[i-1]
        #     self.formation_goals[i][1] = self.formation_center_pos[1] + math.sin(self.formation_center_pos[2] + angles[i-1]) * dists[i-1]
        # elif formation_type == "vertical_line":
        #     dists = [0.5, 1, -0.5, -1]
        #     self.formation_goals[i][0] = self.formation_center_pos[0] 
        #     self.formation_goals[i][1] = self.formation_center_pos[1] + dists[i-1]
        
        
        # elif formation_type == "line":
        #     #直线型
        #     dists = [0.5, 1, -0.5, -1]
        #     self.formation_goals[i][0] = self.formation_center_pos[0] + dists[i-1]
        #     self.formation_goals[i][1] = self.formation_center_pos[1]
        # elif formation_type == "rectangle":
        #     #矩形
        #     displacement_x = [1, -1, 1, -1]
        #     displacement_y = [1, 1, -1, -1]
        #     self.formation_goals[i][0] = self.formation_center_pos[0] + displacement_x[i-1]
        #     self.formation_goals[i][1] = self.formation_center_pos[1] + displacement_y[i-1]
        # self.formation_goals[i][2] = self.formation_center_pos[2]



        if self.current_formation_type == "ren_shape":
            for i in range(len(self.world.agents)):
                if i == 0:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -0.3536
                    goal_pose_tensor[:, 1] = 0.3536
                    goal_poses.append(goal_pose_tensor)
                elif i == 1:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -0.3536
                    goal_pose_tensor[:, 1] = -0.3536
                    goal_poses.append(goal_pose_tensor)
                elif i == 2:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -0.7071
                    goal_pose_tensor[:, 1] = 0.7071
                    goal_poses.append(goal_pose_tensor)
                elif i == 3:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -0.7071
                    goal_pose_tensor[:, 1] = -0.7071
                    goal_poses.append(goal_pose_tensor)
        elif self.current_formation_type == "vertical_line":
            for i in range(len(self.world.agents)):
                if i == 0:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = 0
                    goal_pose_tensor[:, 1] = 0.5
                    goal_poses.append(goal_pose_tensor)
                elif i == 1:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = 0
                    goal_pose_tensor[:, 1] = -0.5
                    goal_poses.append(goal_pose_tensor)
                elif i == 2:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = 0
                    goal_pose_tensor[:, 1] = 1
                    goal_poses.append(goal_pose_tensor)
                elif i == 3:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = 0
                    goal_pose_tensor[:, 1] = -1
                    goal_poses.append(goal_pose_tensor)   
        elif self.current_formation_type == "line":
            for i in range(len(self.world.agents)):
                if i == 0:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -0.5
                    goal_pose_tensor[:, 1] = 0
                    goal_poses.append(goal_pose_tensor)
                elif i == 1:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -1
                    goal_pose_tensor[:, 1] = 0
                    goal_poses.append(goal_pose_tensor)
                elif i == 2:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -1.5
                    goal_pose_tensor[:, 1] = 0
                    goal_poses.append(goal_pose_tensor)
                elif i == 3:
                    goal_pose_tensor = torch.zeros((self.world.batch_dim, 2)).to(self.device)
                    goal_pose_tensor[:, 0] = -2
                    goal_pose_tensor[:, 1] = 0
                    goal_poses.append(goal_pose_tensor)
        elif self.current_formation_type == "rectangle":
            pass
        # goal_rot_poses.append(agent.state.rot - agent.goal.state.rot)

        # observation_tensor = torch.cat(relative_goal_position_tensor_list + relative_to_other_robots_pose + (
        #         [(agent.sensors[0]._max_range - agent.sensors[0].measure()) / agent.sensors[0]._max_range]
        #         if self.collisions
        #         else []
        #     ),
        #     dim=-1
        # )
        observation_tensor = torch.cat(
            relative_to_leader_pose + relative_to_other_robots_pose + relative_to_other_formation_pose +relative_goal_position_tensor_list+ (
                [(agent.sensors[0]._max_range - agent.sensors[0].measure()) / agent.sensors[0]._max_range]
                if self.collisions
                else []
            ),
            dim=-1
        )
        # print("agent {} obs:{} shape:{}".format(current_agent_index, observation_tensor, observation_tensor.shape))

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
        done_status = self.keep_track_time >= 300
        if torch.any(done_status):
            # Count the number of True entries
            num_true = torch.sum(done_status).item()  # .item() to get the value as a Python scalar
            print("Number of True entries:", num_true)
        else:
            pass
            # print("No True entries found.")
        return done_status

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        # print("info agent pos:{}".format(agent.state.pos))
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
            "formation_goal": agent.goal.state.pos,
            "formation_main_rew":self.formation_maintain_rew,
            "agent_pos": agent.state.pos,
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
        D = 0.6  # Example distance threshold
        FOV_min = -0.35 * np.pi
        FOV_max = 0.35 * np.pi
        alpha = 1.0  # Weight for distance difference
        beta = 1.0   # Weight for angle difference
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if i != j:
                    pos_i = agent1.state.pos[env_index]
                    pos_j = agent2.state.pos[env_index]
                    rot_i = agent1.state.rot[env_index]
                    
                    rel_pos = pos_j - pos_i
                    d_ij = torch.norm(rel_pos).item()
                    
                    if d_ij <= D:
                        theta_ij = torch.atan2(rel_pos[1], rel_pos[0]).item() - rot_i.item()
                        theta_ij = np.arctan2(np.sin(theta_ij), np.cos(theta_ij))
                        
                        if FOV_min <= theta_ij <= FOV_max:
                            color = Color.RED.value
                            line = rendering.Line(
                                (agent1.state.pos[env_index]),
                                (agent2.state.pos[env_index]),
                                width=2,
                            )
                            xform = rendering.Transform()
                            line.add_attr(xform)
                            line.set_color(*color)
                            geoms.append(line)

        return geoms
        # for i, agent1 in enumerate(self.world.agents):
        #     agent_direction_pos = agent1.state.pos[env_index].clone()
        #     # print("agent_direction_pos[env_index][0] shape:{}".format(agent_direction_pos.shape))
        #     # print("state.rot shape:{}".format(torch.cos(agent1.state.rot).shape))
        #     agent_direction_pos[0] = agent1.state.pos[env_index][0] + 0.3*torch.cos(agent1.state.rot[env_index])
        #     agent_direction_pos[1] = agent1.state.pos[env_index][1] + 0.3*torch.sin(agent1.state.rot[env_index])

        #     color = Color.RED.value
        #     line = rendering.Line(
        #         (agent1.state.pos[env_index]),
        #         (agent_direction_pos[env_index]),
        #         width=2,
        #     )
        #     xform = rendering.Transform()
        #     line.add_attr(xform)
        #     line.set_color(*color)
        #     geoms.append(line)
        
        # for i, goal1 in enumerate(self.formation_goals_landmark.items()):
        #     agent_direction_pos = goal1[1].state.pos[env_index].clone()
        #     agent_direction_pos[0] = goal1[1].state.pos[env_index][0] + 0.3*torch.cos(goal1[1].state.rot[env_index])
        #     agent_direction_pos[1] = goal1[1].state.pos[env_index][1] + 0.3*torch.sin(goal1[1].state.rot[env_index])

        #     color = Color.GREEN.value
        #     line = rendering.Line(
        #         (goal1[1].state.pos[env_index]),
        #         (agent_direction_pos[env_index]),
        #         width=2,
        #     )
        #     xform = rendering.Transform()
        #     line.add_attr(xform)
        #     line.set_color(*color)
        #     geoms.append(line)
        return geoms

# class HeuristicPolicy(BaseHeuristicPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.max_speed = 1.0
#         self.max_rotation = 0.1

#     def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
#         print("obs shape:{}".format(observation.shape))
#         self.n_env = observation.shape[0]
#         self.device = observation.device
#         agent_pos = observation[:, :2]
#         # agent_vel = observation[:, 2:4]
#         package_pos = observation[:, 6:8] + agent_pos
#         goal_pos = -observation[:, 4:6] + package_pos

#         # def expert_policy(observations, formation_goals_positions, device, max_speed=1.0, max_rotation=0.1):
#         batch_size, num_agents, obs_size = observations.shape
#         actions = torch.zeros((batch_size, num_agents, 3), device=device)

#         for b in range(batch_size):
#             batch_observations = observations[b]
#             batch_goals = formation_goals_positions[b]

#             # Calculate the cost matrix based on the distance between observations and formation goals
#             cost_matrix = torch.cdist(batch_observations[:, :2], batch_goals)
            
#             # Solve the assignment problem to minimize the travel distance
#             row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

#             # Calculate the actions to move towards the assigned goals
#             for i in range(num_agents):
#                 print("col_ind:{}".format(col_ind))
#                 goal = batch_goals[col_ind[i]]
#                 print("goal:{}".format(goal))
#                 current_pos = batch_observations[i, :2]
#                 print("current pos:{}".format(current_pos))
#                 current_rot = batch_observations[i, 2]
#                 action = goal - current_pos

#                 # Normalize the action to ensure it's within the max speed limit
#                 norm = torch.norm(action)
#                 if norm > max_speed:
#                     action = action / norm * max_speed

#                 # Calculate the required rotation to face the target
#                 target_angle = torch.atan2(action[1], action[0])
#                 rotation_force = target_angle - current_rot
#                 rotation_force = (rotation_force + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize to [-pi, pi]

#                 # Clip the rotation force to the maximum rotation
#                 if rotation_force > max_rotation:
#                     rotation_force = max_rotation
#                 elif rotation_force < -max_rotation:
#                     rotation_force = -max_rotation

#                 actions[b, i, :2] = action
#                 actions[b, i, 2] = rotation_force

#         return actions


#         # control = self.get_action(goal_pos, curr_pos=agent_pos, curr_vel=agent_vel)
#         control *= self.speed * u_range
#         return torch.clamp(control, -u_range, u_range)


if __name__ == "__main__":




    render_interactively(
        __file__,
        control_two_agents=True,
    )
