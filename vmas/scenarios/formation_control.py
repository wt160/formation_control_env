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
import torch.optim as optim
from torch import Tensor
from vmas.make_env import make_env
from scipy.optimize import linear_sum_assignment
from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box, Line
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from scipy.spatial import KDTree
if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class ObstacleManager:
    def __init__(self, obstacles):
        # Assumes obstacles is a list of torch.Tensor objects of shape [2]
        self.obstacles = obstacles
        
        # Convert torch.Tensors to numpy arrays for KDTree
        self.obstacle_positions = np.array([obs.cpu().numpy() for obs in self.obstacles])
        
        # Build the KDTree based on the obstacles' positions
        self.tree = KDTree(self.obstacle_positions)
    
    def get_near_obstacles(self, query_pos, radius):
        # Query the KDTree for all obstacles within the given radius
        indices = self.tree.query_ball_point(query_pos, radius)
        
        # Retrieve the corresponding obstacles as torch.Tensors
        return [self.obstacles[i] for i in indices]

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
        self.collisions = kwargs.get("collisions", False)
        self.viewer_size = (1100, 1100)
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

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -10)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 5
        self.min_collision_distance = 0.005
        generator = RobotPathGenerator()
        self.current_formation_type = "ren_shape"
        # Parameters
        x_max, y_max = 3.0, 3.0  # bounds
        num_steps = 1000  # number of steps in the path
        self.need_to_reconfigure = False
        self.reconfigure_success_t = 0
        # Generate the path
        self.random_path = generator.generate_random_path(x_max, y_max, num_steps)
        self.obstacle_manager_list = []
        self.formation_normal_width = 0.0
        self.inter_robot_min_dist = 0.2
        self.inter_robot_obs_min_dist = 0.2
        self.current_step_reset = False
        self.last_optimized_formation_poses = []
        self.last_opening_width = None
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

        # Make 
        world = World(batch_dim, device, substeps=2)
        world._x_semidim = self.world_semidim
        world._y_semidim = self.world_semidim
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
        self.t_since_success_reconfigure = 0
        self.last_leader_robot = torch.tensor([
                        0.0,
                        0.0,
                        0.0,
                    ],
                    device=device,
                )

        self.formation_center_pos = torch.zeros(
                (batch_dim, 3),
                device=device
            )
        self.formation_goals = {}
        self.success_reconfigure_goals = {}
        self.formation_goals_landmark = {}
        self.formation_goals_modified = {}
        for i in range(self.n_agents):
            self.formation_goals[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            self.success_reconfigure_goals[i] = torch.zeros(
                (batch_dim, 3),
                device=device
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
            # self.formation_goals_landmark[i].renderable = False
            world.add_landmark(self.formation_goals_landmark[i])
            self.formation_goals_modified[i] = Landmark(
                name=f"modified_goal_{i}",
                collide=False,
                shape=Sphere(radius = 0.02),
                movable=True,
                rotatable=True,
                color=Color.BLUE,
            )
            self.formation_goals_modified[i].renderable = True
            world.add_landmark(self.formation_goals_modified[i])
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

        self.obstacle_pattern = 0
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
                    if False
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
        self.FOV_min = -0.45 * torch.pi
        self.FOV_max = 0.45 * torch.pi
        self.observe_D = 0.6
        return world

    def create_obstacles(self, obstacle_pattern, world: World):
        self.obstacles = []

        if obstacle_pattern == 0:
            #random located obstalces
            self.n_boxes = 60
            self.box_width = 0.1
            for i in range(self.n_boxes):
                obs = Landmark(
                    name=f"obs_{i}",
                    collide=True,
                    movable=False,
                    shape=Sphere(radius=self.box_width),
                    color=Color.RED,
                    collision_filter=lambda e: not isinstance(e.shape, Sphere),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
        elif obstacle_pattern == 1:
            #random located obstalces
            self.n_boxes = 200
            self.box_width = 0.1
            for i in range(self.n_boxes):
                obs = Landmark(
                    name=f"obs_{i}",
                    collide=True,
                    movable=False,
                    shape=Sphere(radius=self.box_width),
                    color=Color.RED,
                    collision_filter=lambda e: not isinstance(e.shape, Sphere),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
        
        elif obstacle_pattern == 2:
            #random located obstalces
            self.n_boxes = 40
            self.box_width = 0.1
            for i in range(self.n_boxes):
                obs = Landmark(
                    name=f"obs_{i}",
                    collide=True,
                    movable=False,
                    shape=Sphere(radius=self.box_width),
                    color=Color.RED,
                    collision_filter=lambda e: not isinstance(e.shape, Sphere),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
        elif obstacle_pattern == 4:
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
        elif obstacle_pattern == 3:
            #two large boxes, relatively large corridor
            self.n_boxes = 4
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
        line_segments = []  # Store line segments to maintain continuity



        def create_line_segment():
            # Fixed spacing between spheres
            fixed_spacing = 0.01+ 2*self.agent_radius
            # Random number of spheres per segment
            num_spheres = np.random.randint(3, 10)  
            # Calculate the total length of the line segment based on fixed spacing
            segment_length = fixed_spacing * (num_spheres - 1)  
            # Starting position of the line segment
            start_pos = torch.tensor(
                [
                    np.random.uniform(-self.world_semidim, self.world_semidim),
                    np.random.uniform(-self.world_semidim, self.world_semidim),
                ],
                dtype=torch.float32,
                device=self.world.device,
            )
            # Direction vector for the line segment
            direction = torch.tensor(
                [
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                ],
                dtype=torch.float32,
                device=self.world.device,
            )
            direction = direction / torch.norm(direction)  # Normalize to get direction

            # Generate positions for spheres along the line segment with fixed spacing
            positions = []
            for idx in range(num_spheres):
                offset = fixed_spacing * idx  # Fixed spacing between spheres
                sphere_pos = start_pos + offset * direction
                positions.append(sphere_pos)

            return positions
        
        def create_certain_line_segment(num_spheres, start_pos, direction):
            # Fixed spacing between spheres
            fixed_spacing = 0.01+ 2*self.agent_radius
            # Random number of spheres per segment
            # num_spheres = np.random.randint(3, 10)  
            # Calculate the total length of the line segment based on fixed spacing
            # segment_length = fixed_spacing * (num_spheres - 1)  
            # Starting position of the line segment
            # start_pos = torch.tensor(
            #     [
            #         np.random.uniform(-self.world_semidim, self.world_semidim),
            #         np.random.uniform(-self.world_semidim, self.world_semidim),
            #     ],
            #     dtype=torch.float32,
            #     device=self.world.device,
            # )
            # # Direction vector for the line segment
            # direction = torch.tensor(
            #     [
            #         np.random.uniform(-1, 1),
            #         np.random.uniform(-1, 1),
            #     ],
            #     dtype=torch.float32,
            #     device=self.world.device,
            # )
            # direction = direction / torch.norm(direction)  # Normalize to get direction

            # Generate positions for spheres along the line segment with fixed spacing
            positions = []
            for idx in range(num_spheres):
                offset = fixed_spacing * idx  # Fixed spacing between spheres
                sphere_pos = start_pos + offset * direction
                positions.append(sphere_pos)

            return positions
        
        def get_pos(i):
            if obstacle_pattern == 0:
                # Randomly located obstacles
                pos = torch.tensor(
                    [[
                        np.random.uniform(-self.world_semidim, self.world_semidim),
                        np.random.uniform(-self.world_semidim, self.world_semidim),
                    ] for _ in range(i.shape[0])],
                    dtype=torch.float32,
                    device=self.world.device,
                )
                return pos
            elif obstacle_pattern == 1:
                # Generate line segments if not already done
                if len(line_segments) == 0:
                    while len(line_segments) < self.n_boxes:
                        line_segments.extend(create_line_segment())
                
                # Assign positions from the pre-generated line segments
                if i.item() < len(line_segments):
                    return line_segments[i.item()]
                else:
                    # Handle cases where i exceeds the number of pre-generated segments
                    return torch.tensor(
                        [
                            np.random.uniform(-self.world_semidim, self.world_semidim),
                            np.random.uniform(-self.world_semidim, self.world_semidim),
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
            elif obstacle_pattern == 2:

                
                if len(line_segments) == 0:
                    start_pos_1 = torch.tensor(
                        [
                            -1,
                            0.6,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
                    direction_1 = torch.tensor(
                        [
                            1,
                            -0.1,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
                    direction_1 = direction_1 / torch.norm(direction_1)  # Normalize to get direction

                    start_pos_2 = torch.tensor(
                        [
                            -1,
                            -0.5,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
                    direction_2 = torch.tensor(
                        [
                            1,
                            0.04,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
                    direction_2 = direction_2 / torch.norm(direction_2)  # Normalize to get direction

                    line_segments.extend(create_certain_line_segment(20, start_pos_1, direction_1))
                    line_segments.extend(create_certain_line_segment(20, start_pos_2, direction_2))

                # Assign positions from the pre-generated line segments
                if i.item() < len(line_segments):
                    return line_segments[i.item()]
                else:
                    # Handle cases where i exceeds the number of pre-generated segments
                    return torch.tensor(
                        [
                            np.random.uniform(-self.world_semidim, self.world_semidim),
                            np.random.uniform(-self.world_semidim, self.world_semidim),
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
            elif obstacle_pattern == 3:
                # Specific positions based on i value
                is_zero = (i == 0)
                is_one = (i == 1)
                is_two = (i == 2)
                is_third = (i == 3)
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
                        1.4,
                        2.21,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
                elif is_third.any():
                    pos = torch.tensor(
                    [
                        1.4,
                        -2.21,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos

        # Initialize positions and assign to obstacles
        i = torch.zeros(
            (self.world.batch_dim,) if env_index is None else (1,),
            dtype=torch.int,
            device=self.world.device,
        )

        for obs in self.obstacles:
            obs.set_pos(get_pos(i), batch_index=env_index)
            i += 1

        # Create obstacle managers for each batch
        for d in range(self.world.batch_dim):
            single_batch_obstacles = [obs.state.pos[d,:].squeeze() for obs in self.obstacles]
            manager = ObstacleManager(single_batch_obstacles)
            self.obstacle_manager_list.append(manager)
    # def spawn_obstacles(self, obstacle_pattern, env_index):
    #     passage_indexes = []
    #     j = self.n_boxes // 2


    #     def get_pos(i):
    #         if obstacle_pattern == 0:
    #             #random located obstalces

    #             pos = torch.tensor(
    #                 [[
    #                     np.random.uniform(-self.world_semidim, self.world_semidim),
    #                     np.random.uniform(-self.world_semidim, self.world_semidim),
    #                 ] for _ in range(i.shape[0])],
    #                 dtype=torch.float32,
    #                 device=self.world.device,
    #             )
    #             return pos
    #         elif obstacle_pattern == 1:
    #             is_zero = (i == 0)
    #             is_one = (i == 1)
    #             is_two = (i == 2)
    #             if is_zero.any():
    #                 pos = torch.tensor(
    #                     [
    #                         1,
    #                         2.75,
    #                     ],
    #                     dtype=torch.float32,
    #                     device=self.world.device,
    #                 ).repeat(i.shape[0], 1)
    #                 return pos
    #             elif is_one.any():
    #                 pos = torch.tensor(
    #                 [
    #                     1,
    #                     -2.75,
    #                 ],
    #                 dtype=torch.float32,
    #                 device=self.world.device,
    #                 ).repeat(i.shape[0], 1)
    #                 return pos
    #             elif is_two.any():
    #                 pos = torch.tensor(
    #                 [
    #                     3,
    #                     3,
    #                 ],
    #                 dtype=torch.float32,
    #                 device=self.world.device,
    #                 ).repeat(i.shape[0], 1)
    #                 return pos
    #         elif obstacle_pattern == 2:
    #             is_zero = (i == 0)
    #             is_one = (i == 1)
    #             is_two = (i == 2)
    #             is_third = (i == 3)
    #             if is_zero.any():
    #                 pos = torch.tensor(
    #                     [
    #                         0,
    #                         2.5,
    #                     ],
    #                     dtype=torch.float32,
    #                     device=self.world.device,
    #                 ).repeat(i.shape[0], 1)
    #                 return pos
    #             elif is_one.any():
    #                 pos = torch.tensor(
    #                 [
    #                     0,
    #                     -2.5,
    #                 ],
    #                 dtype=torch.float32,
    #                 device=self.world.device,
    #                 ).repeat(i.shape[0], 1)
    #                 return pos
    #             elif is_two.any():
    #                 pos = torch.tensor(
    #                 [
    #                     1.4,
    #                     # 0,
    #                     2.21,
    #                 ],
    #                 dtype=torch.float32,
    #                 device=self.world.device,
    #                 ).repeat(i.shape[0], 1)
    #                 return pos
    #             elif is_third.any():
    #                 pos = torch.tensor(
    #                 [
    #                     1.4,
    #                     # 0,
    #                     -2.21,
    #                 ],
    #                 dtype=torch.float32,
    #                 device=self.world.device,
    #                 ).repeat(i.shape[0], 1)
    #                 return pos
    #     i = torch.zeros(
    #         (self.world.batch_dim,) if env_index is None else (1,),
    #         dtype=torch.int,
    #         device=self.world.device,
    #     )
        
    #     for obs in self.obstacles:
    #         obs.set_pos(get_pos(i), batch_index=env_index)
            
    #         i += 1
        
    #     for d in range(self.world.batch_dim):
    #         single_batch_obstacles = []
    #         for obs in self.obstacles:
    #             single_batch_obstacles.append(obs.state.pos[d,:].squeeze())
    #             # print("single_batch_obstacles:{}".format(single_batch_obstacles))
    #             # print("obs dim:{}".format(obs.state.pos.shape))
    #         manager = ObstacleManager(single_batch_obstacles)
    #         self.obstacle_manager_list.append(manager)


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
        self.leader_robot.set_pos(
                    torch.tensor(
                        [
                            -4.0, 
                            -0.0,
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
        self.leader_robot.set_rot( 
                torch.tensor(
                    [0.0],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        self.formation_center.set_pos(
                    torch.tensor(
                        [
                            -4.0, 
                            -0.0,
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
        self.formation_center.set_rot( 
                torch.tensor(
                    [0.0],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
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

            agent.goal.set_pos(agent.state.pos[env_index,:].squeeze(), batch_index=env_index)

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
        self.velocity_limit = 0.01
        # input("enter process_action")
        is_first = agent == self.world.agents[0]
        if is_first:
            formation_movement = "random"
            if formation_movement == "random":
                ###################  random walk ######################
                paired_obstacles_list, closest_pair_list = self.get_leader_forward_obstacles_2()

                print("closest_pair_list:{}".format(closest_pair_list))
                # obstacles = self.get_leader_forward_obstacles()
                # opening_widths = self.calculate_forward_opening_width()


                # closest_pair = {
                #         'pair': (left_obstacle, right_obstacle),
                #         'opening_width': opening_width_perpendicular
                #     }
                # Iterate through each batch
                # minimal_width =  self.get_formation_minimal_through_width(self.current_formation_type, len(self.world.agents), self.inter_robot_min_dist, self.inter_robot_obs_min_dist, self.agent_radius)
                
                for dim in range(self.world.batch_dim):
            
                
                    # Get the left and right obstacles
                    # closest_pair = {
                    #         'pair': (left_obstacle, right_obstacle),
                    #         'left_opening': left_opening,
                    #         'right_opening': right_opening
                    #     }
                    
                    if closest_pair_list[dim] == None:
                        current_pos = self.formation_center.state.pos[dim, :].squeeze()
                        print("current pos:{}".format(current_pos))
                        current_direction = self.formation_center.state.rot[dim]
                        print("current rot:{}".format(current_direction))

                        step_direction = torch.tensor([torch.cos(current_direction), torch.sin(current_direction)])
                        step_direction = step_direction / torch.norm(step_direction)  # Normalize
                        
                        # Compute a tentative next position
                        tentative_next_pos = current_pos + step_direction * self.velocity_limit
                    else:
                        (left_obstacle, right_obstacle) = closest_pair_list[dim]['pair']
                        left_opening = closest_pair_list[dim]['left_opening']
                        right_opening = closest_pair_list[dim]['right_opening']
                        print("left_obs:{}".format(left_obstacle))
                        print("right_obs:{}".format(right_obstacle))
                        print("left_opening:{}".format(left_opening))
                        print("right opening:{}".format(right_opening))

                    
                        current_pos = self.formation_center.state.pos[dim, :].squeeze()
                        print("current pos:{}".format(current_pos))
                        current_direction = self.formation_center.state.rot[dim]
                        print("current rot:{}".format(current_direction))

                        step_direction = torch.tensor([torch.cos(current_direction), torch.sin(current_direction)])
                        step_direction = step_direction / torch.norm(step_direction)  # Normalize
                        
                        # Compute a tentative next position
                        tentative_next_pos = current_pos + step_direction * self.velocity_limit
                        
                        # Adjust the tentative next position if any obstacles are too close
                        print("left_obs:{}".format(left_obstacle))
                        print("right_obs:{}".format(right_obstacle)) 
                        if left_obstacle != None or right_obstacle != None:

                            if left_obstacle == None or right_obstacle == None:
                                if left_obstacle == None:
                                    dist_to_obstacle = torch.norm(tentative_next_pos - right_obstacle)
                                    if dist_to_obstacle < 1.3:  # Example threshold for obstacle avoidance
                                        # print(f"Obstacle too close at {obs_pos}, adjusting position.")
                                        avoidance_direction = (tentative_next_pos - right_obstacle) / dist_to_obstacle
                                        tentative_next_pos += avoidance_direction * 0.1  # Move slightly away from obstacle
                                        
                                        current_direction += 0.14
                                        # input("turn left")
                                elif right_obstacle == None:
                                    dist_to_obstacle = torch.norm(tentative_next_pos - left_obstacle)
                                    if dist_to_obstacle < 1.3:  # Example threshold for obstacle avoidance
                                        # print(f"Obstacle too close at {obs_pos}, adjusting position.")
                                        avoidance_direction = (tentative_next_pos - left_obstacle) / dist_to_obstacle
                                        tentative_next_pos += avoidance_direction * 0.1  # Move slightly away from obstacle
                                        current_direction   -= 0.14
                                        # input("turn right")

                                else:
                                    print("something is wrong, if opening is 0, should have only one obstacle")
                                    exit()
                            elif left_obstacle is not None and right_obstacle is not None:
                                if left_opening + right_opening > 2*self.agent_radius + self.inter_robot_min_dist:
                                    
                                    if left_opening > right_opening and left_opening - right_opening > 0.3:
                                        current_direction += 0.1
                                    elif right_opening > left_opening and right_opening - left_opening > 0.3:
                                        current_direction -= 0.1
                                else:
                                    tentative_next_pos = current_pos
                                    current_direction += 1
                            else:
                                print("opening width < 0?")
                                exit()
                            
                    
                    # for obs in near_obstacles:
                    #     obs_pos = obs.squeeze()
                    #     dist_to_obstacle = torch.norm(tentative_next_pos - obs_pos)
                        
                    #     if dist_to_obstacle < 0.5:  # Example threshold for obstacle avoidance
                    #         print(f"Obstacle too close at {obs_pos}, adjusting position.")
                    #         avoidance_direction = (tentative_next_pos - obs_pos) / dist_to_obstacle
                    #         tentative_next_pos += avoidance_direction * 0.1  # Move slightly away from obstacle

                    # # Clip the step to the velocity limit
                    # step_vector = tentative_next_pos - current_pos
                    # if torch.norm(step_vector) > self.velocity_limit:
                    #     step_vector = step_vector / torch.norm(step_vector) * self.velocity_limit
                    #     tentative_next_pos = current_pos + step_vector

                    # Update the robot's position and orientation
                    # self.formation_center.state.pos[dim, :] = tentative_next_pos
                    # self.formation_center.state.rot[dim] = current_direction  # Assuming no orientation change for simplicity
                    if tentative_next_pos[0] > self.world_semidim - 1.1 or tentative_next_pos[0] < -self.world_semidim + 1.1 or tentative_next_pos[1] > self.world_semidim - 1.1 or tentative_next_pos[1] < -self.world_semidim +1.1:
                        tentative_next_pos[0] = torch.empty(1).uniform_(-self.world_semidim + 3, self.world_semidim - 3).item()
                        tentative_next_pos[1] = torch.empty(1).uniform_(-self.world_semidim + 3, self.world_semidim - 3).item()
                        # input("change random positions")
                        # Generate a random direction in the range [0, 2*pi)
                        current_direction = torch.empty(1).uniform_(0, 2 * torch.pi).item()
                        self.current_step_reset = True 
                    self.leader_robot.set_pos(
                        torch.tensor(
                            [
                                tentative_next_pos[0],
                                tentative_next_pos[1],
                            ],
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )
                    self.leader_robot.set_rot(
                        torch.tensor(
                            [
                                current_direction,
                            ],
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )


                    self.formation_center.set_pos(
                        torch.tensor(
                            [
                                tentative_next_pos[0],
                                tentative_next_pos[1],
                            ],
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )
                    self.formation_center.set_rot(
                        torch.tensor(
                            [
                                current_direction,
                            ],
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )
                    self.formation_center_pos[dim, 0] = tentative_next_pos[0]
                    self.formation_center_pos[dim, 1] = tentative_next_pos[1]
                    self.formation_center_pos[dim, 2] = current_direction

                    # self.formation_center_pos[dim, 0] = self.random_path[self.t][0]
                    # self.formation_center_pos[dim, 1] = self.random_path[self.t][1]
                    # self.formation_center_pos[dim, 2] = self.random_path[self.t][2]

                ###################  random walk end######################
            # elif formation_movement == "circle":
            #     ###################  circling ######################
            #     t = self.t / 30
            #     self.formation_center.set_pos(
            #         torch.tensor(
            #             [
            #                 math.cos(t),
            #                 math.sin(t),
            #             ],
            #             device=self.world.device,
            #         ),
            #         batch_index=None,
            #     )
            #     self.formation_center_pos[0] = math.cos(t)
            #     self.formation_center_pos[1] = math.sin(t)
            #     self.formation_center.set_rot(
            #         torch.tensor(
            #             torch.pi,
            #             device=self.world.device,
            #         ),
            #         batch_index=None,
            #     )
            #     self.formation_center_pos[2] = torch.pi
                ###################  circling end ######################
            elif formation_movement == "horizental":
                ###move from left to right, test formation's ability to cross through tunnel
                
                for dim in range(self.world.batch_dim):
                    t = self.t / 30
                    if self.t < 20:
                        self.leader_robot.set_pos(
                            torch.tensor(
                                [
                                    -4,
                                    0,
                                ],
                                device=self.world.device,
                            ),
                            batch_index=dim,
                        )
                        self.formation_center.set_pos(
                            torch.tensor(
                                [
                                    -4,
                                    0,
                                ],
                                device=self.world.device,
                            ),
                            batch_index=dim,
                        )
                        self.formation_center_pos[dim, 0] = -4
                        self.formation_center_pos[dim, 1] = 0
                    else:
                        self.last_leader_robot[0] = ((self.t-1)-20)/30*0.5 - 4
                        self.last_leader_robot[1] = 0
                        self.leader_robot.set_pos(
                            torch.tensor(
                                [
                                    (self.t-20)/30*0.5 - 4,
                                    0,
                                ],
                                device=self.world.device,
                            ),
                            batch_index=dim,
                        )
                        self.formation_center.set_pos(
                            torch.tensor(
                                [
                                    (self.t-20)/30*0.5 - 4,
                                    0,
                                ],
                                device=self.world.device,
                            ),
                            batch_index=dim,
                        )
                        self.formation_center_pos[dim, 0] = ((self.t-20)/30*0.5 - 4)
                        self.formation_center_pos[dim, 1] = 0
                    self.leader_robot.set_rot(
                        torch.tensor(
                            0,
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )
                    self.formation_center.set_rot(
                        torch.tensor(
                            0,
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )
                    self.formation_center_pos[dim, 2] = 0


                
        
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
                    self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + math.cos(self.formation_center_pos[:, 2] + angles[i]) * dists[i]
                    self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + math.sin(self.formation_center_pos[:, 2] + angles[i]) * dists[i]
                    self.formation_normal_width = math.sin(45/180.0*math.pi)*0.5* 4

                # elif self.current_formation_type == "vertical_line":
                #     dists = [0.5, 1, -0.5, -1]
                #     self.formation_goals[i][0] = self.formation_center_pos[0] 
                #     self.formation_goals[i][1] = self.formation_center_pos[1] + dists[i-1]
                
                
                # elif self.current_formation_type == "line":
                #     #直线型
                #     dists = [-0.5, -1, -1.5, -2]
                #     self.formation_goals[i][0] = self.formation_center_pos[0] + dists[i-1]
                #     self.formation_goals[i][1] = self.formation_center_pos[1]
                # elif self.current_formation_type == "rectangle":
                #     #矩形
                #     displacement_x = [1, -1, 1, -1]
                #     displacement_y = [1, 1, -1, -1]
                #     self.formation_goals[i][0] = self.formation_center_pos[0] + displacement_x[i-1]
                #     self.formation_goals[i][1] = self.formation_center_pos[1] + displacement_y[i-1]
                self.formation_goals[i][:,2] = self.formation_center_pos[:, 2]
                for dim in range(self.world.batch_dim):
                    self.formation_goals_landmark[i].set_pos(
                        torch.tensor(
                        [
                            self.formation_goals[i][dim, 0],
                            self.formation_goals[i][dim, 1],
                        ],
                        device=self.world.device,
                    ),
                        batch_index=dim,
                    )
    
                    self.formation_goals_landmark[i].set_rot(
                        torch.tensor(
                        [
                            self.formation_goals[i][dim, 2],
                        ],
                        device=self.world.device,
                    ),
                        batch_index=dim,
                    )
                # print("formation goal {}, {}".format(i, self.formation_goals_landmark[i].state.pos))
        
        if agent == self.world.agents[3]:
            print("before self.t:{}".format(self.t))
            if not self.current_step_reset:

            # input("before optimization")
                self.expert_policy()
            else:
                self.current_step_reset = False
            # if self.need_to_reconfigure == False:
            #     self.expert_policy()
            #     print("1")
            # else:
            #     print("2")
            #     print("self.t:{}".format(self.t))
            #     print("self.reconfigure_success_t:{}".format(self.reconfigure_success_t))
            #     if (self.t - self.reconfigure_success_t)% 10 ==0:
            #         self.expert_policy()
            #     else:
            #         for idx in range(self.n_agents):
            #             self.formation_goals_landmark[idx].set_pos(torch.tensor([self.success_reconfigure_goals[idx][0] + self.t_since_success_reconfigure / 30*0.5, self.success_reconfigure_goals[idx][1]], device=self.world.device), batch_index = None)


    def get_formation_surrounding_obstacles(self, dim_index, optim_init_value):
        surrounding_obstacles = []
        tolerance = 0.01
        for agent_index in range(optim_init_value.shape[0]):
            # Get the nearby obstacles for each agent
            # near_obstacles = self.obstacle_manager_list[dim_index].get_near_obstacles(optim_init_value[agent_index, :2], 1)
            # Inside the get_formation_surrounding_obstacles method
            near_obstacles = self.obstacle_manager_list[dim_index].get_near_obstacles(optim_init_value[agent_index, :2].detach().cpu().numpy(), 0.6)

            for obstacle in near_obstacles:
                # Check if this obstacle is already in the surrounding_obstacles list
                obstacle_coordinates = obstacle[:2]  # Assuming the first two dimensions are coordinates
                
                # Check for overlap; you might want to use a tolerance here
                if not any(np.linalg.norm(obstacle_coordinates - obs[:2]) < tolerance for obs in surrounding_obstacles):
                    surrounding_obstacles.append(obstacle)
        
        return surrounding_obstacles



    def get_leader_forward_obstacles_2(self):
        paired_obstacles_list = []
        closest_pair_list = []

        for d in range(self.world.batch_dim):
            obstacle_manager = self.obstacle_manager_list[d]
            near_obstacles = obstacle_manager.get_near_obstacles(self.leader_robot.state.pos[d, :], 1.2)

            leader_direction = self.leader_robot.state.rot[d]
            print("leader direction:{}".format(leader_direction))

            # Define search parameters
            angle_range = torch.tensor(torch.pi + 0.5)  # Extend 180 degrees (π) to each side for a full forward search
            max_distance = 1.2  # Define max distance for relevant obstacles

            # List to hold valid obstacles and their relative angles
            valid_obstacles = []

            # Filter obstacles within the angle range and distance range
            for obstacle in near_obstacles:
                # Get the relative position of the obstacle with respect to the robot
                rel_pos = obstacle - self.leader_robot.state.pos[d, :]

                # Calculate the angle to the obstacle relative to the robot's current direction
                obstacle_angle = torch.atan2(rel_pos[1], rel_pos[0])

                # Normalize the angle difference
                angle_diff = torch.atan2(torch.sin(obstacle_angle - leader_direction), torch.cos(obstacle_angle - leader_direction))

                # Calculate the distance to the obstacle
                distance = torch.norm(rel_pos)

                # Check if the obstacle is within the angle and distance range
                if -angle_range <= angle_diff <= angle_range and distance <= max_distance:
                    valid_obstacles.append((obstacle, angle_diff, distance))

            # Sort the valid obstacles by their relative angle
            valid_obstacles.sort(key=lambda x: x[1])

            # List to hold paired obstacles for this batch
            paired_obstacles = []
            closest_pair = None
            min_angle_diff = float('inf')  # Initialize with a large value to find the closest pair

            # Pair obstacles that are closest to each other in terms of angle
            for i in range(len(valid_obstacles) - 1):
                right_obstacle = valid_obstacles[i][0]
                left_obstacle = valid_obstacles[i + 1][0]

                # Calculate the norm (Euclidean distance) between the paired obstacles
                opening_width = torch.norm(left_obstacle[:2] - right_obstacle[:2])

                # Append the paired obstacles and their opening width
                paired_obstacles.append({
                    'pair': (left_obstacle, right_obstacle),
                    'opening_width': opening_width
                })

                # Check if this pair is the closest to the leader's direction
                avg_angle_diff = (valid_obstacles[i][1] + valid_obstacles[i + 1][1]) / 2
                if abs(avg_angle_diff) < min_angle_diff:
                    min_angle_diff = abs(avg_angle_diff)
                    print("valid_obstacles[i][1]:{}".format(valid_obstacles[i][1]))
                    print("valid_obstacles[i+1][1]:{}".format(valid_obstacles[i+1][1]))
                    # Determine whether the obstacles are on different sides
                    if valid_obstacles[i][1] < 0 < valid_obstacles[i + 1][1]:  # On different sides
                        # Calculate the perpendicular projection for opening width
                        perp_direction = torch.tensor([torch.cos(leader_direction + torch.pi / 2),
                                                    torch.sin(leader_direction + torch.pi / 2)])

                        # Get the relative positions of the left and right obstacles with respect to the leader robot's position
                        left_rel_pos = left_obstacle - self.leader_robot.state.pos[d, :]
                        right_rel_pos = right_obstacle - self.leader_robot.state.pos[d, :]

                        # Project the relative positions onto the perpendicular direction
                        left_projection = torch.dot(left_rel_pos, perp_direction)
                        right_projection = torch.dot(right_rel_pos, perp_direction)

                        # Calculate the left and right openings
                        left_opening = torch.abs(left_projection)
                        right_opening = torch.abs(right_projection)

                        closest_pair = {
                            'pair': (left_obstacle, right_obstacle),
                            'left_opening': left_opening,
                            'right_opening': right_opening
                        }
                        break
                    else:  # Obstacles are on the same side, one obstacle will be set to None
                        if valid_obstacles[i][1] > 0:  # Both on the left
                            left_obstacle = valid_obstacles[i][0]
                            left_opening = torch.abs(torch.dot(left_obstacle - self.leader_robot.state.pos[d, :], torch.tensor([torch.cos(leader_direction + torch.pi / 2), torch.sin(leader_direction + torch.pi / 2)])))
                            right_obstacle = None
                            right_opening = torch.tensor(2.0)
                        elif valid_obstacles[i+1][1] < 0:  # Both on the right
                            right_obstacle = valid_obstacles[i + 1][0]
                            right_opening = torch.abs(torch.dot(right_obstacle - self.leader_robot.state.pos[d, :], torch.tensor([torch.cos(leader_direction + torch.pi / 2), torch.sin(leader_direction + torch.pi / 2)])))
                            left_obstacle = None
                            left_opening = torch.tensor(2.0)

                        closest_pair = {
                            'pair': (left_obstacle, right_obstacle),
                            'left_opening': left_opening,
                            'right_opening': right_opening
                        }

            # Append the results for this batch
            paired_obstacles_list.append(paired_obstacles)
            closest_pair_list.append(closest_pair)

        return paired_obstacles_list, closest_pair_list
    
    def get_leader_forward_obstacles(self):
        forward_obstacles = []
        
        for d in range(self.world.batch_dim):
            obstacle_manager = self.obstacle_manager_list[d]
            near_obstacles = obstacle_manager.get_near_obstacles(self.leader_robot.state.pos[d, :], 1.3)

            self.leader_robot.state.rot[d]
            leader_direction = self.leader_robot.state.rot[d]
            print("leader direction:{}".format(leader_direction))
        
            # Define search parameters
            angle_range = torch.tensor(torch.pi /2.0)  # Extend 60 degrees to each side
            angles = torch.tensor([leader_direction - angle_range, leader_direction + angle_range])

            # Initialize variables to store the first obstacle for each direction
            first_obstacle_left = None
            first_obstacle_right = None
            min_dist_left = torch.tensor(float('inf'))
            min_dist_right = torch.tensor(float('inf'))

            # Iterate through all near obstacles to find the first one on each side
            for obstacle in near_obstacles:
                # Get the relative position of the obstacle with respect to the robot
                rel_pos = obstacle - self.leader_robot.state.pos[d, :]
                
                # Calculate the angle to the obstacle relative to the robot's current direction
                obstacle_angle = torch.atan2(rel_pos[1], rel_pos[0])

                # Normalize the angle difference
                angle_diff = obstacle_angle - leader_direction
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))  # Normalize between -π and π

                # Calculate the distance to the obstacle
                distance = torch.norm(rel_pos)

                # Check if the obstacle is within the left or right search range
                if -angle_range <= angle_diff <= 0 and distance < min_dist_right:
                    first_obstacle_right = obstacle
                    min_dist_right = distance
                elif 0 < angle_diff <= angle_range and distance < min_dist_left:
                    first_obstacle_left = obstacle
                    min_dist_left = distance

            # Append the found obstacles to the list, even if one side doesn't find any
            forward_obstacles.append({
                'left': first_obstacle_left,
                'right': first_obstacle_right
            })
    
        return forward_obstacles

    def calculate_forward_opening_width(self):
        obstacles = self.get_leader_forward_obstacles()

        # Initialize an empty list to store the opening width for each batch
        opening_widths = []

        # Iterate through each batch
        for d, batch_obstacles in enumerate(obstacles):
            # Get the left and right obstacles
            left_obstacle = batch_obstacles['left']
            right_obstacle = batch_obstacles['right']

            # If both obstacles exist, calculate the perpendicular distance between them
            if left_obstacle is not None and right_obstacle is not None:
                # Get the leader robot's direction (assumed to be in radians)
                leader_direction = self.leader_robot.state.rot[d]
                
                # Calculate the direction perpendicular to the leader robot's facing direction
                # Perpendicular direction: rotate the leader's direction by 90 degrees (π/2 radians)
                perp_direction = torch.tensor([torch.cos(leader_direction + torch.pi / 2),
                                            torch.sin(leader_direction + torch.pi / 2)])
                
                # Get the relative positions of the left and right obstacles with respect to the leader robot's position
                left_rel_pos = left_obstacle - self.leader_robot.state.pos[d, :]
                right_rel_pos = right_obstacle - self.leader_robot.state.pos[d, :]
                
                # Project the relative positions onto the perpendicular direction
                left_projection = torch.dot(left_rel_pos, perp_direction)
                right_projection = torch.dot(right_rel_pos, perp_direction)
                
                # Calculate the opening width as the absolute difference between the two projections
                opening_width = torch.abs(left_projection - right_projection)
            elif left_obstacle is not None or right_obstacle is not None:
                # Get the leader robot's direction (assumed to be in radians)
                leader_direction = self.leader_robot.state.rot[d]
                
                # Calculate the direction perpendicular to the leader robot's facing direction
                # Perpendicular direction: rotate the leader's direction by 90 degrees (π/2 radians)
                perp_direction = torch.tensor([torch.cos(leader_direction + torch.pi / 2),
                                            torch.sin(leader_direction + torch.pi / 2)])
                


                # Get the relative positions of the left and right obstacles with respect to the leader robot's position
                if left_obstacle is not None:
                    rel_pos = left_obstacle - self.leader_robot.state.pos[d, :]
                elif right_obstacle is not None:
                    rel_pos = right_obstacle - self.leader_robot.state.pos[d, :]

                projection = torch.dot(rel_pos, perp_direction)
                # Project the relative positions onto the perpendicular direction
                
                # Calculate the opening width as the absolute difference between the two projections
                opening_width = 2*torch.abs(projection)
            else:
                opening_width = torch.tensor(0.0)

                # If one of the obstacles is missing, set the opening width to 0
                

            # Append the opening width to the list
            opening_widths.append(opening_width)

        # Convert the list of opening widths to a tensor and return
        return torch.stack(opening_widths)

    def get_formation_minimal_through_width(self, formation_type, follower_formation_num, d, d_obs, robot_radius):
        if formation_type == "ren_shape":
            minimal_width = follower_formation_num * ((d  + 2*robot_radius)* math.sin(45/180.0*math.pi)) 
            
        elif formation_type == "rectangle":
            minimal_width = d + robot_radius + 2 * d_obs
        return minimal_width

    def init_two_row_formation(self, dim_index, d, leader_position, leader_direction):
        # Get the leader robot's position and direction
        # leader_position = self.leader_robot.state.pos[d, :]
        # leader_direction = self.leader_robot.state.rot[d]
        d = d + 0.05
        # Calculate the unit direction vector for the leader's orientation
        direction_vector = torch.tensor([torch.cos(leader_direction), torch.sin(leader_direction)])
        
        # Perpendicular direction vector for side-by-side row separation
        perp_direction_vector = torch.tensor([-torch.sin(leader_direction), torch.cos(leader_direction)])

        # Initialize a tensor for the formation, with each agent's position
        optim_init_value = torch.zeros((len(self.world.agents), 3))

        # Place agents in two rows behind the leader
        half_agents = len(self.world.agents) // 2
        for i in range(half_agents):
            # Row 1
            position_offset = (i + 1) * d * (-direction_vector)
            optim_init_value[i, :2] = leader_position[:2] + position_offset + (d * perp_direction_vector)
            
            # Row 2
            position_offset = (i + 1) * d * (-direction_vector)
            optim_init_value[i + half_agents, :2] = leader_position[:2] + position_offset - (d * perp_direction_vector)

        return optim_init_value

    def init_three_row_formation(self, dim_index, d, leader_position, leader_direction):
        # Get the leader robot's position and direction
        # leader_position = self.leader_robot.state.pos[dim_index, :]
        # leader_direction = self.leader_robot.state.rot[d]

        # Calculate the unit direction vector for the leader's orientation
        direction_vector = torch.tensor([torch.cos(leader_direction), torch.sin(leader_direction)])
        
        # Perpendicular direction vector for side-by-side row separation
        perp_direction_vector = torch.tensor([-torch.sin(leader_direction), torch.cos(leader_direction)])

        # Initialize a tensor for the formation, with each agent's position
        optim_init_value = torch.zeros((len(self.world.agents), 3))

        # Place agents in three rows behind the leader
        third_agents = len(self.world.agents) // 3
        for i in range(third_agents):
            # Row 1
            position_offset = (i + 1) * d * (-direction_vector)
            optim_init_value[i, :2] = leader_position[:2] + position_offset + (2 * d * perp_direction_vector)
            
            # Row 2
            position_offset = (i + 1) * d * (-direction_vector)
            optim_init_value[i + third_agents, :2] = leader_position[:2] + position_offset
            
            # Row 3
            position_offset = (i + 1) * d * (-direction_vector)
            optim_init_value[i + 2 * third_agents, :2] = leader_position[:2] + position_offset - (2 * d * perp_direction_vector)

        return optim_init_value

    def init_single_row_formation(self, dim_index, d, leader_position, leader_direction):
        # Get the leader robot's position and direction
        # leader_position = self.leader_robot.state.pos[dim_index, :]
        # leader_direction = self.leader_robot.state.rot[dim_index]
        d = d + 0.05
        # Calculate the unit direction vector for the leader's orientation
        direction_vector = torch.tensor([torch.cos(leader_direction), torch.sin(leader_direction)])

        # Initialize a tensor for the formation, with each agent's position
        optim_init_value = torch.zeros((len(self.world.agents), 3))

        # Place each agent in a single row behind the leader, spaced by d_obs
        for i in range(len(self.world.agents)):
            # Calculate the position of the i-th agent
            position_offset = (i + 1) * d * (-direction_vector)  # Move agents behind the leader
            optim_init_value[i, :2] = leader_position[:2] + position_offset

        return optim_init_value

    def get_original_formation_goals(self, dim_index):
        # Retrieve the original formation goals for the specified dimension index
        original_goals = torch.stack([self.formation_goals[i][dim_index, :] for i in range(len(self.world.agents))])
        
        # Calculate the scaling factor based on the opening width
        scaled_goals = original_goals.clone()
        for i in range(len(self.world.agents)):
            angles, dists = self.get_formation_params(i, 1)
            scaled_goals[i, 0] = self.formation_center_pos[dim_index, 0] + math.cos(self.formation_center_pos[dim_index, 2] + angles) * dists
            scaled_goals[i, 1] = self.formation_center_pos[dim_index, 1] + math.sin(self.formation_center_pos[dim_index, 2] + angles) * dists

        return scaled_goals
    

    def scale_formation_goals_to_both_width(self, dim_index, left_opening, right_opening):
        # Retrieve the original formation goals for the specified dimension index
        original_goals = self.get_original_formation_goals(dim_index)
        
        # Clone the original goals to avoid modifying them directly
        scaled_goals = original_goals.clone()
        
        # Calculate scale factors for the left and right sides based on their respective openings
        left_scale_factor = left_opening / self.formation_normal_width if left_opening < 2.0 else 1.0  # Default to 1 if no obstacle (left_opening == 2.0)
        right_scale_factor = right_opening / self.formation_normal_width if right_opening < 2.0 else 1.0  # Default to 1 if no obstacle (right_opening == 2.0)
        print("formation_normal_width:{}".format(self.formation_normal_width))
        print("left scale:{}".format(left_scale_factor))
        print("right scale:{}".format(right_scale_factor))
        
        for i in range(len(self.world.agents)):
            # Retrieve formation parameters
            angles, dists = self.get_formation_params(i, 1.0)  # Assume scale_factor=1.0 returns the original parameters
            
            # Determine the side of the agent relative to the center of the formation
            agent_angle = angles + self.formation_center_pos[dim_index, 2]  # Absolute angle with respect to the formation center
            if math.sin(agent_angle) < 0:  # Left side
                scale_factor = left_scale_factor
            else:  # Right side
                scale_factor = right_scale_factor
            
            # Apply scaling factor based on which side the agent is on
            scaled_goals[i, 0] = self.formation_center_pos[dim_index, 0] + math.cos(agent_angle) * dists * scale_factor
            scaled_goals[i, 1] = self.formation_center_pos[dim_index, 1] + math.sin(agent_angle) * dists * scale_factor
        return scaled_goals

    def scale_formation_goals_to_width(self, dim_index, opening_width):
        # Retrieve the original formation goals for the specified dimension index
        original_goals = self.get_original_formation_goals(dim_index)
        
        # Calculate the scaling factor based on the opening width
        scaled_goals = original_goals.clone()
        scale_factor = opening_width / self.formation_normal_width
        for i in range(len(self.world.agents)):
            angles, dists = self.get_formation_params(i, scale_factor)
            scaled_goals[i, 0] = self.formation_center_pos[dim_index, 0] + math.cos(self.formation_center_pos[dim_index, 2] + angles) * dists
            scaled_goals[i, 1] = self.formation_center_pos[dim_index, 1] + math.sin(self.formation_center_pos[dim_index, 2] + angles) * dists

        return scaled_goals


    def get_optimization_init_value(self, d, d_obs, robot_radius, last_optimized_poses):
        paired_obstacles_list, closest_pair_list = self.get_leader_forward_obstacles_2()
        
        
        
        opening_widths = self.calculate_forward_opening_width()
        optim_init_value_list = []


        for dim_index in range(self.world.batch_dim):
        
            optim_init_value = torch.zeros((len(self.world.agents), 3))
            opening_width = opening_widths[dim_index]
            print("oenning_width:{}".format(opening_width))
            
            
            formation_row_num = 0
            if closest_pair_list[dim_index] == None:
                # keep original formation as init value 
                print("original formation value")
                optim_init_value = self.get_original_formation_goals(dim_index)  # Assuming this method returns the current formation

            else:
                (left_obstacle, right_obstacle) = closest_pair_list[dim_index]['pair']
                left_opening = closest_pair_list[dim_index]['left_opening']
                right_opening = closest_pair_list[dim_index]['right_opening']
                # if self.last_opening_width is not None and self.last_opening_width[dim_index] is not 0.0:
                #     optim_init_value = last_optimized_poses[dim_index] 
                # else:
                total_opening = left_opening + right_opening
                formation_minimal_through_width = self.get_formation_minimal_through_width(self.current_formation_type, len(self.world.agents), d, d_obs, robot_radius)
                print("left_opening:{}".format(left_opening))
                print("right_opening:{}".format(right_opening))
                print("formation_minimal_through_width:{}".format(formation_minimal_through_width))
                print("total_opening:{}".format(total_opening))
                # input("check")
                if total_opening > formation_minimal_through_width:
                    scaled_goals = self.scale_formation_goals_to_both_width(dim_index, left_opening, right_opening)
                    print("scaled original formation value")
                    optim_init_value = scaled_goals                  
                else:



                    
                    if total_opening > 2*robot_radius + d  and total_opening < 4* robot_radius + d :
                        formation_row_num = 1
                        print("single row init value")
                        optim_init_value = self.init_single_row_formation(dim_index, d, self.leader_robot.state.pos[dim_index,:], self.leader_robot.state.rot[dim_index])
                    elif total_opening > 4* robot_radius + d : #and opening_width < 6 * robot_radius + 2 * d + 2* d_obs:
                        formation_row_num = 2
                        print("2 row init value")
                        optim_init_value = self.init_two_row_formation(dim_index, d, self.leader_robot.state.pos[dim_index,:], self.leader_robot.state.rot[dim_index])
                    else:
                        print("what hell is this?{}".format(total_opening))
                        # input("what thf")
                        scaled_goals = self.scale_formation_goals_to_both_width(dim_index, left_opening, right_opening)
                        print("scaled original formation value")
                        optim_init_value = scaled_goals            
                        # input("!!!!!!!!!!!!")
                    # elif opening_width > 6 * robot_radius + 2 * d + 2* d_obs:
                    #     formation_row_num = 3
                    #     print("3 row init value")
                    #     optim_init_value = self.init_three_row_formation(dim_index, d, self.leader_robot.state.pos[dim_index,:], self.leader_robot.state.rot[dim_index])
                    # else:
                    #     print("what hell is this?")
                    #     input("!!!!!!!!!!!!")
            optim_init_value_list.append(optim_init_value)
        # self.last_opening_width = copy.deepcopy(total_opening)
        return optim_init_value_list

    def expert_policy(self):
        # Attempt to update formation with initial scale of 1.0

        optim_init_value_list = self.get_optimization_init_value(self.inter_robot_min_dist, self.inter_robot_obs_min_dist, self.agent_radius, self.last_optimized_formation_poses)
        print("optim_init_value_list:{}".format(optim_init_value_list))
        # input("1")
        optimized_formation_goals_list = None
        optimized_formation_goals_list = self.optimization_process(optim_init_value_list)
        self.last_optimized_formation_poses = []
        
        for dim_index in range(self.world.batch_dim):
            optim_init_value = optim_init_value_list[dim_index]
            optimized_formation_goals = optimized_formation_goals_list[dim_index]
            # print("optim_init_value:{}".format(optim_init_value))
            for agent_index in range(len(self.world.agents)):
                self.formation_goals_landmark[agent_index].set_pos(torch.tensor([optimized_formation_goals[agent_index,0], optimized_formation_goals[agent_index,1]], device=self.world.device), batch_index = dim_index)
            self.last_optimized_formation_poses.append(optimized_formation_goals)
        # if not self.update_formation_positions(1.0):
        #     self.scale_formation_down()
        # else:
        #     print("no collision")

        # actions = []
        # for agent in self.world.agents:
        #     agent_action = self.calculate_agent_action(agent)
        # actions.append(agent_action)
    
        # return actions

    def check_conditions(self, init_pos, leader_pose, surrounding_obstacles):
        positions = init_pos[:, :2]
        # Check for collision-free condition
        for agent_index in range(positions.shape[0]):
            if self.check_collisions(positions[agent_index, :], surrounding_obstacles):
                print("collision with obstalces")
                return False

        # Check for minimal distance between robots
        num_robots = positions.shape[0]
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                if self.distance(torch.tensor(positions[i]), torch.tensor(positions[j])) < self.inter_robot_min_dist:
                    print("too near inter_robots")
                    return False
        for i in range(num_robots):
            if self.distance(positions[i], leader_pose[:2]) < self.inter_robot_min_dist:
                print("too close to leader_robot")
                return False


        orientations = init_pos[:, 2]
        positions_with_leader = torch.cat((positions, leader_pose[:2].unsqueeze(dim = 0)), dim=0)
        print("positions_with_leader shape:{}".format(positions_with_leader.shape))
        print("orientations shape:{}".format(orientations.shape))
        print("leader_pose orientation shape:{}".format(leader_pose[2].unsqueeze(dim=0).shape))
        orientations_with_leader = torch.cat((orientations, leader_pose[2].unsqueeze(dim=0)), dim=0)
        mst_edges, edge_weights = self.form_mst_tensor(positions_with_leader, orientations_with_leader)
        
        for (u, v), weight in zip(mst_edges, edge_weights):
            if not self.is_observable(positions_with_leader[u, :2], positions_with_leader[v, :2], orientations_with_leader[u]) and not self.is_observable(positions_with_leader[v, :2], positions_with_leader[u, :2], orientations_with_leader[v]):
                print("not observable between {} and {}".format(positions_with_leader[u,:2], positions_with_leader[v, :2]))
                return False

        is_leader_observed = False
        for i in range(num_robots):
            if self.is_observable(positions[i,:], leader_pose[:2], orientations[i]):
                is_leader_observed = True
                break
        if is_leader_observed:
            return True
        else:
            print("leader not observable")
            return False
        # return True

    
    def optimization_process(self, optim_init_value_list):
        torch.autograd.set_detect_anomaly(True)
        global robot_poses, cost_map, map_origin, resolution, optimizer, enhanced_cost_map
        start_time = time.time()
        optimized_formation_goals = []
        for dim_index in range(self.world.batch_dim):  
            optim_init_value = optim_init_value_list[dim_index].requires_grad_(True)      
            # optim_init_value = optim_init_value_list[dim_index]
            # print("leader state:{}".format(self.leader_robot.state.rot))
            leader_pos = self.leader_robot.state.pos[dim_index, :].squeeze()  # Shape: [2]
            leader_rot = self.leader_robot.state.rot[dim_index, :]  # Shape: [1]
            # print("leader_pos shape:{}".format(leader_pos.shape))
            # print("leader_rot shape:{}".format(leader_rot))
            leader_pose = torch.cat((leader_pos, leader_rot), dim=0)
            # Combine leader's position and rotation into a [3] tensor
            # leader_state = torch.cat((leader_pos, leader_rot), dim=-1)  # Shape: [3]
    
            # Add leader's state at the beginning of optim_init_value to form optim_init_value_with_leader
            # optim_init_value_with_leader = torch.cat((leader_state.unsqueeze(0), optim_init_value), dim=0)  # Shape: [agent_num + 1, 3]
            # print("optim_init_value_with_leader shape:{}".format(optim_init_value_with_leader.shape))


            init_fixed_optim_value = optim_init_value.clone().detach()     
            # print("init_fixed_optim_value:{}".format(init_fixed_optim_value))                                                       
            has_found_valid_positions = False
            surrounding_obstacles = self.get_formation_surrounding_obstacles(dim_index, optim_init_value)
            for agent_index in range(len(self.world.agents)):
                self.formation_goals_modified[agent_index].set_pos(torch.tensor([init_fixed_optim_value[agent_index,0], init_fixed_optim_value[agent_index, 1]], device=self.world.device), batch_index = dim_index)
            if self.check_conditions(optim_init_value, leader_pose, surrounding_obstacles):
                has_found_valid_positions = True
            # input("after init checking")
            while has_found_valid_positions == False:         

                
                optimizer = optim.Adam([optim_init_value], lr=0.03)
                for i in range(30):
                    # print("befre optimization value:{}".format(optim_init_value))
                    optim_cycle_start = time.time()
                    optimizer.zero_grad()
                    loss = self.objective_function(optim_init_value, init_fixed_optim_value, leader_pos, surrounding_obstacles)
                    loss.backward()
                    optimizer.step()
                    # print("Gradients of optim_init_value:", optim_init_value.grad)
                    # print("optimed value:{}".format(optim_init_value))
                    print("ite:{}, loss:{}".format(i,loss))
                    # input("update")
                    if self.check_conditions(optim_init_value, leader_pose, surrounding_obstacles):
                        input("success")
                        print("**************************************************************************************")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("**************************************************************************************")
                        has_found_valid_positions = True
                        break
                    # input("pause between optim")
                    # print("optim cycle time:{}".format(time.time() - optim_cycle_start))
                input("no success, to end")
                has_found_valid_positions = True
            optimized_positions = optim_init_value.detach()
            
            # print("optim time:{}".format(time.time() - start_time))
            optimized_formation_goals.append(optimized_positions)
        return optimized_formation_goals
            # initial_positions = initial_positions.detach()
            # for i in range(initial_positions.size(0)):
            #     initial_positions[i, 0] -= np.cos(yaw) * 8
            #     initial_positions[i, 1] -= np.sin(yaw) * 8

    def inter_robot_distance(self, pos, leader_pos):
        """
        Calculate the inter-agent distance cost to penalize agents that are too close to each other,
        including the distance between each agent and the leader robot.
        
        Args:
            leader_pos (torch.Tensor): Tensor of shape [2] representing the position of the leader robot.
            pos (torch.Tensor): Tensor of shape [agent_size, 3] representing the positions of agents.
        
        Returns:
            torch.Tensor: The total inter-agent distance cost.
        """
        # Ensure pos has requires_grad=True if gradients are needed
        if not pos.requires_grad:
            pos.requires_grad_(True)
        
        # Calculate pairwise distances between agents (considering only the first two dimensions, x and y)
        dists = torch.norm(pos[:, :2].unsqueeze(1) - pos[:, :2].unsqueeze(0), dim=-1)  # Shape: [agent_size, agent_size]
        
        # Use a large finite value instead of infinity to mask the diagonal
        dists = dists + torch.eye(pos.size(0), device=pos.device) * 1e6  # Large value to ignore self-distances

        # Apply penalty only for distances below the minimum allowed distance
        penalty = torch.relu(self.inter_robot_min_dist - dists)
        
        # Calculate distances between each agent and the leader robot (using only x and y dimensions)
        leader_dists = torch.norm(pos[:, :2] - leader_pos.unsqueeze(0), dim=-1)  # Shape: [agent_size]
        
        # Calculate penalty for distances between agents and the leader robot that are too close
        leader_penalty = torch.relu(self.inter_robot_min_dist - leader_dists)
        
        # Sum the penalties: inter-agent penalties (divided by 2 for double counting) and leader penalties
        total_cost = (penalty.sum() / 2) + leader_penalty.sum()
        
        return total_cost
    # def inter_robot_distance(self, pos, leader_pos):
    #     """
    #     Calculate the inter-agent distance cost to penalize agents that are too close to each other,
    #     including the distance between each agent and the leader robot.
        
    #     Args:
    #         leader_pos (torch.Tensor): Tensor of shape [2] representing the position of the leader robot.
    #         pos (torch.Tensor): Tensor of shape [agent_size, 3] representing the positions of agents.
        
    #     Returns:
    #         torch.Tensor: The total inter-agent distance cost.
    #     """
    #     # Ensure the tensors involved have requires_grad set if needed
    #     pos = pos.clone().detach().requires_grad_(True)
        
    #     # Calculate pairwise distances between agents (considering only the first two dimensions, x and y)
    #     dists = torch.norm(pos[:, :2].unsqueeze(1) - pos[:, :2].unsqueeze(0), dim=-1)  # Shape: [agent_size, agent_size]
        
    #     # Avoid in-place modification: create a new tensor for masking the diagonal
    #     dists = dists + torch.eye(pos.size(0), device=pos.device) * 1e6 
        
    #     # Apply penalty only for distances below the minimum allowed distance
    #     penalty = torch.relu(self.inter_robot_min_dist - dists)
        
    #     # Calculate distances between each agent and the leader robot (using only x and y dimensions)
    #     leader_dists = torch.norm(pos[:, :2] - leader_pos.unsqueeze(0), dim=-1)  # Shape: [agent_size]
        
    #     # Calculate penalty for distances between agents and the leader robot that are too close
    #     leader_penalty = torch.relu(self.inter_robot_min_dist - leader_dists)
        
    #     # Sum the penalties: inter-agent penalties (divided by 2 for double counting) and leader penalties
    #     total_cost = (penalty.sum() / 2) + leader_penalty.sum()
        
    #     return total_cost


    # def inter_robot_distance(self, pos, leader_pos):
    #     """
    #     Calculate the inter-agent distance cost to penalize agents that are too close to each other,
    #     including the distance between each agent and the leader robot.
        
    #     Args:
    #         leader_pos (torch.Tensor): Tensor of shape [2] representing the position of the leader robot.
    #         pos (torch.Tensor): Tensor of shape [agent_size, 3] representing the positions of agents.
        
    #     Returns:
    #         torch.Tensor: The total inter-agent distance cost.
    #     """
    #     # Calculate pairwise distances between agents (considering only the first two dimensions, x and y)
    #     dists = torch.norm(pos[:, :2].unsqueeze(1) - pos[:, :2].unsqueeze(0), dim=-1)  # Shape: [agent_size, agent_size]
        
    #     # Mask the diagonal by setting distances to themselves as infinity to ignore self-distances
    #     dists.fill_diagonal_(float('inf'))
        
    #     # Calculate penalties only for distances below the minimum allowed distance
    #     penalty = torch.relu(self.inter_robot_obs_min_dist - dists)
        
    #     # Calculate distances between each agent and the leader robot (using only x and y dimensions)
    #     leader_dists = torch.norm(pos[:, :2] - leader_pos.unsqueeze(0), dim=-1)  # Shape: [agent_size]
        
    #     # Calculate penalty for distances between agents and the leader robot that are too close
    #     leader_penalty = torch.relu(self.inter_robot_obs_min_dist - leader_dists)
        
    #     # Sum the penalties: inter-agent penalties (divided by 2 for double counting) and leader penalties
    #     total_cost = (penalty.sum() / 2) + leader_penalty.sum()
        
    #     return total_cost
    
    def is_observable(self, pos_i, pos_j, orientation_i):
        dist = torch.norm(pos_i - pos_j)
        # print("pos_i shape:{}".format(pos_i.shape))
        # print("pos_j shape:{}".format(pos_j.shape))
        # print("orien shape:{}".format(orientation_i.shape))
        if dist > self.observe_D:
            return False
        direction = torch.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
        rel_angle = direction - orientation_i
        rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))  # Normalize to [-pi, pi]
        # print("rel_angle shape:{}".format(rel_angle.shape))
        return self.FOV_min <= rel_angle <= self.FOV_max

    def required_rotation_to_observe(self, pos_i, pos_j, orientation_i):
        direction = torch.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
        rel_angle = direction - orientation_i
        rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))  # Normalize to [-pi, pi]

        if self.FOV_min <= rel_angle <= self.FOV_max:
            return torch.tensor(0.0, device=pos_i.device)

        rotation_needed = torch.min(torch.abs(rel_angle - self.FOV_min), torch.abs(self.FOV_max - rel_angle))
        return rotation_needed
    
    def distance(self, pos_i, pos_j):
        return torch.norm(pos_i - pos_j)
    
    def form_mst_tensor(self, positions, orientations):
        num_robots = positions.shape[0]
        edges = []
        edge_weights = []
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                weight = self.distance(positions[i, :], positions[j, :])
                rotation_i = self.required_rotation_to_observe(positions[i, :], positions[j, :], orientations[i])
                rotation_j = self.required_rotation_to_observe(positions[j, :], positions[i, :], orientations[j])
                total_rotation = torch.min(rotation_i, rotation_j)
                edges.append((i, j))
                edge_weights.append(weight + total_rotation)

        sorted_edges = sorted(zip(edge_weights, edges), key=lambda x: x[0])

        parent = list(range(num_robots))
        rank = [0] * num_robots

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                elif rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    rank[root1] += 1

        mst_edges = []
        mst_weights = []
        for weight, (i, j) in sorted_edges:
            if find(i) != find(j):
                union(i, j)
                mst_edges.append((i, j))
                mst_weights.append(weight)

        return mst_edges, mst_weights

    def mst_cost(self, pos, leader_pos):
        """
        Calculate the cost of the minimum spanning tree (MST) among the positions of agents,
        including the connection to the leader robot's position.
        
        Args:
            pos (torch.Tensor): Tensor of shape [agent_size, 3] representing the positions of agents.
            leader_pos (torch.Tensor): Tensor of shape [2] representing the position of the leader robot.
        
        Returns:
            torch.Tensor: The total MST cost, including penalties for unobservable connections.
        """
        # Form the MST among the agent positions using the existing function
        positions = pos[:,:2]
        orientations = pos[:, 2]
        mst_edges, edge_weights = self.form_mst_tensor(positions, orientations)
        
        # Find the closest agent position to the leader position
        leader_dists = torch.norm(pos[:, :2] - leader_pos.unsqueeze(0), dim=-1)  # Shape: [agent_size]
        closest_agent_idx = torch.argmin(leader_dists)  # Index of the closest agent to the leader
        
        # Connect the closest agent to the leader and calculate the distance
        leader_connection_cost = leader_dists[closest_agent_idx]
        
        # Initialize total MST cost
        total_cost = torch.tensor(0.0, device=pos.device, requires_grad=True)
        
        # Add MST edges and penalties for unobservable connections
        for (u, v), weight in zip(mst_edges, edge_weights):
            if not self.is_observable(pos[u, :2], pos[v, :2], pos[u, 2]):
                total_cost = total_cost + weight
        
        # Add the cost of connecting the leader to the MST via the closest agent
        if not self.is_observable(pos[closest_agent_idx, :2], leader_pos, pos[closest_agent_idx, 2]):
            total_cost = total_cost + leader_connection_cost
        
        return total_cost

    

    def collision_cost(self, pos, obstacles):
        """
        Calculate a differentiable collision cost for agents based on their proximity to obstacles.
        
        Args:
            pos (torch.Tensor): Tensor of shape [agent_size, 3] representing agent positions.
            obstacles (list of torch.Tensor): List of tensors, each of shape [3], representing obstacle positions.
        
        Returns:
            torch.Tensor: The total collision cost.
        """
        # Convert the list of obstacle tensors to a single tensor of shape [num_obstacles, 3]
        if obstacles:
            obstacles_tensor = torch.stack(obstacles)  # Convert list to tensor
        else:
            # If no obstacles, return a zero cost
            return torch.tensor(0.0, device=pos.device, requires_grad=True)
        
        # Define a small value to avoid division by zero or extremely high costs
        epsilon = 1e-6
        
        # Calculate the pairwise Euclidean distances between each agent and each obstacle
        # pos[:, :2] is of shape [agent_size, 2]
        # obstacles_tensor[:, :2] is of shape [num_obstacles, 2]
        # This operation will result in a [agent_size, num_obstacles] tensor of distances
        distances = torch.cdist(pos[:, :2], obstacles_tensor[:, :2], p=2)  # Using p=2 for Euclidean distance

        # Calculate the inverse of distances to get the cost; add epsilon to avoid division by zero
        inverse_distances = 1.0 / (distances + epsilon)

        # Sum the inverse distances to get the total collision cost
        cost = torch.sum(inverse_distances)

        return cost


    def objective_function(self, pos, initial_pos, leader_pos, obstacles):
        print("pos shape:{}".format(pos.shape))
        print("obstacle num:{}".format(len(obstacles)))
        print("pos :{}".format(pos))
        print("leader_pos:{}".format(leader_pos))
        change_cost = torch.norm(initial_pos - pos).sum()
        collision_cost_value = 0.1*self.collision_cost(pos, obstacles)
        inter_robot_distance_penalty = 150*self.inter_robot_distance(pos, leader_pos)
        observation_graph_cost = 20*self.mst_cost(pos, leader_pos )
        print("change cost:{}".format(change_cost))
        print("collision cost:{}".format(collision_cost_value))
        print("inter_robot cost:{}".format(inter_robot_distance_penalty))
        print("observation cost:{}".format(observation_graph_cost))
        total_cost =  change_cost +  collision_cost_value + inter_robot_distance_penalty + observation_graph_cost
        # total_cost = 0
        return total_cost

    def initialize_maps(self, num_agents, grid_size=100):
        # Initialize individual maps for each agent
        # individual_maps = {
        #     'radial': torch.zeros((num_agents, grid_size)),
        #     'angular': torch.zeros((num_agents, grid_size))
        # }
        individual_maps = {
            'joint': torch.zeros((num_agents, grid_size, grid_size))
        }
        return individual_maps

    def initialize_group_maps(self, num_agents, grid_size=100):
        # Create joint maps with increased size for groups
        # joint_map = {
        #     'radial': torch.zeros((num_agents, grid_size)),
        #     'angular': torch.zeros((num_agents, grid_size))
        # }
        joint_map = {
            'joint': torch.zeros((num_agents, grid_size, grid_size))
        }
        return joint_map
    
    def reset_joint_maps(self, joint_map, agent_index):
        grid_size = joint_map['joint'].shape[1]
        # Reset all entries to zero
        joint_map['joint'][agent_index, :, :] = torch.zeros((grid_size, grid_size))


    def group_agents(self, agents, search_radius, agent_radius):
        proximity_threshold = 2 * search_radius + 2 * agent_radius
        groups = []
        # Initialize maps for all agents at once
        
        for i, agent in enumerate(agents):
            found_group = False
            for group in groups:
                # Check if current agent is close enough to any agent in the group
                if any(torch.norm(self.formation_goals_landmark[i].state.pos - self.formation_goals_landmark[j].state.pos) <= proximity_threshold for j in group):
                    group.append(i)
                    found_group = True
                    break
            if not found_group:
                # Create a new group with this agent
                groups.append([i])

        return groups
    
    # def group_agents(self, agents, search_radius, agent_radius):
    #     proximity_threshold = 2 * search_radius + 2 * agent_radius
    #     groups = []
    #     joint_maps = {}
    #     for i, agent in enumerate(agents):
    #         found_group = False
    #         for group_key in list(joint_maps.keys()):  # Iterate over keys to handle mutable groups
    #             group = list(group_key)
    #             if any(torch.norm(agent.state.pos - agents[j].state.pos) <= proximity_threshold for j in group):
    #                 group.append(i)
    #                 # Update joint map for the modified group
    #                 new_joint_map = self.initialize_group_maps(len(group))
    #                 joint_maps.pop(group_key)  # Remove old map
    #                 joint_maps[tuple(group)] = new_joint_map  # Insert updated map
    #                 groups.remove(list(group_key))
    #                 groups.append(group)
    #                 found_group = True
    #                 break
    #         if not found_group:
    #             # Create a new group and joint map for the new group
    #             new_group = [i]
    #             groups.append(new_group)
    #             joint_maps[tuple(new_group)] = self.initialize_group_maps(len(new_group))

    #     return groups, joint_maps
    
    def check_obstacle_collisions(self, position, obstacles):
        # for obs in obstacles:
        #     # if self.world.collides(new_position, obs):
        #     distance = self.world.get_distance(position, obs)
        #     if distance<= (self.min_collision_distance + 0.01):
        #         return True
        if position[1] > 0.14 or position[1] < -0.14:
            return True
        
        # for obstacle in obstacles:
            # if torch.norm(position - obstacle.position) < self.min_distance:
                # return True
        return False

    def sample_positions_for_group(self, agents, group, individual_maps, joint_map, search_radius, grid_size, max_group_attempts=50, max_single_attempts=100):
        # Recursive function to attempt to find valid positions for all agents in the group
        def backtrack(index, current_positions):
            if index == len(group):
                return True, current_positions  # Successfully found positions for all agents
            
            agent_index = group[index]
            attempts = 0
            # print("agent index {}  agent pos:{}".format(agent_index, self.formation_goals[agent_index][:2]))
            while attempts < max_single_attempts:
                dx, dy = self.sample_from_maps(individual_maps, joint_map, agent_index, search_radius)
                new_pos = self.formation_goals[agent_index][:2] + torch.tensor([dx, dy], device=agents[agent_index].state.pos.device)
                # print("       ")
                
                # print("new pos:{}".format(new_pos))
                if self.check_obstacle_collisions(new_pos, self.obstacles):
                    # print("collision with obstacle at {}, for index {}".format(attempts, index))

                    self.update_maps(individual_maps, torch.tensor([dx, dy]), agent_index, grid_size, search_radius, is_joint=False)
                elif all(torch.norm(new_pos - pos) >= 2 * (self.agent_radius + 0.04) for pos in current_positions):
                    # Try to solve for the next agent with this new position included
                    # print("success with at {} times, for index {}".format(attempts, index))
                    result, final_positions = backtrack(index + 1, current_positions + [new_pos])
                    if result:
                        return True, final_positions  # Successful for all subsequent agents
                else:
                    # print("collision with robots at {}, for index {}".format(attempts, index))
                    self.update_maps(joint_map, torch.tensor([dx, dy]), agent_index, grid_size, search_radius, is_joint=True)

                attempts += 1
            self.reset_joint_maps(joint_map, agent_index)
            return False, []  # Failed to find a valid position after max_single_attempts

        # Attempt to solve for all agents starting with the first one, allowing multiple retries for the first agent
        group_attempts = 0
        print("          ")
        print("          ")

        while group_attempts < max_group_attempts:
            success, positions = backtrack(0, [])
            if success:
                print("success-------------------")
                return True, positions
            group_attempts += 1
        print("failed-------------------")
        print("          ")
        print("          ")

        return False, []  # Could not find valid positions for all agents after multiple retries



    # def sample_positions_for_group(self, agents, group, individual_maps, joint_map, search_radius, grid_size):
    #     new_positions = []

    #     for index in group:
    #         valid_position = False
    #         for t in range(10000):  # Retry sampling if collisions occur
    #             dx, dy = self.sample_from_maps(individual_maps, joint_map, index, search_radius)
    #             new_pos = self.formation_goals[index][:2] + torch.tensor([dx, dy], device=agents[index].state.pos.device)
    #             # print("new_pos:{}".format(new_pos))
    #             if self.check_obstacle_collisions(new_pos, self.obstacles):
    #                 self.update_maps(individual_maps, torch.tensor([dx, dy]), index, grid_size, search_radius, is_joint=False)
    #                 continue  # Re-sample due to obstacle collision
    #             if all(torch.norm(new_pos - pos) >= 2 * (self.agent_radius+0.04) for pos in new_positions):
    #                 new_positions.append(new_pos)
    #                 valid_position = True
    #                 print("found!!!!!!!!!!!at {}".format(t))
    #                 break
    #             else:
    #                 self.update_maps(joint_map, torch.tensor([dx, dy]), index, grid_size, search_radius, is_joint=True)
    #         if not valid_position:
    #             return False, []  # Failed to find a valid position for this agent
    #     return True, new_positions

    # def sample_from_maps(self, individual_maps, joint_map, agent_index, search_radius):
    #     grid_size = len(individual_maps['radial'][agent_index])
        
    #     # Combine maps directly, ensuring penalties decrease values
    #     combined_radial_map = individual_maps['radial'][agent_index] + joint_map['radial'][agent_index]
    #     combined_angular_map = individual_maps['angular'][agent_index] + joint_map['angular'][agent_index]

    #     # Normalize and sample from the combined radial and angular maps
    #     # Softmax will naturally give lower probabilities to higher negative values (penalties)
    #     radial_distribution = torch.softmax(combined_radial_map, 0)
    #     angular_distribution = torch.softmax(combined_angular_map, 0)

    #     radius_idx = torch.multinomial(radial_distribution, 1).item()
    #     angle_idx = torch.multinomial(angular_distribution, 1).item()

    #     # Convert index to actual radius and angle
    #     radius = (radius_idx / grid_size) * search_radius
    #     angle = (angle_idx / grid_size) * 2 * torch.pi

    #     # Convert polar coordinates to Cartesian coordinates
    #     dx = radius * math.cos(angle)
    #     dy = radius * math.sin(angle)
    #     return dx, dy
    
    def sample_from_maps(self, individual_maps, joint_map, agent_index, search_radius):
        # Flatten the 2D map into a 1D probability distribution for sampling
        flat_map_individual = individual_maps['joint'][agent_index].flatten()
        flat_map_joint = joint_map['joint'][agent_index].flatten()
        # print("flat_map_individual: {}".format(flat_map_individual))
        # print("flat_map_joint: {}".format(flat_map_joint))
        # print("flat_map_individual size:{}".format(flat_map_individual.shape))
        # print("flat_map_joint size:{}".format(flat_map_joint.shape))
        prob_map = flat_map_individual + flat_map_joint
        # print("prob map size:{}".format(prob_map.shape))
        probabilities = torch.softmax(flat_map_individual + flat_map_joint, 0)
        # print("probability:{}".format(probabilities))
        # Sample a single index from the flattened map
        idx = torch.multinomial(probabilities, 1).item()
        
        # Convert the flat index back to two-dimensional indices
        radial_size, angular_size = individual_maps['joint'][agent_index].shape
        radius_idx = idx // angular_size
        angle_idx = idx % angular_size
        
        # Convert indices to actual radius and angle
        radius = (radius_idx / radial_size) * search_radius
        angle = (angle_idx / angular_size) * 2 * math.pi

        # Convert polar coordinates to Cartesian coordinates
        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        return dx, dy

    def update_maps(self, maps, position, agent_index, grid_size, search_radius, is_joint=False):
        radius = torch.sqrt(position[0]**2 + position[1]**2)
        angle = torch.atan2(position[1], position[0])

        radius_idx = int((radius / search_radius) * grid_size)
        angle_idx = int(((angle + math.pi) / (2 * math.pi)) * grid_size)

        # Define Gaussian spread parameters
        radial_sigma = 1
        angular_sigma = 1

        # Compute the Gaussian range indices
        radial_range = torch.arange(max(0, radius_idx - radial_sigma), min(grid_size, radius_idx + radial_sigma + 1))
        angular_range = torch.arange(max(0, angle_idx - angular_sigma), min(grid_size, angle_idx + angular_sigma + 1))

        # Compute a 2D Gaussian update
        r, a = torch.meshgrid(radial_range - radius_idx, angular_range - angle_idx, indexing='ij')
        gaussian_update = torch.exp(-0.5 * (r ** 2 / radial_sigma**2 + a ** 2 / angular_sigma**2)).float()

        # Apply the Gaussian update to the joint map, ensuring only the specific area around the sampled point is affected
        # print("before update:{}".format(maps['joint'][agent_index, radial_range[0]:radial_range[-1]+1, angular_range[0]:angular_range[-1]+1]))
        
        maps['joint'][agent_index, radial_range[0]:radial_range[-1]+1, angular_range[0]:angular_range[-1]+1] = \
        torch.clamp(maps['joint'][agent_index, radial_range[0]:radial_range[-1]+1, angular_range[0]:angular_range[-1]+1] - gaussian_update, min=-2)
        
        
        # maps['joint'][agent_index, radial_range[0]:radial_range[-1]+1, angular_range[0]:angular_range[-1]+1] -= gaussian_update
        # maps['joint'][agent_index, radial_range[0]:radial_range[-1]+1, angular_range[0]:angular_range[-1]+1] = max(-2, maps['joint'][agent_index, radial_range[0]:radial_range[-1]+1, angular_range[0]:angular_range[-1]+1])
        # print("after update: {}".format(maps['joint'][agent_index, radial_range[0]:radial_range[-1]+1, angular_range[0]:angular_range[-1]+1]))



    # def update_maps(self, maps, position, agent_index, grid_size, search_radius, is_joint):
    #     # Compute the radial and angular indices from position
    #     radius = torch.sqrt(position[0]**2 + position[1]**2)
    #     angle = torch.atan2(position[1], position[0])

    #     radius_idx = int((radius / search_radius) * grid_size)
    #     angle_idx = int(((angle + torch.pi) / (2 * torch.pi)) * grid_size)  # Normalize angle to [0, 2*pi]

    #     # Gaussian spread parameters
    #     sigma = 3  # Spread of the Gaussian update
    #     update_range = torch.arange(max(0, radius_idx - sigma), min(grid_size, radius_idx + sigma + 1))
    #     print("update rnage {}".format(update_range))
    #     # Update radial maps    
    #     gaussian_update = torch.exp(-0.5 * ((update_range - radius_idx) / sigma)**2)
    #     if is_joint:
    #         maps['radial'][agent_index][update_range] -= gaussian_update
    #         maps['angular'][agent_index][max(0, angle_idx - sigma):min(grid_size, angle_idx + sigma + 1)] -= 1
    #     else:
    #         maps['radial'][agent_index][update_range] -= gaussian_update
    #         maps['angular'][agent_index][max(0, angle_idx - sigma):min(grid_size, angle_idx + sigma + 1)] -= 1



    def search_for_target_conf_with_optimization(self, agents, search_radius=0.25, grid_size=5, agent_radius=0.1):
        
        groups= self.group_agents(agents, search_radius, agent_radius)

        for group in groups:
            valid, positions = self.sample_positions_for_group(agents, group, individual_maps, joint_maps, search_radius, grid_size)
            if not valid:
                print(f"Failed to find a valid configuration for group {group}.")
                self.t_since_success_reconfigure += 1
                # if self.success_reconfigure_goals[0][0] != 0:
                for idx in group:
                    print("set as last success_reconfigure_goals!!!!!!!!!!!!!!!!!!!!!!!!{},{}!!!!!!!!!".format(self.success_reconfigure_goals[idx][0], self.success_reconfigure_goals[idx][1]))
                    self.formation_goals_landmark[idx].set_pos(torch.tensor([self.success_reconfigure_goals[idx][0] + self.t_since_success_reconfigure / 30*0.5, self.success_reconfigure_goals[idx][1]], device=self.world.device), batch_index = None)
                return  # Optionally retry or handle this failure differently
            # Update agent positions if valid

            for idx, new_pos in zip(group, positions):
                self.formation_goals_landmark[idx].set_pos(torch.tensor([new_pos[0]-0.1, new_pos[1]], device=self.world.device), batch_index = None)
                # agents[idx].state.pos = new_pos
            for i in range(self.n_agents):
                self.t_since_success_reconfigure = 0

                self.success_reconfigure_goals[i][0] = self.formation_goals_landmark[i].state.pos[0][0]
                self.success_reconfigure_goals[i][1] = self.formation_goals_landmark[i].state.pos[0][1]
                print("update success reconfigure goals!!!!!!!!!!!!!!!!!!!!{}, {}".format(self.success_reconfigure_goals[i][0], self.success_reconfigure_goals[i][1]))
        print("All positions updated successfully.")

    def sampling_search_for_target_conf(self, agents, search_radius=0.25, grid_size=5, agent_radius=0.1):
        individual_maps = self.initialize_maps(len(agents), grid_size)
        joint_maps = self.initialize_group_maps(len(agents), grid_size)
        groups= self.group_agents(agents, search_radius, agent_radius)

        for group in groups:
            valid, positions = self.sample_positions_for_group(agents, group, individual_maps, joint_maps, search_radius, grid_size)
            if not valid:
                print(f"Failed to find a valid configuration for group {group}.")
                self.t_since_success_reconfigure += 1
                # if self.success_reconfigure_goals[0][0] != 0:
                for idx in group:
                    print("set as last success_reconfigure_goals!!!!!!!!!!!!!!!!!!!!!!!!{},{}!!!!!!!!!".format(self.success_reconfigure_goals[idx][0], self.success_reconfigure_goals[idx][1]))
                    self.formation_goals_landmark[idx].set_pos(torch.tensor([self.success_reconfigure_goals[idx][0] + self.t_since_success_reconfigure / 30*0.5, self.success_reconfigure_goals[idx][1]], device=self.world.device), batch_index = None)
                return  # Optionally retry or handle this failure differently
            # Update agent positions if valid

            for idx, new_pos in zip(group, positions):
                self.formation_goals_landmark[idx].set_pos(torch.tensor([new_pos[0]-0.1, new_pos[1]], device=self.world.device), batch_index = None)
                # agents[idx].state.pos = new_pos
            for i in range(self.n_agents):
                self.t_since_success_reconfigure = 0

                self.success_reconfigure_goals[i][0] = self.formation_goals_landmark[i].state.pos[0][0]
                self.success_reconfigure_goals[i][1] = self.formation_goals_landmark[i].state.pos[0][1]
                print("update success reconfigure goals!!!!!!!!!!!!!!!!!!!!{}, {}".format(self.success_reconfigure_goals[i][0], self.success_reconfigure_goals[i][1]))
        print("All positions updated successfully.")
         
    def get_expert_action(self, agent):
        index = self.world.agents.index(agent)
        goal_pos = self.formation_goals_landmark[index].state.pos
        current_pos = agent.state.pos
        max_speed = 2.0
        max_rotation = 0.1
        # Compute the vector pointing from agent to the goal
        move_vector = goal_pos - current_pos
        distance_to_goal = torch.norm(move_vector)
        print("move_vector:{}".format(move_vector))
        if distance_to_goal > 0:
            move_vector_normalized = move_vector[0] / distance_to_goal

            # Calculate force magnitude needed to move toward the goal
            force_magnitude = min(max_speed, distance_to_goal / 0.1)

            # Compute the force vector
            force_vector = move_vector_normalized * force_magnitude

            # Calculate the rotational dynamics
            desired_angle = torch.atan2(move_vector_normalized[1], move_vector_normalized[0])
            current_angle = agent.state.rot
            angle_difference = (desired_angle - current_angle + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]

            # Assuming agent has a max_rotational_force attribute
            rotational_force = max(min(angle_difference / 0.1, max_rotation), -max_rotation)

        else:
            # If the agent is already at the goal, no force is needed
            force_vector = torch.zeros(2, dtype=torch.float32, device=agent.state.pos.device)
            rotational_force = 0.0

        # Return a tensor that represents the forces to be applied: [force_x, force_y, rotational_force]
        return torch.tensor([[force_vector[0], force_vector[1], rotational_force]], dtype=torch.float32, device=agent.state.pos.device)
    

    def scale_formation_down(self):
        scale = 0.9
        print("scale:0.9")
        while scale > 0.4:
            if self.update_formation_positions(scale):
                print(f"Formation scaled down to {scale} successfully.")
                self.need_to_reconfigure = False
                return
            scale -= 0.1
        
        print("Failed to avoid collisions with minimum formation size. Needs reconfiguration.")

        self.search_for_target_conf_with_optimization(self.world.agents, search_radius=0.35, grid_size=15, agent_radius=0.1)

        # self.sampling_search_for_target_conf(self.world.agents, search_radius=0.35, grid_size=15, agent_radius=0.1)
        self.need_to_reconfigure = True
        self.reconfigure_success_t = copy.deepcopy(self.t)


    def update_formation_positions(self, scale):
        collision_detected = False
        print("update formation positions scale:{}".format(scale))
        for i, agent in enumerate(self.world.agents):
            angles, dists = self.get_formation_params(i, scale)
            new_goal_x = self.formation_center_pos[0] + math.cos(self.formation_center_pos[2] + angles) * dists
            new_goal_y = self.formation_center_pos[1] + math.sin(self.formation_center_pos[2] + angles) * dists
            new_goal_x_forward = new_goal_x + 0.35


            # Check if new goal position causes a collision
            self.formation_goals_landmark[i].set_pos(torch.tensor([new_goal_x, new_goal_y], device=self.world.device), batch_index = None)
            self.formation_goals_modified[i].set_pos(torch.tensor([new_goal_x_forward, new_goal_y], device=self.world.device), batch_index = None)
            self.formation_goals[i][0] = new_goal_x
            self.formation_goals[i][1] = new_goal_y
            if self.check_collisions(self.formation_goals_modified[i], agent, i):
                collision_detected = True


            

        return not collision_detected

    def check_collisions(self, agent_pos, surrounding_obstacles):
        
        for obs in surrounding_obstacles:
            # if self.world.collides(new_position, obs):
            distance = self.distance(obs, agent_pos)
            if distance<= (self.inter_robot_obs_min_dist + 0.01):
                return True
        return False

    def get_formation_params(self, index, scale):
        if self.current_formation_type == "ren_shape":
            if scale > 1.0:
                scale = 1.0
            angles = [-45/180.0 * math.pi, 45/180.0 * math.pi, -45/180.0 * math.pi, 45/180.0 * math.pi][index]

            # angles = [-135/180.0 * math.pi, 135/180.0 * math.pi, -135/180.0 * math.pi, 135/180.0 * math.pi][index]
            dists = [-0.5, -0.5, -1, -1][index] * scale
        elif self.current_formation_type == "line":
            angles = 0  # All agents align in a straight line
            dists = [-0.5 * (i + 1) for i in range(len(self.world.agents))][index] * scale
        # Add other formation types here
        return angles, dists
        
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
            # for i, a in enumerate(self.world.agents):
            #     for obs in self.obstacles:
            #         # if self.world.collides(a, obs):
            #         distance = self.world.get_distance(a, obs)
            #         a.agent_collision_rew[
            #             distance <= self.min_collision_distance
            #         ] += self.agent_collision_penalty    
            
            # print("---------------------before----------------------")
            # self.formation_maintain_rew = self.agent_reward_graph_formation_maintained()
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
        return agent.agent_collision_rew + self.formation_maintain_rew
    


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
        # for e in range(num_envs):
            # print("agent.goal.state.pos:{}".format(agent.goal.state.pos))
            # self.formation_goals_modified[current_agent_index].state.pos[e] = agent.goal.state.pos[e]
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
        # for e in range(num_envs):
            # print("agent.goal.state.pos:{}".format(agent.goal.state.pos))
            # self.formation_goals_modified[current_agent_index].state.rot[e] = agent.goal.state.rot[e]
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
        # observation_tensor = torch.cat(
        #     relative_to_leader_pose + relative_to_other_robots_pose + relative_to_other_formation_pose +relative_goal_position_tensor_list+ (
        #         [(agent.sensors[0]._max_range - agent.sensors[0].measure()) / agent.sensors[0]._max_range]
        #         if self.collisions
        #         else []
        #     ),
        #     dim=-1
        # )

        observation_tensor = torch.cat(
            relative_to_leader_pose + relative_to_other_robots_pose  + (
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
                    # geoms.append(line)
        D = 0.6  # Example distance threshold
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
                        # Calculate the relative angle using PyTorch operations only
                        theta_ij = torch.atan2(rel_pos[1], rel_pos[0]) - rot_i

                        # Normalize the angle to be within the range [-π, π] using PyTorch
                        theta_ij = torch.atan2(torch.sin(theta_ij), torch.cos(theta_ij))

                        if self.FOV_min <= theta_ij <= self.FOV_max:
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
