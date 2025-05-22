#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import copy
import math
import os
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
from vmas.make_vmas_env import make_env
from scipy.optimize import linear_sum_assignment
from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box, Line
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from scipy.spatial import KDTree
import random
if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class ObstacleManager:
    def __init__(self, obstacles):
        # 假设 obstacles 是形状为 [N,2] 的 torch.Tensor，且已在 GPU 上
        if len(obstacles) > 0:
            # 直接将障碍物堆叠成张量并保留在 GPU
            self.obstacles_tensor = torch.stack(obstacles) if isinstance(obstacles, list) else obstacles
            self.obstacles_tensor = self.obstacles_tensor.to(torch.float16)  # 统一数据类型
        else:
            self.obstacles_tensor = None
    
    def get_near_obstacles(self, query_pos, radius):
        """完全基于 GPU 的向量化实现"""
        if self.obstacles_tensor is None:
            return []
        
        # 计算距离平方（避免开平方）
        deltas = self.obstacles_tensor - query_pos.unsqueeze(0)
        distances_sq = torch.sum(deltas ** 2, dim=1)
        
        # 使用布尔掩码筛选
        mask = distances_sq <= (radius ** 2)
        return self.obstacles_tensor[mask]

# class ObstacleManager:
#     def __init__(self, obstacles):
#         # Assumes obstacles is a list of torch.Tensor objects of shape [2]
#         self.obstacles = obstacles
        
#         if len(obstacles) > 0:
#             # Convert torch.Tensors to numpy arrays for KDTree
#             self.obstacle_positions = np.array([obs.cpu().numpy() for obs in self.obstacles])
            
#             # Build the KDTree based on the obstacles' positions
#             self.tree = KDTree(self.obstacle_positions)
#         else:
#             # Handle the empty obstacles case
#             self.obstacle_positions = None
#             self.tree = None
    
#     def get_near_obstacles(self, query_pos, radius):
#         # If no obstacles exist, return an empty list
#         if self.tree is None:
#             return []
        
#         # Query the KDTree for all obstacles within the given radius
#         indices = self.tree.query_ball_point(query_pos, radius)
        
#         # Retrieve the corresponding obstacles as torch.Tensors
#         return [self.obstacles[i] for i in indices]

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

class AgentHistory:
    def __init__(self, batch_dim, num_agents, history_length, device):
        """
        Initializes the history buffer for agents across all batch dimensions.

        Args:
            batch_dim (int): Number of parallel environments.
            num_agents (int): Number of agents.
            history_length (int): Number of past positions to store.
            device (torch.device): Device to store the history tensors.
        """
        self.batch_dim = batch_dim
        self.num_agents = num_agents
        self.history_length = history_length
        self.device = device
        # Initialize history buffer with zeros
        # Shape: [batch_dim, num_agents, history_length, 2]
        self.history = torch.zeros((batch_dim, num_agents, history_length, 2), device=self.device)

    def update(self, current_positions):
        """
        Updates the history buffer with current positions.

        Args:
            current_positions (torch.Tensor): Current positions, shape [batch_dim, num_agents, 2].
        """
        if current_positions.shape != (self.batch_dim, self.num_agents, 2):
            raise ValueError(f"Expected current_positions shape {(self.batch_dim, self.num_agents, 2)}, "
                             f"but got {current_positions.shape}")
        # Shift history to the left (remove oldest) and append new positions at the end
        self.history[:, :, :-1, :] = self.history[:, :, 1:, :]
        self.history[:, :, -1, :] = current_positions

    def get_previous_positions(self):
        """
        Retrieves the previous positions for each agent across all batches.

        Returns:
            torch.Tensor: Previous positions, shape [batch_dim, num_agents, history_length, 2].
        """
        return self.history.clone()

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.n_agents = kwargs.get("n_agents", 5)
        self.init_positions_noise_level = kwargs.get("init_position_noise_level", 0)
        self.collisions = kwargs.get("collisions", True)
        self.viewer_size = (1100, 1100)
        self.plot_grid = False
        self.grid_spacing = 1
        self.device =device
        # self.agents_with_same_goal = kwargs.get("agents_with_same_goal", 1)
        # self.split_goals = kwargs.get("split_goals", False)
        self.observe_all_goals = kwargs.get("observe_all_goals", True)
        self.is_imitation = kwargs.get("is_imitation", True)
        self.working_mode = kwargs.get("working_mode", "imitation")
        # if self.is_imitation:
        #     self.working_mode = "imitation"
        # else:
        #     self.working_mode = "RL"
        self.obstacle_pattern = 1
        self.env_type = kwargs.get("env_type", "narrow")
        self.is_evaluation_mode = kwargs.get("is_evaluation_mode", False)
        self.evaluation_index = kwargs.get("evaluation_index", 0)

        self.lidar_range = kwargs.get("lidar_range", 1.8)
        if self.working_mode == "imitation":
            self.agent_radius = kwargs.get("agent_radius", 0.5)
        else:
            self.agent_radius = kwargs.get("agent_radius", 0.15)

        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", False)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.2)
        self.evaluation_noise = kwargs.get("evaluation_noise", 0.0)
        

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 7
        if self.env_type == "mixed_in_distribution":
            self.world_semidim_x = 40
            self.world_semidim_y = 7
        else:
            self.world_semidim_x = 20
            self.world_semidim_y = 7
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

        #reward-related parameters
        self.max_obstacle_include_range = 3.5
        self.max_obstacle_edge_range = 1.5

        # self.max_connection_distance = 1.7  # Example distance threshold
        self.max_connection_distance = 1.9  # Example distance threshold
        
        self.FOV_min = -0.3 * torch.pi
        self.FOV_max = 0.3 * torch.pi
        self.min_collision_distance = 0.6
        self.min_target_collision_distance = 0.3
        self.connection_reward_positive = 0.1  # Positive reward
        self.connection_reward_negative = -0.1  # Negative reward
        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -0.1)
        self.agent_velocity_target_direction_alignment_reward_weight = 0.3
        self.obstacle_center_list = []
        self.obstacle_radius_list = []
        self.route_point_list = []
        self.precomputed_route_point_list = {}
        self.env_observation = []

        self.precompute_obs_dict = {}
        self.evaluation_num = 10



        world = World(batch_dim, device, substeps=2)
        world._x_semidim = self.world_semidim_x
        world._y_semidim = self.world_semidim_y
        # self.world = world
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
                color=Color.GREEN,
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
        self.last_action_u = {}
        self.formation_goals = {}
        self.success_reconfigure_goals = {}
        self.formation_goals_landmark = {}
        for i in range(self.n_agents):   
            self.formation_goals[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            self.success_reconfigure_goals[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            
            
            self.formation_goals_landmark[i] = Landmark(
                name=f"formation goal{i}",
                collide=False,
                movable=True,
                rotatable=True,
                color=Color.GREEN,
                renderable=False,
            )
            # self.formation_goals_landmark[i].renderable = False
            world.add_landmark(self.formation_goals_landmark[i])
            
        world.add_landmark(self.formation_center)
        world.add_landmark(self.leader_robot)

        self.create_obstacles(self.obstacle_pattern, world)

        def detect_obstacles(x):
            return x.name.startswith("obs_") or x.name.startswith("agent_") or x.name.startswith("wall")




        #add leader agent
        self.leader_agent = Agent(
                name=f"agent_0",
                collide=self.collisions,
                color=Color.RED,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                dynamics=HolonomicWithRotation(), 
                linear_friction=0,
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
        self.leader_agent.pos_rew = torch.zeros(batch_dim, device=device)
        self.leader_agent.angle_rew = torch.zeros(batch_dim, device=device)
        self.leader_agent.agent_collision_rew = self.leader_agent.pos_rew.clone()
        self.leader_agent.formation_rew = self.leader_agent.pos_rew.clone()
        self.leader_agent.target_collision_rew = self.leader_agent.pos_rew.clone()
        self.leader_agent.angle_diff_rew = self.leader_agent.pos_rew.clone()
        self.leader_agent.angle_diff_with_leader_rew = self.leader_agent.pos_rew.clone()
        self.leader_agent.connection_rew = torch.zeros(batch_dim, device=device)


        self.leader_agent.action_diff_rew = self.leader_agent.pos_rew.clone()
        self.leader_agent.target_distance = torch.zeros(batch_dim, device=device)
        self.leader_agent.goal = self.formation_goals_landmark[0]
        world.add_agent(self.leader_agent)
        # world.add_agent(self.leader_agent)
        # Add agents
        for i in range(1, self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)

                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=Color.BLUE,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                dynamics=HolonomicWithRotation(), 
                linear_friction=0,
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
            agent.formation_rew = torch.zeros(batch_dim, device=device)
            agent.target_collision_rew = torch.zeros(batch_dim, device=device)
            agent.connection_rew = torch.zeros(batch_dim, device=device)
            agent.action_diff_rew = torch.zeros(batch_dim, device=device)

            agent.angle_diff_rew = torch.zeros(batch_dim, device=device)
            agent.angle_diff_with_leader_rew = torch.zeros(batch_dim, device=device)
            agent.prev_velocity = torch.zeros((batch_dim, 2), device=device)

            agent.formation_diff_rew = torch.zeros(batch_dim, device=device)
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
        self.connection_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.keep_track_time = torch.zeros(batch_dim, device=device)
        self.update_formation_assignment_time = torch.zeros(batch_dim, device=device)
        self.current_assignments = None
        
        self.observe_D = 0.6
        self.batch_dim = batch_dim
        self.last_policy_output = None
        history_length = 1  # Example: store last 5 positions
        self.agent_history = AgentHistory(
            batch_dim=batch_dim,
            num_agents=self.n_agents,
            history_length=history_length,
            device=device
        )



        self.eva_collision_num = torch.zeros(self.n_agents - 1, self.batch_dim, device=device)
        #number of agents that are connected to leader
        self.eva_connection_num = torch.zeros(self.n_agents - 1, self.batch_dim, device=device)
        self.precompute_evaluation_scene(self.env_type)
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
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
        elif obstacle_pattern == 1:
            #random located obstalces
            self.n_boxes = 50
            self.box_width = 0.1
            # for i in range(self.n_boxes):
            #     obs = Landmark(
            #         name=f"obs_{i}",
            #         collide=True,
            #         movable=False,
            #         shape=Sphere(radius=self.box_width),
            #         color=Color.RED,
            #         # collision_filter=lambda e: not isinstance(e.shape, Sphere),
            #     )
                
            #     self.obstacles.append(obs)
            #     world.add_landmark(obs)
        elif obstacle_pattern == 5:
            #random located obstalces
            self.n_boxes = 80
            self.box_width = 0.1
            for i in range(self.n_boxes):
                obs = Landmark(
                    name=f"obs_{i}",
                    collide=True,
                    movable=False,
                    shape=Sphere(radius=self.box_width),
                    color=Color.RED,
                    # collision_filter=lambda e: not isinstance(e.shape, Sphere),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
        elif obstacle_pattern == 2:
            #random located obstalces
            self.n_boxes = 50
            self.box_width = 0.1
            # for i in range(self.n_boxes):
            #     obs = Landmark(
            #         name=f"obs_{i}",
            #         collide=True,
            #         movable=False,
            #         shape=Sphere(radius=self.box_width),
            #         color=Color.RED,
            #         # collision_filter=lambda e: not isinstance(e.shape, Sphere),
            #     )
                
            #     self.obstacles.append(obs)
            #     world.add_landmark(obs)
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
                    # collision_filter=lambda e: not isinstance(e.shape, Box),
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
                    # collision_filter=lambda e: not isinstance(e.shape, Box),
                )
                
                self.obstacles.append(obs)
                world.add_landmark(obs)
            pass

    def create_polygon_with_center(self, center, num_vertices = 8, polygon_radius = 0.5,sphere_radius=0.1, 
            max_spheres_per_polygon=50, 
            world_semidim=5.0, 
            device='cuda'):
        
        base_angles = torch.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]  # Exclude the last point (2π)
        random_offsets = (torch.rand(num_vertices) - 0.5) * (2 * np.pi / num_vertices * 0.5)  # Up to ±25% of segment
        angles = base_angles + random_offsets
        angles = angles % (2 * np.pi)  # Ensure angles are within [0, 2π)
        angles, _ = torch.sort(angles)  # Sort angles to maintain order around the center
        
        radii = torch.rand(num_vertices) * (0.1) + polygon_radius
        
        # Generate vertices of the polygon
        vertices = []
        for angle, radius in zip(angles, radii):
            vertex = center + torch.tensor(
                [torch.cos(angle) * radius, torch.sin(angle) * radius],
                device=device
            )
            vertices.append(vertex)
        vertices.append(vertices[0])  # Close the polygon
        
        # Prepare to generate positions for spheres along the boundary
        positions = []
        sphere_diameter = 2 * sphere_radius
        total_spheres = 0  # Keep track of the total number of spheres
        
        for i in range(len(vertices) - 1):
            start_pos = vertices[i]
            end_pos = vertices[i + 1]
            segment_vector = end_pos - start_pos
            segment_length = torch.norm(segment_vector)
            direction = segment_vector / segment_length  # Normalize to get direction
            
            # Calculate the number of spheres to minimize gaps
            num_spheres_along_edge = max(int(torch.ceil(segment_length / sphere_diameter).item()), 1) + 1
            
            # Adjust if total spheres exceed the maximum allowed
            if total_spheres + num_spheres_along_edge > max_spheres_per_polygon:
                num_spheres_along_edge = max_spheres_per_polygon - total_spheres
                if num_spheres_along_edge <= 0:
                    break  # Stop adding spheres if maximum reached
            
            # Adjust the spacing to minimize gaps
            if num_spheres_along_edge > 1:
                spacing = segment_length / (num_spheres_along_edge - 1)
            else:
                spacing = segment_length / 2.0  # Only one sphere on this edge
            
            # Place spheres along the edge
            for idx in range(num_spheres_along_edge):
                offset = spacing * idx
                sphere_pos = start_pos + offset * direction
                positions.append(sphere_pos)
                total_spheres += 1
                if total_spheres >= max_spheres_per_polygon:
                    break  # Stop adding spheres if maximum reached
            if total_spheres >= max_spheres_per_polygon:
                break  # Stop adding spheres if maximum reached

        return positions
    def generate_route_points(self, distance_threshold, opening_threshold):

        """
        Generate route points between nearby polygons in the 2D space.

        Args:
        - distance_threshold (float): Maximum distance to consider polygons as nearby.
        - opening_threshold (float): Minimum free opening to create a route point.

        Returns:
        - self.route_point_list (list of torch.Tensor): Generated route points.
        """
        self.route_point_list = []

        # Number of polygons
        num_polygons = len(self.obstacle_center_list)
        centers = torch.stack(self.obstacle_center_list)  # Shape: (N, 2)
        radii = torch.tensor(self.obstacle_radius_list, dtype=torch.float32, device=centers.device)  # Shape: (N,)

        # Compute pairwise distances between centers
        dist_matrix = torch.cdist(centers, centers, p=2)  # Shape: (N, N)

        # Iterate over all unique pairs of polygons
        for i in range(num_polygons):
            for j in range(i + 1, num_polygons):
                # Check if polygons are nearby
                if dist_matrix[i, j] < distance_threshold:
                    # Compute free opening
                    free_opening = dist_matrix[i, j] - radii[i] - radii[j]
                    if free_opening > opening_threshold:
                        # Compute route point
                        route_point = (centers[i] + centers[j]) / 2
                        self.route_point_list.append(route_point)

        return self.route_point_list
    
    
    def create_small_fixed_obstacle(
            self, 
            existing_centers,
            sphere_radius=0.1,
            min_center_distance=2.0,
            world_semidim=5.0,
            device='cuda',
            left_x = -3,
            right_x = 13,
            bottom_y = -4,
            upper_y = 4,

        ):
            """
            Creates a small obstacle of 1 to 4 spheres arranged in a fixed pattern with random rotation.
            Ensures no overlap with existing obstacles by checking minimum distance between centers.
            
            Patterns:
            - 1 sphere: single sphere at the center.
            - 2 spheres: horizontal line of two spheres touching each other.
            - 3 spheres: equilateral triangle.
            - 4 spheres: square pattern.
            
            Args:
                existing_centers (list[torch.Tensor]): List of existing obstacle centers as (2,)-tensors.
                sphere_radius (float): Radius of each sphere.
                min_center_distance (float): Minimum distance between new obstacle center and existing centers.
                world_semidim (float): Half-dimension of the world. Used to limit random center placement.
                device (str): Torch device.

            Returns:
                positions (list[torch.Tensor]): List of (2,)-tensors giving the positions of spheres in this obstacle.
                existing_centers (list[torch.Tensor]): Updated list of existing centers after adding the new obstacle center.
                center (torch.Tensor): Center of the newly created obstacle.
            """
            # Randomly choose how many spheres form this obstacle (1 to 4)
            num_spheres = np.random.randint(1, 5)
            
            # Attempt to find a suitable center not overlapping with existing obstacles
            max_attempts = 30000
            for attempt in range(max_attempts):
                center = torch.tensor(
                    [
                        np.random.uniform(left_x, right_x),
                        np.random.uniform(bottom_y, upper_y),
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                # Check if center is far enough from existing centers
                if all(torch.norm(center - c) >= min_center_distance for c in existing_centers):
                    break
            else:
                raise ValueError("Could not find a non-overlapping center after maximum attempts.")
            
            # Add the new center
            existing_centers.append(center)
            
            S = 2 * sphere_radius  # Sphere diameter
            positions = []
            
            # Define fixed layouts for each number of spheres
            if num_spheres == 1:
                # One sphere at the center
                offsets = [(0.0, 0.0)]
            elif num_spheres == 2:
                # Two spheres in a horizontal line
                offsets = [
                    (0.0, 0.0),
                    (S, 0.0)
                ]
            elif num_spheres == 3:
                # Three spheres forming an equilateral triangle
                h = np.sqrt(3)/2 * S  # Height of the equilateral triangle
                # Initial positions centered around (0,0)
                offsets = [
                    (-S/2, -h/3),
                    (S/2, -h/3),
                    (0.0, 2*h/3)
                ]
            else:
                # Four spheres forming a square
                centroid = (S/2, S/2)
                offsets = [
                    (-centroid[0], -centroid[1]),
                    (S - centroid[0], -centroid[1]),
                    (-centroid[0], S - centroid[1]),
                    (S - centroid[0], S - centroid[1]),
                ]
            
            # Generate a random rotation angle between 0 and 2*pi
            rotation_angle = np.random.uniform(0, 2 * np.pi)
            cos_theta = math.cos(rotation_angle)
            sin_theta = math.sin(rotation_angle)
            rotation_matrix = torch.tensor(
                [
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ],
                dtype=torch.float32,
                device=device
            )
            
            # Apply rotation to all offsets
            rotated_offsets = [torch.matmul(rotation_matrix, torch.tensor([ox, oy], dtype=torch.float32, device=device)) for ox, oy in offsets]
            
            # Convert offsets to absolute positions by adding the center
            for offset in rotated_offsets:
                pos = center + offset
                positions.append(pos)
            
            return positions, existing_centers, center
    
    def create_line_segment_between_pos(self, start_pos, end_pos, radius=0.1):
            """
            Creates positions for spheres along a line segment defined by a starting point and an end point.

            Args:
                start_pos (torch.Tensor): Starting position. Shape: [2].
                end_pos (torch.Tensor): End position. Shape: [2].
                radius (float): Radius of each sphere. Determines the spacing between spheres. Default is 0.1.

            Returns:
                list[torch.Tensor]: List of positions along the line segment.
            """
            # Calculate the direction and total distance between start and end points
            direction = end_pos - start_pos
            total_distance = torch.norm(direction)
            direction = direction / total_distance  # Normalize to get direction

            # Calculate the fixed spacing (diameter of the sphere)
            fixed_spacing = 2 * radius+ 0.001

            # Calculate the number of spheres needed
            num_spheres = int(total_distance // fixed_spacing) + 1

            # Generate positions along the line segment
            positions = []
            for idx in range(num_spheres):
                offset = fixed_spacing * idx
                sphere_pos = start_pos + offset * direction
                positions.append(sphere_pos)

            return positions


    def spawn_obstacles(self, obstacle_pattern, env_index):
        # print("env_index:{}".format(env_index))
        passage_indexes = []
        j = self.n_boxes // 2
        line_segments = []  # Store line segments to maintain continuity
        invisible_line_segments = []
        polygon_list = []
        
        
        def create_polygon(
            existing_centers,
            num_vertices_min=8, 
            num_vertices_max=12, 
            sphere_radius=0.1, 
            polygon_min_radius = 0.1,
            polygon_max_radius = 0.3, 
            min_center_distance=2,
            max_spheres_per_polygon=50, 
            world_semidim=self.world_semidim, 
            device='cuda'
        ):
            # Random number of vertices between min and max
            num_vertices = np.random.randint(num_vertices_min, num_vertices_max)
            
            # Determine the minimum distance between centers to prevent overlap
            
            # Loop until a suitable center is found
            max_attempts = 3000
            for attempt in range(max_attempts):
                # Generate a random center
                center = torch.tensor(
                    [
                        np.random.uniform(-world_semidim + 4, world_semidim - 3),
                        np.random.uniform(-world_semidim + 3, world_semidim - 3),
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                
                # Check if the center is far enough from existing centers
                if all(torch.norm(center - c) >= min_center_distance for c in existing_centers):
                    break  # Suitable center found
            else:
                raise ValueError("Could not find a non-overlapping center after maximum attempts.")
            
            # Add the new center to the list of existing centers
            existing_centers.append(center)
            
            # Generate evenly spaced angles with a small random perturbation
            base_angles = torch.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]  # Exclude the last point (2π)
            random_offsets = (torch.rand(num_vertices) - 0.5) * (2 * np.pi / num_vertices * 0.5)  # Up to ±25% of segment
            angles = base_angles + random_offsets
            angles = angles % (2 * np.pi)  # Ensure angles are within [0, 2π)
            angles, _ = torch.sort(angles)  # Sort angles to maintain order around the center
            
            min_radius = polygon_min_radius
            max_radius = polygon_max_radius
            # min_radius = 0.02  # Minimum distance from the center
            # max_radius = 0.05  # Maximum distance from the center
            mean_radius = np.random.uniform(min_radius, max_radius)
            radii = torch.rand(num_vertices) * (0.1) + mean_radius
            
            # Generate vertices of the polygon
            vertices = []
            for angle, radius in zip(angles, radii):
                vertex = center + torch.tensor(
                    [torch.cos(angle) * radius, torch.sin(angle) * radius],
                    device=device
                )
                vertices.append(vertex)
            vertices.append(vertices[0])  # Close the polygon
            
            # Prepare to generate positions for spheres along the boundary
            positions = []
            sphere_diameter = 2 * sphere_radius
            total_spheres = 0  # Keep track of the total number of spheres
            
            for i in range(len(vertices) - 1):
                start_pos = vertices[i]
                end_pos = vertices[i + 1]
                segment_vector = end_pos - start_pos
                segment_length = torch.norm(segment_vector)
                direction = segment_vector / segment_length  # Normalize to get direction
                
                # Calculate the number of spheres to minimize gaps
                num_spheres_along_edge = max(int(torch.ceil(segment_length / sphere_diameter).item()), 1) + 1
                
                # Adjust if total spheres exceed the maximum allowed
                if total_spheres + num_spheres_along_edge > max_spheres_per_polygon:
                    num_spheres_along_edge = max_spheres_per_polygon - total_spheres
                    if num_spheres_along_edge <= 0:
                        break  # Stop adding spheres if maximum reached
                
                # Adjust the spacing to minimize gaps
                if num_spheres_along_edge > 1:
                    spacing = segment_length / (num_spheres_along_edge - 1)
                else:
                    spacing = segment_length / 2.0  # Only one sphere on this edge
                
                # Place spheres along the edge
                for idx in range(num_spheres_along_edge):
                    offset = spacing * idx
                    sphere_pos = start_pos + offset * direction
                    positions.append(sphere_pos)
                    total_spheres += 1
                    if total_spheres >= max_spheres_per_polygon:
                        break  # Stop adding spheres if maximum reached
                if total_spheres >= max_spheres_per_polygon:
                    break  # Stop adding spheres if maximum reached

            return positions, existing_centers, center, mean_radius
                

        def create_line_segment():
            # Fixed spacing between spheres
            fixed_spacing = 0.01+ 2*0.1
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
            fixed_spacing = 0.01+ 2*0.1
     
            positions = []
            for idx in range(num_spheres):
                offset = fixed_spacing * idx  # Fixed spacing between spheres
                sphere_pos = start_pos + offset * direction
                # print("sphere_pos:{}".format(sphere_pos))
                # print("sphere_pos shape:{}".format(sphere_pos.shape))
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

                    num_polygons = np.random.randint(5, 8)
                    outer_existing_centers = []
                    # Generate polygons and collect positions
                    for _ in range(num_polygons):
                        positions, outer_existing_centers, _, _ = create_polygon(
                            existing_centers=outer_existing_centers,
                            num_vertices_min=8,
                            num_vertices_max=12,
                            sphere_radius=0.1,
                            world_semidim=self.world_semidim,
                            device=self.device
                        )
                        line_segments.extend(positions)

                    # Now, create obstacles equal to the number of positions
                    total_positions = len(line_segments)
                    self.obstacles = []  # Clear any existing obstacles
                    for obs_idx in range(total_positions):
                        obs = Landmark(
                            name=f"obs_{obs_idx}",
                            collide=True,
                            movable=False,
                            shape=Sphere(radius=0.1),
                            color=Color.RED,
                        )
                        self.obstacles.append(obs)
                        self.world.add_landmark(obs)

                    # Assign positions to obstacles
                    for idx, obs in enumerate(self.obstacles):
                        obs.set_pos(line_segments[idx], batch_index=env_index)
                                
                    # print("obs num:{}".format(len(self.obstacles)))    
                                
                    
                    
                    
                    
                
                # Assign positions from the pre-generated line segments
                positions = []
                for idx in range(i.shape[0]):
                    # Access each element of i
                    i_value = i[idx].item()  # Convert single element tensor to scalar
                    if i_value < len(line_segments):
                        # Add random noise to line_segments[i_value]
                        noise = torch.randn(line_segments[i_value].shape, device=self.device) * 0.001  # Scale noise as needed
                        noisy_position = line_segments[i_value] + noise  # Add noise to the line segment
                        positions.append(noisy_position)
                    else:
                        # Handle cases where i exceeds the number of pre-generated segments
                        random_position = torch.tensor(
                            [
                                np.random.uniform(-self.world_semidim, self.world_semidim),
                                np.random.uniform(-self.world_semidim, self.world_semidim),
                            ],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        positions.append(random_position)

                # Stack the positions into a tensor
                return torch.stack(positions)
            elif obstacle_pattern == 2:

                
                if len(line_segments) == 0:
                    start_pos_1 = torch.tensor(
                        [
                            -3,
                            1.05,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
                    noise_level = 0.05

                    direction_1 = torch.tensor(
                        [
                            1,
                            -0.1,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
                    noise = (torch.rand((), device=direction_1.device) * 2 - 1) * noise_level
                    direction_1[1] += noise
                    direction_1 = direction_1 / torch.norm(direction_1)  # Normalize to get direction

                    start_pos_2 = torch.tensor(
                        [
                            -3,
                            -1.05,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )
                    direction_2 = torch.tensor(
                        [
                            1,
                            0.12,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    )

# Add random noise to the second dimension of direction_2
                    noise = (torch.rand((), device=direction_2.device) * 2 - 1) * noise_level
                    direction_2[1] += noise
                    direction_2 = direction_2 / torch.norm(direction_2)  # Normalize to get direction

                    line_segments.extend(create_certain_line_segment(25, start_pos_1, direction_1))
                    line_segments.extend(create_certain_line_segment(25, start_pos_2, direction_2))

                # Assign positions from the pre-generated line segments
                positions = []
                for idx in range(i.shape[0]):
                    # Access each element of i
                    i_value = i[idx].item()  # Convert single element tensor to scalar
                    if i_value < len(line_segments):
                        # Add random noise to line_segments[i_value]
                        noise = torch.randn(line_segments[i_value].shape, device=self.device) * 0.04 # Scale noise as needed
                        noisy_position = line_segments[i_value] + noise  # Add noise to the line segment
                        positions.append(noisy_position)
                    else:
                        # Handle cases where i exceeds the number of pre-generated segments
                        random_position = torch.tensor(
                            [
                                np.random.uniform(-self.world_semidim, self.world_semidim),
                                np.random.uniform(-self.world_semidim, self.world_semidim),
                            ],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        positions.append(random_position)

                # Stack the positions into a tensor
                return torch.stack(positions)
                
         
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
            elif obstacle_pattern == 5:
                # New obstacle pattern: vertical walls formed by spheres

                if len(line_segments) == 0:
                    # Generate obstacle positions
                    num_clusters = np.random.randint(5, 8)
                    x_positions = np.linspace(-self.world_semidim + 2, self.world_semidim - 1, num_clusters)
                    # print("x_positions:{}".format(x_positions))
                    # input("1")
                    for x in x_positions:
                        y_start = -self.world_semidim + 1
                        y_end = self.world_semidim - 1
                        total_y_length = y_end - y_start

                        num_segments = np.random.randint(3, 5)
                        segment_length = total_y_length / num_segments

                        for seg_idx in range(num_segments):
                            if np.random.rand() < 0.7:
                                
                                wall_y_start = y_start + seg_idx * segment_length
                                wall_y_end = wall_y_start + segment_length * np.random.uniform(0.3, 0.6)

                                y_positions = np.arange(wall_y_start, wall_y_end, 2 * self.box_width + 0.01)

                                for y in y_positions:
                                    position = torch.tensor([x, y], dtype=torch.float32, device=self.device)
                                    line_segments.append(position)

                # Assign positions from the pre-generated line_segments
                positions = []
                for idx in range(i.shape[0]):
                    i_value = i[idx].item()
                    if i_value < len(line_segments):
                        positions.append(line_segments[i_value])
                    else:
                        return None
                        # If we run out of pre-generated positions, assign random positions
                        # positions.append(torch.tensor(
                        #     [
                        #         np.random.uniform(-self.world_semidim, self.world_semidim),
                        #         np.random.uniform(-self.world_semidim, self.world_semidim),
                        #     ],
                        #     dtype=torch.float32,
                        #     device=self.device,
                        # ))
                return torch.stack(positions)


        if self.is_evaluation_mode == False:
            if obstacle_pattern == 2:
                if len(line_segments) == 0:
                    start_pos = torch.tensor([-2.5, -4], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-2.5, -4.5], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([-2.5, -2.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-2.5, 3.5], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([-2.5, -4], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([4, -3.7], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([-2.5, -2.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([2.7, -2.5], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([2.7, -2.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([2.7, 2.8], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([4, -3.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([4, 3.5], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)


                    total_positions = len(line_segments)
                    self.obstacles = []  # Clear any existing obstacles
                    for obs_idx in range(total_positions):
                        obs = Landmark(
                            name=f"obs_{obs_idx}",
                            collide=True,
                            movable=False,
                            shape=Sphere(radius=0.1),
                            color=Color.RED,
                        )
                        self.obstacles.append(obs)
                        self.world.add_landmark(obs)

                    for obs_idx in range(total_positions):
                        for d in range(self.world.batch_dim):
                        
                            noise = torch.randn(line_segments[0].shape, device=self.device) * 0.001 # Scale noise as needed
                        
                            self.obstacles[obs_idx].set_pos(line_segments[obs_idx] + noise, batch_index = d)

                # Create obstacle managers for each batch
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d,:].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
            elif self.env_type == "clutter":
                if len(line_segments) == 0:

                    num_polygons = np.random.randint(15, 18)
                    # num_polygons = 1
                    # num_polygons = 5
                    outer_existing_centers = []
                    polygon_dict = {}
                    # Generate polygons and collect positions
                    for poly_idx in range(num_polygons):
                        positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                            existing_centers=outer_existing_centers,
                            sphere_radius=0.1,
                            min_center_distance=2.0,
                            world_semidim=self.world_semidim,
                            device=self.device,
                            left_x = -self.world_semidim_x + 17,
                            right_x = self.world_semidim_x - 7,
                            bottom_y = -self.world_semidim_y + 4,
                            upper_y = self.world_semidim_y - 4
                        )
                        # positions, outer_existing_centers, _, _ = create_polygon(
                        #     existing_centers=outer_existing_centers,
                        #     num_vertices_min=6,
                        #     num_vertices_max=8,
                        #     polygon_min_radius = 0.02,
                        #     polygon_max_radius = 0.05, 
                        #     min_center_distance=2,
                        #     max_spheres_per_polygon=50, 
                        #     sphere_radius=0.1,
                        #     world_semidim=self.world_semidim,
                        #     device=self.device
                        # )
                        sphere_start_idx = len(line_segments)
                        line_segments.extend(positions)
                        sphere_end_idx = len(line_segments) 
                        polygon_list.append(positions)
                        polygon_dict[poly_idx] = list(range(sphere_start_idx, sphere_end_idx))
                    # Now, create obstacles equal to the number of positions
                    total_positions = len(line_segments)
                    self.obstacles = []  # Clear any existing obstacles
                    for obs_idx in range(total_positions):
                        obs = Landmark(
                            name=f"obs_{obs_idx}",
                            collide=True,
                            movable=False,
                            shape=Sphere(radius=0.1),
                            color=Color.RED,
                        )
                        self.obstacles.append(obs)
                        self.world.add_landmark(obs)

                    # Assign positions to obstacles
                    for polygon_idx, polygon_positions in enumerate(polygon_list):
                        sphere_list = polygon_dict[polygon_idx]
                        # noisy_position = line_segments[i_value] + noise  # Add noise to the line segment
                        for d in range(self.world.batch_dim):
                        
                            noise = torch.randn(line_segments[0].shape, device=self.device) * 0.08 # Scale noise as needed
                        
                            for sphere_idx in sphere_list:
                                self.obstacles[sphere_idx].set_pos(line_segments[sphere_idx] + noise, batch_index = d)
                    # for idx, obs in enumerate(self.obstacles):
                    #     obs.set_pos(line_segments[idx], batch_index=env_index)
                    # print("obs num:{}".format(len(self.obstacles)))

                    self.obstacle_manager_list = []
                    for d in range(self.world.batch_dim):
                        single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                        manager = ObstacleManager(single_batch_obstacles)
                        self.obstacle_manager_list.append(manager)
            
            
            elif self.env_type == "door_and_narrow":
                current_idx = 0
                num_polygons_for_clutter = 5
                # num_polygons = 1
                # num_polygons = 5
                outer_existing_centers = []
                polygon_dict = {}
                
                clutter_left_x = -15
                clutter_right_x = -12
                clutter_bottom_y = -5
                clutter_upper_y = 4

                door_x = -10
                self.door_x = door_x
                door_y = -1 + 2*random.random()
                narrow_x_1 = -7
                narrow_x_2 = -5
                print("door_Y:{}".format(door_y))
                self.route_point_list.append(torch.tensor([-15, 0], dtype=torch.float32, device=self.device))
                self.route_point_list.append(torch.tensor([door_x + 1, door_y], dtype=torch.float32, device=self.device))
                # self.route_point_list.append(torch.tensor([4, 0], dtype=torch.float32, device=self.device))
                self.route_point_list.append(torch.tensor([narrow_x_1, 0], dtype=torch.float32, device=self.device))
                if door_y > 0:
                    self.route_point_list.append(torch.tensor([narrow_x_2, -2], dtype=torch.float32, device=self.device))
                else:
                    self.route_point_list.append(torch.tensor([narrow_x_2, 2], dtype=torch.float32, device=self.device))

                # Generate polygons and collect positions
                for poly_idx in range(int(num_polygons_for_clutter/2)):
                    positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                        existing_centers=outer_existing_centers,
                        sphere_radius=0.1,
                        min_center_distance=2.0,
                        world_semidim=self.world_semidim,
                        device=self.device,
                        left_x = clutter_left_x,
                        right_x = clutter_right_x,
                        bottom_y = 1.0,
                        upper_y = clutter_upper_y,
                    )
                    # sphere_start_idx = len(line_segments)
                    line_segments.extend(positions)
                    # sphere_end_idx = len(line_segments) 
                    # polygon_list.append(positions)
                    # polygon_dict[poly_idx] = list(range(sphere_start_idx, sphere_end_idx))
                for poly_idx in range(int(num_polygons_for_clutter/2)):
                    positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                        existing_centers=outer_existing_centers,
                        sphere_radius=0.1,
                        min_center_distance=2.0,
                        world_semidim=self.world_semidim,
                        device=self.device,
                        left_x = clutter_left_x,
                        right_x = clutter_right_x,
                        bottom_y = clutter_bottom_y,
                        upper_y = -1.0,
                    )
                    # sphere_start_idx = len(line_segments)
                    line_segments.extend(positions)
                
                ##############
                #add obs for door
                ##############
                down_start = door_y -0.75 -0.3 + random.random()*0.6
                up_start    = door_y + 0.75 -0.3 + random.random()*0.6
                down_end = door_y -0.6 -0.1 + random.random()*0.2
                up_end = door_y + 0.6 - 0.1 + random.random()*0.2
                if random.random() < 0.4:
                    length_left = 0.1
                    length_right = 0.1
                else:
                    length_right = random.random()*0.5
                    length_left = random.random()*0.4
                
                    
                
                start_pos = torch.tensor([door_x, down_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x, door_y -4.5], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([door_x, up_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x, door_y + 4.0], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([door_x, down_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x + length_right, down_end], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([door_x, up_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x + length_left, up_end], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)

                ##############
                #add obs for narrow
                ##############
                def add_noise(tensor, noise_level=0.1):
                    noise = torch.randn_like(tensor) * noise_level  # Gaussian noise with mean 0 and std dev = noise_level
                    return tensor + noise
                center_noise_level = 0.01  # Adjust as needed for center position noise
                polygon_radius_noise_level = 0.01  # Adju
                center_list = []
                center_list.append(add_noise(torch.tensor([narrow_x_1, 2.0], dtype=torch.float32, device=self.device), center_noise_level))
                center_list.append(add_noise(torch.tensor([narrow_x_1, -2.0], dtype=torch.float32, device=self.device), center_noise_level))
                center_list.append(add_noise(torch.tensor([narrow_x_2, 0], dtype=torch.float32, device=self.device), center_noise_level))

                polygon_dict = {}
                for poly_idx, center in enumerate(center_list):
                    # polygon_radius = 0
                    polygon_radius = 0.8 + np.random.uniform(-polygon_radius_noise_level, polygon_radius_noise_level)
                    positions =  self.create_polygon_with_center(center, num_vertices = 8, polygon_radius=polygon_radius,sphere_radius=0.1, max_spheres_per_polygon=50, 
            world_semidim=self.world_semidim, 
            device=self.device)
                    print("center device:{}".format(center.device))
                    self.obstacle_center_list.append(center)
                    self.obstacle_radius_list.append(polygon_radius)
                    sphere_start_idx = len(line_segments)
                    line_segments.extend(positions)
                
                
                
                
                total_positions = len(line_segments)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{current_idx + obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # Assign positions to obstacles
                for obs_idx in range(total_positions):
                    self.obstacles[obs_idx].set_pos(line_segments[obs_idx], batch_index = None)
                
                #     obs.set_pos(line_segments[idx], batch_index=env_index)

                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
                    
            elif self.env_type == "narrow":
                def add_noise(tensor, noise_level=0.1):
                    noise = torch.randn_like(tensor) * noise_level  # Gaussian noise with mean 0 and std dev = noise_level
                    return tensor + noise
                center_noise_level = 0.3  # Adjust as needed for center position noise
                polygon_radius_noise_level = 0.1  # Adju
                center_list = []
                center_list.append(add_noise(torch.tensor([-2.5, 1.8], dtype=torch.float32, device=self.device), center_noise_level))
                center_list.append(add_noise(torch.tensor([-2.5, -1.8], dtype=torch.float32, device=self.device), center_noise_level))
                center_list.append(add_noise(torch.tensor([1.2, 0], dtype=torch.float32, device=self.device), center_noise_level))

                polygon_dict = {}
                for poly_idx, center in enumerate(center_list):
                    # polygon_radius = 0
                    polygon_radius = 0.8 + np.random.uniform(-polygon_radius_noise_level, polygon_radius_noise_level)
                    positions =  self.create_polygon_with_center(center, num_vertices = 8, polygon_radius=polygon_radius,sphere_radius=0.1, max_spheres_per_polygon=50, 
            world_semidim=self.world_semidim, 
            device=self.device)
                    self.obstacle_center_list.append(center)
                    self.obstacle_radius_list.append(polygon_radius)
                    sphere_start_idx = len(line_segments)
                    line_segments.extend(positions)
                    sphere_end_idx = len(line_segments) 
                    polygon_list.append(positions)
                    polygon_dict[poly_idx] = list(range(sphere_start_idx, sphere_end_idx))
                total_positions = len(line_segments)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # Assign positions to obstacles
                for polygon_idx, polygon_positions in enumerate(polygon_list):
                    sphere_list = polygon_dict[polygon_idx]
                    # noisy_position = line_segments[i_value] + noise  # Add noise to the line segment
                    for d in range(self.world.batch_dim):
                    
                        noise = torch.randn(line_segments[0].shape, device=self.device) * 0.08 # Scale noise as needed
                    
                        for sphere_idx in sphere_list:
                            self.obstacles[sphere_idx].set_pos(line_segments[sphere_idx] + noise, batch_index = d)
                # for idx, obs in enumerate(self.obstacles):
                #     obs.set_pos(line_segments[idx], batch_index=env_index)
                print("obs num:{}".format(len(self.obstacles)))

                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
                
                third_obs_center = center_list[2]
                first_route_center = copy.deepcopy(third_obs_center)
                first_route_center[0] -= 2.5
                first_route_center[1] = 0
                self.route_point_list.append(add_noise(first_route_center, 0.1))
                if random.random() < 0.5:
                    third_obs_center[1] = -2
                    self.route_point_list.append(third_obs_center)
                else:
                    third_obs_center[1] = 2
                    self.route_point_list.append(third_obs_center)

                # self.route_point_list.append(add_noise(torch.tensor([-1, 0], dtype=torch.float32, device=self.device), 0.2))
                # if random.random() < 0.5:
                #     self.route_point_list.append(add_noise(torch.tensor([1.2, -2], dtype=torch.float32, device=self.device), 0.3))
                # else:
                #     self.route_point_list.append(add_noise(torch.tensor([1.2, 2], dtype=torch.float32, device=self.device), 0.3))

                # self.route_point_list = self.generate_route_points(distance_threshold=3.0, opening_threshold=1.2)

            elif self.env_type == "empty":
                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
            

            elif self.env_type == "door":
                if len(line_segments) == 0:
                    down_start = -0.75 -0.3 + random.random()*0.6
                    up_start    = 0.75 -0.3 + random.random()*0.6
                    down_end = -0.6 -0.1 + random.random()*0.2
                    up_end = 0.6 - 0.1 + random.random()*0.2
                    if random.random() < 0.4:
                        length_left = 0.1
                        length_right = 0.1
                    else:
                        length_right = random.random()*2
                        length_left = random.random()*2
                    
                    self.door_x = -0.5   
                    
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, -4.5], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, 4.0], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)


                   
                    # start_pos = torch.tensor([2.7, -2.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([2.7, 2.8], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
                    # start_pos = torch.tensor([4, -3.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([4, 3.5], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
                    start_pos = torch.tensor([-2.5, -2.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.world.device)
                    invisible_positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    invisible_line_segments.extend(invisible_positions)

                    start_pos = torch.tensor([-2.5, 1.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.world.device)
                    invisible_positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    invisible_line_segments.extend(invisible_positions)



                    total_positions = len(line_segments)
                    self.obstacles = []  # Clear any existing obstacles
                    for obs_idx in range(total_positions):
                        obs = Landmark(
                            name=f"obs_{obs_idx}",
                            collide=True,
                            movable=False,
                            shape=Sphere(radius=0.1),
                            color=Color.RED,
                            renderable=True,
                        )
                        self.obstacles.append(obs)
                        self.world.add_landmark(obs)

                    total_invisible_positions = len(invisible_line_segments)
                    # self.obstacles = []  # Clear any existing obstacles
                    for obs_idx in range(total_invisible_positions):
                        obs = Landmark(
                            name=f"obs_{obs_idx}",
                            collide=True,
                            movable=False,
                            shape=Sphere(radius=0.1),
                            color=Color.BLUE,
                            renderable=False,
                        )
                        
                        self.world.add_landmark(obs)
                        for d in range(self.world.batch_dim):
                            obs.set_pos(invisible_line_segments[obs_idx], batch_index = d)
                    for obs_idx in range(total_positions):
                        for d in range(self.world.batch_dim):
                        
                            noise = torch.randn(line_segments[0].shape, device=self.device) * 0.001 # Scale noise as needed
                        
                            self.obstacles[obs_idx].set_pos(line_segments[obs_idx] + noise, batch_index = d)

                # Create obstacle managers for each batch
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d,:].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
            elif self.env_type == "tunnel":
                if len(line_segments) == 0:
                    down_start = -1.2 -0.6 + random.random()*1.2
                    up_start    = 1.2 -0.6 + random.random()*1.2
                    down_end = -1.2 -0.6 + random.random()*1.2
                    up_end = 1.2 - 0.6 + random.random()*1.2
                    if random.random() > 0.5:
                        up_end = up_start
                        length_left = 4 - 1.5 + random.random()*3
                        length_right = length_left + 2.2 - 0.4 + random.random()*0.8
                        end_target_1 = (torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.world.device) + torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.world.device)) / 2
                        end_target_2 = end_target_1.clone()
                        end_target_2[1] += (6.5 + random.random())
                    else:
                        down_end = down_start
                        length_right = 4 - 1.5 + random.random()*3
                        length_left = length_right + 2.2 - 0.4 + random.random()*0.8

                        end_target_1 = (torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.world.device) + torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.world.device)) / 2
                        end_target_2 = end_target_1.clone()
                        end_target_2[1] -= (6.5 + random.random())
                    

                    #下方入口到下边界
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, -4.5], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    
                    #上方入口到上边界
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, 4.0], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)
                    
                    #下方入口向tunnel内延伸
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)

                    #上方入口向tunnel内延伸
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.world.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    line_segments.extend(positions)


                    if length_right > length_left:
                        start_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.world.device)
                        end_pos = torch.tensor([-0.5 + length_left + 0.5, up_end + 4.5], dtype=torch.float32, device=self.world.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        line_segments.extend(positions)

                        start_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.world.device)
                        end_pos = torch.tensor([-0.5 + length_right - 0.5, down_end + 6.5], dtype=torch.float32, device=self.world.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        line_segments.extend(positions)
                    elif length_right < length_left:
                        start_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.world.device)
                        end_pos = torch.tensor([-0.5 + length_left - 0.5, up_end - 6.5], dtype=torch.float32, device=self.world.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        line_segments.extend(positions)

                        start_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.world.device)
                        end_pos = torch.tensor([-0.5 + length_right + 0.5, down_end - 4.5], dtype=torch.float32, device=self.world.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        line_segments.extend(positions)
                    # start_pos = torch.tensor([2.7, -2.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([2.7, 2.8], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
                    # start_pos = torch.tensor([4, -3.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([4, 3.5], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
                    start_pos = torch.tensor([-2.5, -2.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.world.device)
                    invisible_positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    invisible_line_segments.extend(invisible_positions)

                    start_pos = torch.tensor([-2.5, 2.5], dtype=torch.float32, device=self.world.device)
                    end_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.world.device)
                    invisible_positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    invisible_line_segments.extend(invisible_positions)



                    total_positions = len(line_segments)
                    self.obstacles = []  # Clear any existing obstacles
                    for obs_idx in range(total_positions):
                        obs = Landmark(
                            name=f"obs_{obs_idx}",
                            collide=True,
                            movable=False,
                            shape=Sphere(radius=0.1),
                            color=Color.RED,
                            renderable=True,
                        )
                        self.obstacles.append(obs)
                        self.world.add_landmark(obs)

                    total_invisible_positions = len(invisible_line_segments)
                    for obs_idx in range(total_invisible_positions):
                        obs = Landmark(
                            name=f"obs_{obs_idx}",
                            collide=True,
                            movable=False,
                            shape=Sphere(radius=0.1),
                            color=Color.BLUE,
                            renderable=False,
                        )
                        
                        self.world.add_landmark(obs)
                        for d in range(self.world.batch_dim):
                            obs.set_pos(invisible_line_segments[obs_idx], batch_index = d)
                    for obs_idx in range(total_positions):
                        for d in range(self.world.batch_dim):
                        
                            noise = torch.randn(line_segments[0].shape, device=self.device) * 0.001 # Scale noise as needed
                        
                            self.obstacles[obs_idx].set_pos(line_segments[obs_idx] + noise, batch_index = d)

                # Create obstacle managers for each batch
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d,:].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)

                self.route_point_list.append(end_target_1)

                # center_list = []
                # center_list.append(torch.tensor([-2.5, 1.3+np.random.uniform(0.0, 0.3)], dtype=torch.float32, device=self.device))
                # center_list.append(torch.tensor([-2.5, -1.3-np.random.uniform(0.0, 0.3)], dtype=torch.float32, device=self.device))
                self.route_point_list.append(end_target_2)
                # center_list.append(torch.tensor([0.5, 1.5], dtype=torch.float32, device=self.device))
      
            elif self.env_type == "mixed_in_distribution":
                
                ##############
                #add obs for clutter
                ##############
                current_idx = 0
                num_polygons_for_clutter = 10
                # num_polygons = 1
                # num_polygons = 5
                outer_existing_centers = []
                polygon_dict = {}
                
                clutter_left_x = -15
                clutter_right_x = -4
                clutter_bottom_y = -5
                clutter_upper_y = 4

                door_x = -2

                narrow_x_1 = 9
                narrow_x_2 = 11
                tunnel_x = 2

                self.route_point_list.append(torch.tensor([-4, 0], dtype=torch.float32, device=self.device))
                
                self.route_point_list.append(torch.tensor([1, 0], dtype=torch.float32, device=self.device))
                self.route_point_list.append(torch.tensor([4, 0], dtype=torch.float32, device=self.device))
                self.route_point_list.append(torch.tensor([9, 0], dtype=torch.float32, device=self.device))
                self.route_point_list.append(torch.tensor([11, 2], dtype=torch.float32, device=self.device))

                # Generate polygons and collect positions
                for poly_idx in range(num_polygons_for_clutter):
                    positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                        existing_centers=outer_existing_centers,
                        sphere_radius=0.1,
                        min_center_distance=2.0,
                        world_semidim=self.world_semidim,
                        device=self.device,
                        left_x = clutter_left_x,
                        right_x = clutter_right_x,
                        bottom_y = clutter_bottom_y,
                        upper_y = clutter_upper_y,
                    )
                    # sphere_start_idx = len(line_segments)
                    line_segments.extend(positions)
                    # sphere_end_idx = len(line_segments) 
                    # polygon_list.append(positions)
                    # polygon_dict[poly_idx] = list(range(sphere_start_idx, sphere_end_idx))

                
                ##############
                #add obs for door
                ##############
                down_start = -0.75 -0.3 + random.random()*0.6
                up_start    = 0.75 -0.3 + random.random()*0.6
                down_end = -0.6 -0.1 + random.random()*0.2
                up_end = 0.6 - 0.1 + random.random()*0.2
                if random.random() < 0.4:
                    length_left = 0.1
                    length_right = 0.1
                else:
                    length_right = random.random()*0.5
                    length_left = random.random()*0.4
                
                    
                
                start_pos = torch.tensor([door_x, down_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x, -4.5], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([door_x, up_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x, 4.0], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([door_x, down_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x + length_right, down_end], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([door_x, up_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([door_x + length_left, up_end], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)

                ##############
                #add obs for narrow
                ##############
                def add_noise(tensor, noise_level=0.1):
                    noise = torch.randn_like(tensor) * noise_level  # Gaussian noise with mean 0 and std dev = noise_level
                    return tensor + noise
                center_noise_level = 0.01  # Adjust as needed for center position noise
                polygon_radius_noise_level = 0.01  # Adju
                center_list = []
                center_list.append(add_noise(torch.tensor([narrow_x_1, 2.0], dtype=torch.float32, device=self.device), center_noise_level))
                center_list.append(add_noise(torch.tensor([narrow_x_1, -2.0], dtype=torch.float32, device=self.device), center_noise_level))
                center_list.append(add_noise(torch.tensor([narrow_x_2, 0], dtype=torch.float32, device=self.device), center_noise_level))

                polygon_dict = {}
                for poly_idx, center in enumerate(center_list):
                    # polygon_radius = 0
                    polygon_radius = 0.8 + np.random.uniform(-polygon_radius_noise_level, polygon_radius_noise_level)
                    positions =  self.create_polygon_with_center(center, num_vertices = 8, polygon_radius=polygon_radius,sphere_radius=0.1, max_spheres_per_polygon=50, 
            world_semidim=self.world_semidim, 
            device=self.device)
                    self.obstacle_center_list.append(center)
                    self.obstacle_radius_list.append(polygon_radius)
                    sphere_start_idx = len(line_segments)
                    line_segments.extend(positions)

                ##############
                #add obs for tunnel
                ##############
                down_start = -0.75 -0.3 + random.random()*0.6
                up_start    = 0.75 -0.3 + random.random()*0.6
                down_end = -0.6 -0.1 + random.random()*0.2
                up_end = 0.6 - 0.1 + random.random()*0.2
                length_left = 4.5 - 1.5 + random.random()*1
                if random.random() > 0.5:
                    length_right = length_left + 1.2 - 0.2 + random.random()*0.4
                    end_target_1 = (torch.tensor([tunnel_x + length_left, up_end], dtype=torch.float32, device=self.world.device) + torch.tensor([tunnel_x + length_right, down_end], dtype=torch.float32, device=self.world.device)) / 2
                    end_target_2 = end_target_1.clone()
                    end_target_2[1] += 2.0
                else:
                    length_right = length_left - 1.2 - 0.2 + random.random()*0.4
                    end_target_1 = (torch.tensor([tunnel_x + length_left, up_end], dtype=torch.float32, device=self.world.device) + torch.tensor([tunnel_x + length_right, down_end], dtype=torch.float32, device=self.world.device)) / 2
                    end_target_2 = end_target_1.clone()
                    end_target_2[1] -= 2.0
                
                start_pos = torch.tensor([tunnel_x, down_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([tunnel_x, -4.5], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([tunnel_x, up_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([tunnel_x, 4.0], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([tunnel_x, down_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([tunnel_x + length_right, down_end], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)
                start_pos = torch.tensor([tunnel_x, up_start], dtype=torch.float32, device=self.world.device)
                end_pos = torch.tensor([tunnel_x + length_left, up_end], dtype=torch.float32, device=self.world.device)
                positions = self.create_line_segment_between_pos(start_pos, end_pos)
                line_segments.extend(positions)


                # if length_right > length_left:
                #     start_pos = torch.tensor([tunnel_x + length_left, up_end], dtype=torch.float32, device=self.world.device)
                #     end_pos = torch.tensor([tunnel_x + length_left, up_end + 2], dtype=torch.float32, device=self.world.device)
                #     positions = create_line_segment_between_pos(start_pos, end_pos)
                #     line_segments.extend(positions)

                #     start_pos = torch.tensor([tunnel_x + length_right, down_end], dtype=torch.float32, device=self.world.device)
                #     end_pos = torch.tensor([tunnel_x + length_right, down_end + 3], dtype=torch.float32, device=self.world.device)
                #     positions = create_line_segment_between_pos(start_pos, end_pos)
                #     line_segments.extend(positions)
                # elif length_right < length_left:
                #     start_pos = torch.tensor([tunnel_x + length_left, up_end], dtype=torch.float32, device=self.world.device)
                #     end_pos = torch.tensor([tunnel_x + length_left, up_end - 3], dtype=torch.float32, device=self.world.device)
                #     positions = create_line_segment_between_pos(start_pos, end_pos)
                #     line_segments.extend(positions)

                #     start_pos = torch.tensor([tunnel_x + length_right, down_end], dtype=torch.float32, device=self.world.device)
                #     end_pos = torch.tensor([tunnel_x + length_right, down_end - 2], dtype=torch.float32, device=self.world.device)
                #     positions = create_line_segment_between_pos(start_pos, end_pos)
                #     line_segments.extend(positions)
                




                # Now, create obstacles equal to the number of positions
                total_positions = len(line_segments)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{current_idx + obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # Assign positions to obstacles
                for obs_idx in range(total_positions):
                    self.obstacles[obs_idx].set_pos(line_segments[obs_idx], batch_index = None)
                
                #     obs.set_pos(line_segments[idx], batch_index=env_index)

                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
                    


            elif self.env_type == "mixed_out_of_distribution":
                pass    
        
        # is_evaluation_mode == True
        else:
            if self.env_type == "narrow":
                current_idx = 0
                obs_list = self.precompute_obs_dict[self.evaluation_index]
                total_positions = len(obs_list)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # Assign positions to obstacles
                for obs_idx in range(total_positions):
                    for d in range(self.world.batch_dim):
                    
                        # noise = torch.randn(line_segments[0].shape, device=self.device) * 0.001 # Scale noise as needed
                    
                        self.obstacles[obs_idx].set_pos(obs_list[obs_idx], batch_index = d)



                
                # for idx, obs in enumerate(self.obstacles):
                #     obs.set_pos(line_segments[idx], batch_index=env_index)
                print("obs num:{}".format(len(self.obstacles)))

                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
                    


                
                
            elif self.env_type == "tunnel":
                

                current_idx = 0
                obs_list = self.precompute_obs_dict[self.evaluation_index]

                total_positions = len(obs_list)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                        renderable=True,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # total_invisible_positions = len(invisible_line_segments)
                # for obs_idx in range(total_invisible_positions):
                #     obs = Landmark(
                #         name=f"obs_{obs_idx}",
                #         collide=True,
                #         movable=False,
                #         shape=Sphere(radius=0.1),
                #         color=Color.BLUE,
                #         renderable=False,
                #     )
                    
                #     self.world.add_landmark(obs)
                #     for d in range(self.world.batch_dim):
                #         obs.set_pos(invisible_line_segments[obs_idx], batch_index = d)
                for obs_idx in range(total_positions):
                    for d in range(self.world.batch_dim):
                    
                        # noise = torch.randn(line_segments[0].shape, device=self.device) * 0.001 # Scale noise as needed
                    
                        self.obstacles[obs_idx].set_pos(obs_list[obs_idx], batch_index = d)

                # Create obstacle managers for each batch
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d,:].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)

                
                # center_list.append(torch.tensor([0.5, 1.5], dtype=torch.float32, device=self.device))
            elif self.env_type == "door_and_narrow":
                current_idx = 0
                obs_list = self.precompute_obs_dict[self.evaluation_index]

                # Now, create obstacles equal to the number of positions
                total_positions = len(obs_list)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # Assign positions to obstacles
                for obs_idx, polygon_positions in enumerate(obs_list):
                    # noisy_position = line_segments[i_value] + noise  # Add noise to the line segment
                    for d in range(self.world.batch_dim):
                        self.obstacles[obs_idx].set_pos(obs_list[obs_idx], batch_index = d)
                # for idx, obs in enumerate(self.obstacles):
                #     obs.set_pos(line_segments[idx], batch_index=env_index)
                print("obs num:{}".format(len(self.obstacles)))

                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
                
                
                
                
                
            elif self.env_type == "clutter":
                obs_list = self.precompute_obs_dict[self.evaluation_index]

                # Now, create obstacles equal to the number of positions
                total_positions = len(obs_list)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # Assign positions to obstacles
                for obs_idx, polygon_positions in enumerate(obs_list):
                    # noisy_position = line_segments[i_value] + noise  # Add noise to the line segment
                    for d in range(self.world.batch_dim):
                        self.obstacles[obs_idx].set_pos(obs_list[obs_idx], batch_index = d)
                # for idx, obs in enumerate(self.obstacles):
                #     obs.set_pos(line_segments[idx], batch_index=env_index)
                print("obs num:{}".format(len(self.obstacles)))

                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)
            elif self.env_type == "free":
                pass
            elif self.env_type == "mixed":
                pass
            elif self.env_type == "door":
                current_idx = 0
                obs_list = self.precompute_obs_dict[self.evaluation_index]

                total_positions = len(obs_list)
                self.obstacles = []  # Clear any existing obstacles
                for obs_idx in range(total_positions):
                    obs = Landmark(
                        name=f"obs_{obs_idx}",
                        collide=True,
                        movable=False,
                        shape=Sphere(radius=0.1),
                        color=Color.RED,
                        renderable=True,
                    )
                    self.obstacles.append(obs)
                    self.world.add_landmark(obs)

                # total_invisible_positions = len(invisible_line_segments)
                # for obs_idx in range(total_invisible_positions):
                #     obs = Landmark(
                #         name=f"obs_{obs_idx}",
                #         collide=True,
                #         movable=False,
                #         shape=Sphere(radius=0.1),
                #         color=Color.BLUE,
                #         renderable=False,
                #     )
                    
                #     self.world.add_landmark(obs)
                #     for d in range(self.world.batch_dim):
                #         obs.set_pos(invisible_line_segments[obs_idx], batch_index = d)
                for obs_idx in range(total_positions):
                    for d in range(self.world.batch_dim):
                    
                        # noise = torch.randn(line_segments[0].shape, device=self.device) * 0.001 # Scale noise as needed
                    
                        self.obstacles[obs_idx].set_pos(obs_list[obs_idx], batch_index = d)

                # Create obstacle managers for each batch
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d,:].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)

    def precompute_evaluation_scene(self, env_type):
        # Construct the file name based on the env_type
        file_name = f"precomputed_{env_type}_data.pkl"
        route_file_name = f"precomputed_{env_type}_route_data.pkl"
        # Check if the file exists
        if os.path.exists(file_name):
            if env_type == "clutter":
                pass
            else:
                if os.path.exists(route_file_name):
                    with open(route_file_name, 'rb') as file:
                        self.precomputed_route_point_list = pickle.load(file)
                        

                        # print(f"Data loaded from {route_file_name}")
            # Load the precomputed data from the file
            with open(file_name, 'rb') as file:
                map_location = {'cuda:0': 'cpu'}
                self.precompute_obs_dict = pickle.load(file)
                # print(f"Data loaded from {file_name}")
        else:
            if env_type == "clutter":


                for eva_index in range(self.evaluation_num):
                    self.precompute_obs_dict[eva_index] = []
                    num_polygons = np.random.randint(15, 18)
                    outer_existing_centers = []
                    # Generate polygons and collect positions
                    for poly_idx in range(num_polygons):
                        positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                            existing_centers=outer_existing_centers,
                            sphere_radius=0.1,
                            min_center_distance=2.0,
                            world_semidim=self.world_semidim,
                            device=self.device,
                            left_x = -self.world_semidim_x + 17,
                            right_x = self.world_semidim_x - 7,
                            bottom_y = -self.world_semidim_y + 4,
                            upper_y = self.world_semidim_y - 4
                        )
                        self.precompute_obs_dict[eva_index].extend(positions)

            elif env_type == "door_and_narrow":
                for eva_index in range(self.evaluation_num):
                    self.precompute_obs_dict[eva_index] = []
                    self.precomputed_route_point_list[eva_index] = []

                    current_idx = 0
                    num_polygons_for_clutter = 5
                    # num_polygons = 1
                    # num_polygons = 5
                    outer_existing_centers = []
                    polygon_dict = {}
                    
                    clutter_left_x = -15
                    clutter_right_x = -12
                    clutter_bottom_y = -5
                    clutter_upper_y = 4

                    door_x = -10
                    self.door_x = door_x
                    door_y = -1 + 2*random.random()
                    narrow_x_1 = -7
                    narrow_x_2 = -5
                    print("door_Y:{}".format(door_y))
                    self.precomputed_route_point_list[eva_index].append(torch.tensor([-15, 0], dtype=torch.float32, device=self.device))
                    self.precomputed_route_point_list[eva_index].append(torch.tensor([door_x + 1, door_y], dtype=torch.float32, device=self.device))
                    # self.route_point_list.append(torch.tensor([4, 0], dtype=torch.float32, device=self.device))
                    self.precomputed_route_point_list[eva_index].append(torch.tensor([narrow_x_1, 0], dtype=torch.float32, device=self.device))
                    if door_y > 0:
                        self.precomputed_route_point_list[eva_index].append(torch.tensor([narrow_x_2, -2], dtype=torch.float32, device=self.device))
                    else:
                        self.precomputed_route_point_list[eva_index].append(torch.tensor([narrow_x_2, 2], dtype=torch.float32, device=self.device))

                    # Generate polygons and collect positions
                    for poly_idx in range(int(num_polygons_for_clutter/2)):
                        positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                            existing_centers=outer_existing_centers,
                            sphere_radius=0.1,
                            min_center_distance=2.0,
                            world_semidim=self.world_semidim,
                            device=self.device,
                            left_x = clutter_left_x,
                            right_x = clutter_right_x,
                            bottom_y = 1.0,
                            upper_y = clutter_upper_y,
                        )
                        # sphere_start_idx = len(line_segments)
                        self.precompute_obs_dict[eva_index].extend(positions)
                    # sphere_end_idx = len(line_segments) 
                    # polygon_list.append(positions)
                    # polygon_dict[poly_idx] = list(range(sphere_start_idx, sphere_end_idx))
                    for poly_idx in range(int(num_polygons_for_clutter/2)):
                        positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                            existing_centers=outer_existing_centers,
                            sphere_radius=0.1,
                            min_center_distance=2.0,
                            world_semidim=self.world_semidim,
                            device=self.device,
                            left_x = clutter_left_x,
                            right_x = clutter_right_x,
                            bottom_y = clutter_bottom_y,
                            upper_y = -1.0,
                        )
                        # sphere_start_idx = len(line_segments)
                        self.precompute_obs_dict[eva_index].extend(positions)
                    
                    # for poly_idx in range(num_polygons_for_clutter):
                    #     positions, outer_existing_centers, _ = self.create_small_fixed_obstacle(
                    #         existing_centers=outer_existing_centers,
                    #         sphere_radius=0.1,
                    #         min_center_distance=2.0,
                    #         world_semidim=self.world_semidim,
                    #         device=self.device,
                    #         left_x = clutter_left_x,
                    #         right_x = clutter_right_x,
                    #         bottom_y = clutter_bottom_y,
                    #         upper_y = clutter_upper_y,
                    #     )
                    #     # sphere_start_idx = len(line_segments)
                    #     self.precompute_obs_dict[eva_index].extend(positions)
                        # sphere_end_idx = len(line_segments) 
                        # polygon_list.append(positions)
                        # polygon_dict[poly_idx] = list(range(sphere_start_idx, sphere_end_idx))

                    
                    ##############
                    #add obs for door
                    ##############
                    down_start = door_y -0.75 -0.3 + random.random()*0.6
                    up_start    = door_y + 0.75 -0.3 + random.random()*0.6
                    down_end = door_y -0.6 -0.1 + random.random()*0.2
                    up_end = door_y + 0.6 - 0.1 + random.random()*0.2
                    if random.random() < 0.4:
                        length_left = 0.1
                        length_right = 0.1
                    else:
                        length_right = random.random()*0.5
                        length_left = random.random()*0.4
                    
                        
                    
                    start_pos = torch.tensor([door_x, down_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([door_x, door_y -4.5], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    start_pos = torch.tensor([door_x, up_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([door_x, door_y + 4.0], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    start_pos = torch.tensor([door_x, down_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([door_x + length_right, down_end], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    start_pos = torch.tensor([door_x, up_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([door_x + length_left, up_end], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)

                    ##############
                    #add obs for narrow
                    ##############
                    def add_noise(tensor, noise_level=0.1):
                        noise = torch.randn_like(tensor) * noise_level  # Gaussian noise with mean 0 and std dev = noise_level
                        return tensor + noise
                    center_noise_level = 0.01  # Adjust as needed for center position noise
                    polygon_radius_noise_level = 0.01  # Adju
                    center_list = []
                    center_list.append(add_noise(torch.tensor([narrow_x_1, 2.0], dtype=torch.float32, device=self.device), center_noise_level))
                    center_list.append(add_noise(torch.tensor([narrow_x_1, -2.0], dtype=torch.float32, device=self.device), center_noise_level))
                    center_list.append(add_noise(torch.tensor([narrow_x_2, 0], dtype=torch.float32, device=self.device), center_noise_level))

                    polygon_dict = {}
                    for poly_idx, center in enumerate(center_list):
                        # polygon_radius = 0
                        polygon_radius = 0.8 + np.random.uniform(-polygon_radius_noise_level, polygon_radius_noise_level)
                        positions =  self.create_polygon_with_center(center, num_vertices = 8, polygon_radius=polygon_radius,sphere_radius=0.1, max_spheres_per_polygon=50, 
                world_semidim=self.world_semidim, 
                device=self.device)
                        self.obstacle_center_list.append(center)
                        self.obstacle_radius_list.append(polygon_radius)
                        sphere_start_idx = len(self.precompute_obs_dict[eva_index])
                        self.precompute_obs_dict[eva_index].extend(positions)
                    
                    
                with open(route_file_name, 'wb') as file:
                    pickle.dump(self.precomputed_route_point_list, file)
                    print(f"Data saved to {route_file_name}")    
                    
                    total_positions = len(self.precompute_obs_dict[eva_index])

            elif env_type == "tunnel":
                for eva_index in range(self.evaluation_num):
                    self.precompute_obs_dict[eva_index] = []
                    self.precomputed_route_point_list[eva_index] = []
                    down_start = -1.2 -0.6 + random.random()*1.2
                    up_start    = 1.2 -0.6 + random.random()*1.2
                    down_end = -1.2 -0.6 + random.random()*1.2
                    up_end = 1.2 - 0.6 + random.random()*1.2
                    if random.random() > 0.5:
                        up_end = up_start
                        length_left = 4 - 1.5 + random.random()*3
                        length_right = length_left + 2.2 - 0.4 + random.random()*0.8
                        end_target_1 = (torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.device) + torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.device)) / 2
                        end_target_2 = end_target_1.clone()
                        end_target_2[1] += (6.5 + random.random())
                    else:
                        down_end = down_start
                        length_right = 4 - 1.5 + random.random()*3
                        length_left = length_right + 2.2 - 0.4 + random.random()*0.8

                        end_target_1 = (torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.device) + torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.device)) / 2
                        end_target_2 = end_target_1.clone()
                        end_target_2[1] -= (6.5 + random.random())
                    

                    self.precomputed_route_point_list[eva_index].append(end_target_1)

                
                    self.precomputed_route_point_list[eva_index].append(end_target_2)
                    #下方入口到下边界
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5, -4.5], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    
                    #上方入口到上边界
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5, 4.0], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    
                    #下方入口向tunnel内延伸
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)

                    #上方入口向tunnel内延伸
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)


                    if length_right > length_left:
                        start_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.device)
                        end_pos = torch.tensor([-0.5 + length_left + 0.5, up_end + 4.5], dtype=torch.float32, device=self.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        self.precompute_obs_dict[eva_index].extend(positions)

                        start_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.device)
                        end_pos = torch.tensor([-0.5 + length_right - 0.5, down_end + 6.5], dtype=torch.float32, device=self.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        self.precompute_obs_dict[eva_index].extend(positions)
                    elif length_right < length_left:
                        start_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.device)
                        end_pos = torch.tensor([-0.5 + length_left - 0.5, up_end - 6.5], dtype=torch.float32, device=self.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        self.precompute_obs_dict[eva_index].extend(positions)

                        start_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.device)
                        end_pos = torch.tensor([-0.5 + length_right + 0.5, down_end - 4.5], dtype=torch.float32, device=self.device)
                        positions = self.create_line_segment_between_pos(start_pos, end_pos)
                        self.precompute_obs_dict[eva_index].extend(positions)
                    # start_pos = torch.tensor([2.7, -2.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([2.7, 2.8], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
                    # start_pos = torch.tensor([4, -3.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([4, 3.5], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
                    # start_pos = torch.tensor([-2.5, -2.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.world.device)
                    # invisible_positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    # invisible_line_segments.extend(invisible_positions)

                    # start_pos = torch.tensor([-2.5, 2.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.world.device)
                    # invisible_positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    # invisible_line_segments.extend(invisible_positions)
                with open(route_file_name, 'wb') as file:
                    pickle.dump(self.precomputed_route_point_list, file)
                    print(f"Data saved to {route_file_name}")    
            
            elif env_type == "door":
                for eva_index in range(self.evaluation_num):
                    self.precompute_obs_dict[eva_index] = []
                    self.precomputed_route_point_list[eva_index] = []
                    down_start = -0.75 -0.3 + random.random()*0.6
                    up_start    = 0.75 -0.3 + random.random()*0.6
                    down_end = -0.6 -0.1 + random.random()*0.2
                    up_end = 0.6 - 0.1 + random.random()*0.2
                    if random.random() < 0.4:
                        length_left = 0.1
                        length_right = 0.1
                    else:
                        length_right = random.random()*2
                        length_left = random.random()*2
                    
                    self.door_x = -0.5   
                    
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5, -4.5], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5, 4.0], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    start_pos = torch.tensor([-0.5, down_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5 + length_right, down_end], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)
                    start_pos = torch.tensor([-0.5, up_start], dtype=torch.float32, device=self.device)
                    end_pos = torch.tensor([-0.5 + length_left, up_end], dtype=torch.float32, device=self.device)
                    positions = self.create_line_segment_between_pos(start_pos, end_pos)
                    self.precompute_obs_dict[eva_index].extend(positions)

                # with open(route_file_name, 'wb') as file:
                #     pickle.dump(self.precomputed_route_point_list, file)
                #     print(f"Data saved to {route_file_name}")    
                   
                    # start_pos = torch.tensor([2.7, -2.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([2.7, 2.8], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
                    # start_pos = torch.tensor([4, -3.5], dtype=torch.float32, device=self.world.device)
                    # end_pos = torch.tensor([4, 3.5], dtype=torch.float32, device=self.world.device)
                    # positions = create_line_segment_between_pos(start_pos, end_pos)
                    # line_segments.extend(positions)
            elif env_type == "narrow":
                def add_noise(tensor, noise_level=0.1):
                    noise = torch.randn_like(tensor) * noise_level  # Gaussian noise with mean 0 and std dev = noise_level
                    return tensor + noise
                for eva_index in range(self.evaluation_num):
                    self.precompute_obs_dict[eva_index] = []
                    self.precomputed_route_point_list[eva_index] = []
                
                    center_noise_level = 0.3  # Adjust as needed for center position noise
                    polygon_radius_noise_level = 0.1  # Adju
                    center_list = []
                    center_list.append(add_noise(torch.tensor([-2.5, 1.8], dtype=torch.float32, device=self.device), center_noise_level))
                    center_list.append(add_noise(torch.tensor([-2.5, -1.8], dtype=torch.float32, device=self.device), center_noise_level))
                    center_list.append(add_noise(torch.tensor([1.2, 0], dtype=torch.float32, device=self.device), center_noise_level))

                    polygon_dict = {}
                    for poly_idx, center in enumerate(center_list):
                        # polygon_radius = 0
                        polygon_radius = 0.8 + np.random.uniform(-polygon_radius_noise_level, polygon_radius_noise_level)
                        positions =  self.create_polygon_with_center(center, num_vertices = 8, polygon_radius=polygon_radius,sphere_radius=0.1, max_spheres_per_polygon=50, 
                world_semidim=self.world_semidim, 
                device=self.device)
                        self.obstacle_center_list.append(center)
                        self.obstacle_radius_list.append(polygon_radius)
                        sphere_start_idx = len(self.precompute_obs_dict[eva_index])
                        self.precompute_obs_dict[eva_index].extend(positions)
                        sphere_end_idx = len(self.precompute_obs_dict[eva_index]) 
                        # polygon_list.append(positions)
                        polygon_dict[poly_idx] = list(range(sphere_start_idx, sphere_end_idx))
                    
                    third_obs_center = center_list[2]
                    first_route_center = copy.deepcopy(third_obs_center)
                    first_route_center[0] -= 2.5
                    first_route_center[1] = 0
                    self.precomputed_route_point_list[eva_index].append(add_noise(first_route_center, 0.1))
                    if random.random() < 0.5:
                        third_obs_center[1] = -2
                        self.precomputed_route_point_list[eva_index].append(third_obs_center)
                    else:
                        third_obs_center[1] = 2
                        self.precomputed_route_point_list[eva_index].append(third_obs_center)

                    # self.precomputed_route_point_list[eva_index].append(add_noise(torch.tensor([-1, 0], dtype=torch.float32, device=self.device), 0.1))
                    # if random.random() < 0.5:
                    #     self.precomputed_route_point_list[eva_index].append(add_noise(torch.tensor([1.2, -2], dtype=torch.float32, device=self.device), 0.1))
                    # else:
                    #     self.precomputed_route_point_list[eva_index].append(add_noise(torch.tensor([1.2, 2], dtype=torch.float32, device=self.device), 0.1))

                with open(route_file_name, 'wb') as file:
                    pickle.dump(self.precomputed_route_point_list, file)
                    print(f"Data saved to {route_file_name}")    

                    
            # Save the computed data to a file
            with open(file_name, 'wb') as file:
                pickle.dump(self.precompute_obs_dict, file)
                print(f"Data saved to {file_name}")

    def reset_world_at(self, env_index: int = None):
        # print("reset_world_at {}".format(env_index))
        self.update_formation_assignment_time[env_index] = time.time()
        # ScenarioUtils.spawn_entities_randomly(
        #     self.world.agents,
        #     self.world,
        #     env_index,
        #     self.min_distance_between_entities,
        #     (-self.world_semidim+1, self.world_semidim),
        #     (-self.world_semidim+4, self.world_semidim),
        # )

        # ScenarioUtils.spawn_entities_randomly(
        #     self.world.landmarks,
        #     self.world,
        #     env_index,
        #     0.1,
        #     (-self.world_semidim, self.world_semidim),
        #     (-self.world_semidim, self.world_semidim),
        # )
        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        noise_scale = self.init_positions_noise_level
        if self.env_type == "mixed_in_distribution" or self.env_type == "door_and_narrow":
            self.leader_robot.set_pos(
                        torch.tensor(
                            [
                                -16.0, 
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
                                -16.0, 
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
                                -16.0, 
                                -0.0,
                            ],
                            device=self.world.device,
                        ) + noise,
                        batch_index=env_index,
                    )
                elif i == 1:
                    agent.set_pos(
                        torch.tensor(
                            [
                                -17.2, 
                                0.6,
                            ],
                            device=self.world.device,
                        ) + noise,
                        batch_index=env_index,
                    )
                elif i == 2:
                    #-3.6464,  0.3536
                    agent.set_pos(    
                        torch.tensor(
                            [
                                -17.2,  
                                -0.6,
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
                                -18.4,
                                1.2,
                            ],
                            device=self.world.device,
                        ) + noise,
                        batch_index=env_index,
                    )
                elif i == 4:
                    #-3.6464, -0.3536
                    agent.set_pos(    
                        torch.tensor(
                            [
                                -18.4,
                                -1.2,
                            ],
                            device=self.world.device,
                        ) + noise,
                        batch_index=env_index,
                    )
        else:
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
                                -4.0, 
                                0.0,
                            ],
                            device=self.world.device,
                        ) + noise,
                        batch_index=env_index,
                    )
                elif i == 1:
                    agent.set_pos(
                        torch.tensor(
                            [
                                -5.2, 
                                0.6,
                            ],
                            device=self.world.device,
                        ) + noise,
                        batch_index=env_index,
                    )
                elif i == 2:
                    #-3.6464,  0.3536
                    agent.set_pos(    
                        torch.tensor(
                            [
                                -5.2,  
                                -0.6,
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
                                -6.4,
                                1.2,
                            ],
                            device=self.world.device,
                        ) + noise,
                        batch_index=env_index,
                    )
                elif i == 4:
                    #-3.6464, -0.3536
                    agent.set_pos(    
                        torch.tensor(
                            [
                                -6.4,
                                -1.2,
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

        
        self.spawn_obstacles(self.obstacle_pattern, env_index)
        self.t = 0

    def find_clear_direction(self, current_pos, current_direction, obstacle_manager, max_scan_angle, scan_step, dim, angle_toward_route=None):
        """
        Find a clear direction to move in, starting from the current_direction.
        Scans left and right up to max_scan_angle, in increments of scan_step.
        """
        # First, check the forward direction
        if self.is_direction_clear(current_pos, current_direction, obstacle_manager, dim):
            # print("direction clear")
            return current_direction

        # Initialize variables
        angles_to_check = []
        if angle_toward_route == 0 or angle_toward_route == None:
            for delta_angle in np.arange(scan_step, max_scan_angle + scan_step, scan_step):
                angles_to_check.append(current_direction + delta_angle)  # Left side
                angles_to_check.append(current_direction - delta_angle)  # Right side
        elif angle_toward_route == 1:
            for delta_angle in np.arange(scan_step, max_scan_angle + scan_step, scan_step):
                angles_to_check.append(current_direction + delta_angle)  # Left side
        elif angle_toward_route == -1:
            for delta_angle in np.arange(scan_step, max_scan_angle + scan_step, scan_step):
                angles_to_check.append(current_direction - delta_angle)  # Right side


        # Check each angle
        for angle in angles_to_check:
            if self.is_direction_clear(current_pos, angle, obstacle_manager, dim):

                return angle

        # If no clear direction found, return current_direction
        return current_direction

    def is_line_clear(self, current_pos, target_pos, obstacle_manager):
        """
        Checks if the formation can move in the given direction without colliding with obstacles.
        """
        scan_distance = torch.norm(target_pos - current_pos)
        # print("scan distance:{}".format(scan_distance))
        # Number of points across the formation width and along the path
        num_checks_distance = int(scan_distance / 0.15)  # Increase for finer resolution

        vector_to_target = target_pos - current_pos
        vector_to_target = vector_to_target / torch.norm(vector_to_target)


        # Positions to check across the formation width
        for i in range(1, num_checks_distance + 1):
            fraction = i / num_checks_distance
            point = current_pos + fraction * scan_distance * vector_to_target
            # Check for obstacles at this point
            obstacles = obstacle_manager.get_near_obstacles(point, 0.1)
            if obstacles.numel() > 0:
                return False  # Path is not clear
            # Also check bounds
            if not self.is_within_bounds(point):
                return False
        return True

    def is_direction_clear(self, current_pos, direction, obstacle_manager, dim):
        """
        Checks if the formation can move in the given direction without colliding with obstacles.
        """
        scan_distance = self.scan_distance
        formation_width = self.calculate_formation_width()

        # Number of points across the formation width and along the path
        num_checks_width = 2
        num_checks_distance = 6  # Increase for finer resolution

        if self.env_type == "narrow":
            half_width = formation_width / 3
        else:
            half_width = formation_width / 4

        # Create vectors
        direction_vector = torch.tensor([torch.cos(direction), torch.sin(direction)], device=self.device)
        perp_direction = torch.tensor([-torch.sin(direction), torch.cos(direction)], device=self.device)

        # Positions to check across the formation width
        offsets = torch.linspace(-half_width, half_width, num_checks_width, device=self.device)
        for offset in offsets:
            lateral_offset = perp_direction * offset
            # Sample along the path
            for i in range(1, num_checks_distance + 1):
                fraction = i / num_checks_distance
                point = current_pos + lateral_offset + fraction * scan_distance * direction_vector
                # Check for obstacles at this point
                obstacles = obstacle_manager.get_near_obstacles(point, 0.15)
                if obstacles.numel() > 0:
                    return False  # Path is not clear
                # Also check bounds
                if not self.is_within_bounds(point):
                    return False
        return True

    def avoid_boundaries(self, tentative_next_pos, current_direction, current_pos, max_steering_angle):
        boundary_threshold_distance = 2.5
        repulsion_strength = 0.5

        # Compute distances to boundaries
        dx_right = self.world_semidim_x - tentative_next_pos[0]
        dx_left = tentative_next_pos[0] + self.world_semidim_x
        dy_top = self.world_semidim_y - tentative_next_pos[1]
        dy_bottom = tentative_next_pos[1] + self.world_semidim_y

        # Initialize repulsion vector
        repulsion_vector = torch.zeros(2, device=self.device)

        if dx_right < boundary_threshold_distance:
            repulsion_vector[0] -= (boundary_threshold_distance - dx_right) * repulsion_strength
        if dx_left < boundary_threshold_distance:
            repulsion_vector[0] += (boundary_threshold_distance - dx_left) * repulsion_strength
        if dy_top < boundary_threshold_distance:
            repulsion_vector[1] -= (boundary_threshold_distance - dy_top) * repulsion_strength
        if dy_bottom < boundary_threshold_distance:
            repulsion_vector[1] += (boundary_threshold_distance - dy_bottom) * repulsion_strength

        # Adjust the direction
        new_direction_vector = torch.tensor([torch.cos(current_direction), torch.sin(current_direction)], device=self.device) + repulsion_vector
        new_direction_vector = new_direction_vector / torch.norm(new_direction_vector)

        # Compute the change in angle
        delta_angle = torch.atan2(new_direction_vector[1], new_direction_vector[0]) - current_direction
        delta_angle = torch.atan2(torch.sin(delta_angle), torch.cos(delta_angle))  # Normalize

        # Limit the change in angle
        delta_angle = torch.clamp(delta_angle, -max_steering_angle, max_steering_angle)
        current_direction += delta_angle

        # Recompute the step direction and tentative next position
        step_direction = torch.tensor([torch.cos(current_direction), torch.sin(current_direction)], device=self.device)
        tentative_next_pos = current_pos + step_direction * self.velocity_limit

        return tentative_next_pos, current_direction

    def calculate_formation_width(self):
        # Calculate the maximum width of the formation based on agent positions
        # For simplicity, you can set it to a fixed value or compute based on the formation type
        # Here, we assume a fixed formation width
        return 0.5  # Adjust as necessary

    def is_within_bounds(self, pos):
        x, y = pos[0], pos[1]
        return (-self.world_semidim_x < x < self.world_semidim_x) and (-self.world_semidim_y < y < self.world_semidim_y)    
    
    def transform_local_to_global(self, optimized_target_pos, leader_robot_pos, leader_robot_rot, device):
        """
        Transforms positions from the local frame to the global frame.

        Args:
            optimized_target_pos (torch.Tensor): Positions in the local frame. Shape: [batch_dim, 2] or [2]
            leader_robot_pos (torch.Tensor): Leader robot's position in the global frame. Shape: [batch_dim, 2] or [2]
            leader_robot_rot (torch.Tensor): Leader robot's rotation angle (in radians). Shape: [batch_dim] or scalar
            device (torch.device): Device to perform computations on.

        Returns:
            torch.Tensor: Positions in the global frame. Shape: [batch_dim, 2] or [2]
        """
        # Ensure inputs are on the correct device
        optimized_target_pos = optimized_target_pos.to(device)
        leader_robot_pos = leader_robot_pos.to(device)
        leader_robot_rot = leader_robot_rot.to(device)
        # print("optimized target pos shape:{}".format(optimized_target_pos.shape))
        optimized_target_position = optimized_target_pos[ :, :2]
        # print("optimized target pos shape after :{}".format(optimized_target_position.shape))

        # Step 1: Compute Inverse Rotation Matrices
        cos_theta = torch.cos(leader_robot_rot).unsqueeze(-1)  # Shape: [batch_dim, 1] or [1, 1]
        sin_theta = torch.sin(leader_robot_rot).unsqueeze(-1)  # Shape: [batch_dim, 1] or [1, 1]

        # Original rotation matrices:
        # [cos(theta)  sin(theta)]
        # [-sin(theta) cos(theta)]
        rotation_matrices = torch.cat([cos_theta, sin_theta, -sin_theta, cos_theta], dim=1).reshape(-1, 2, 2)  # Shape: [batch_dim, 2, 2] or [1, 2, 2]
        rotation_matrices_inv = rotation_matrices.transpose(1, 2)  # Inverse rotation for orthonormal matrices

        # Step 2: Apply Inverse Rotation
        # Use torch.matmul instead of torch.bmm for flexibility
        # optimized_target_pos: [batch_dim, 2] or [2]
        # optimized_target_pos.unsqueeze(-1): [batch_dim, 2, 1] or [2, 1]
        translated_pos_global = torch.matmul(rotation_matrices_inv, optimized_target_position.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_dim, 2] or [2]

        # Step 3: Apply Inverse Translation
        # agent_pos_global = translated_pos_global + leader_robot_pos
        agent_position_global = translated_pos_global + leader_robot_pos  # Shape: [batch_dim, 2] or [2]

        optimized_target_rot = optimized_target_pos[:, 2]
        # print("optimized_target_rot shape:{}".format(optimized_target_rot.shape))
        # print("leader_robot_rot shape:{}".format(leader_robot_rot.shape))
        agent_rot_global = optimized_target_rot + leader_robot_rot.squeeze(dim=1)
        # print("agent_rot_global shape:{}".format(agent_rot_global.shape))
        # print("agent_position_global shape:{}".format(agent_position_global.shape))
        agent_pose_global = torch.cat((agent_position_global, agent_rot_global.unsqueeze(dim=1)), dim=1)
        # print("agent_pose_global shape:{}".format(agent_pose_global.shape))
        return agent_pose_global
    def process_action(self, agent: Agent):
        self.velocity_limit = 0.03  # Adjusted speed for smoother movement
        is_first = agent == self.world.agents[0]
        if is_first:
            current_positions = torch.stack([agent.state.pos for agent in self.world.agents])  # [num_agents, batch_dim, 2]
            current_positions = current_positions.permute(1, 0, 2)  # [batch_dim, num_agents, 2]
            self.agent_history.update(current_positions)
            formation_movement = "random"
            if self.env_type == "clutter":
                formation_movement = "random"
            elif self.env_type == "narrow":
                formation_movement = "route"
            elif self.env_type == "tunnel":
                formation_movement = "route"
            elif self.env_type == "door":
                formation_movement = "horizental"
            elif self.env_type == "mixed_in_distribution":
                formation_movement = "route"
            elif self.env_type == "door_and_narrow":
                formation_movement = "route"

            if formation_movement == "random":
                for dim in range(self.world.batch_dim):
                    # Parameters
                    max_scan_angle = torch.pi / 1.2  # Maximum scanning angle (90 degrees)
                    scan_step = torch.pi / 18      # Scanning step (10 degrees)
                    scan_distance = 2.0    # Distance to check ahead
                    self.scan_distance = scan_distance
                    self.formation_width = self.calculate_formation_width()

                    current_pos = self.formation_center.state.pos[dim, :].squeeze()
                    current_direction = self.formation_center.state.rot[dim]

                    obstacle_manager = self.obstacle_manager_list[dim]

                    # Find a clear direction
                    new_direction = self.find_clear_direction(
                        current_pos,
                        current_direction,
                        obstacle_manager,
                        max_scan_angle,
                        scan_step,
                        dim
                    )
                    # new_direction = current_direction
                    # Smoothly adjust the current direction
                    max_steering_angle = 0.15  # Maximum change in direction per step
                    delta_angle = new_direction - current_direction
                    delta_angle = torch.atan2(torch.sin(delta_angle), torch.cos(delta_angle))  # Normalize
                    delta_angle = torch.clamp(delta_angle, -max_steering_angle, max_steering_angle)
                    current_direction += delta_angle

                    # Update the step direction and tentative next position
                    step_direction = torch.tensor([torch.cos(current_direction), torch.sin(current_direction)], device=self.device)
                    tentative_next_pos = current_pos + step_direction * self.velocity_limit

                    # Boundary avoidance
                    tentative_next_pos, current_direction = self.avoid_boundaries(
                        tentative_next_pos,
                        current_direction,
                        current_pos,
                        max_steering_angle
                    )

                    # Update positions and rotations
                    self.leader_robot.set_pos(tentative_next_pos, batch_index=dim)
                    self.leader_robot.set_rot(current_direction, batch_index=dim)
                    self.leader_agent.set_pos(tentative_next_pos, batch_index=dim)
                    self.leader_agent.set_rot(current_direction, batch_index=dim)
                    self.formation_center.set_pos(tentative_next_pos, batch_index=dim)
                    self.formation_center.set_rot(current_direction, batch_index=dim)
                    self.formation_center_pos[dim, :2] = tentative_next_pos
                    self.formation_center_pos[dim, 2] = current_direction
                  

            elif formation_movement == "route":
                for dim in range(self.world.batch_dim):
                    max_scan_angle = torch.pi / 1.2  # Maximum scanning angle (90 degrees)
                    scan_step = torch.pi / 18      # Scanning step (10 degrees)
                    
                    
                    self.angle_toward_route = 0
                    scan_distance = 2.0    # Distance to check ahead

                    self.scan_distance = scan_distance
                    self.formation_width = self.calculate_formation_width()
                    
                    current_pos = self.formation_center.state.pos[dim, :].squeeze()
                    current_direction = self.formation_center.state.rot[dim]

                    obstacle_manager = self.obstacle_manager_list[dim]


                    # print("cur dic:{}".format(current_direction))
                    # Find a clear direction (same as current logic)
                    if self.env_type == "narrow" or self.env_type == "mixed_in_distribution" or self.env_type == "door_and_narrow":
                        new_direction = self.find_clear_direction(
                            current_pos,
                            current_direction,
                            obstacle_manager,
                            max_scan_angle,
                            scan_step,
                            dim,
                            self.angle_toward_route,
                        )
                        max_steering_angle = 0.15  # Maximum change in direction per step
                        delta_angle = new_direction - current_direction
                        delta_angle = torch.atan2(torch.sin(delta_angle), torch.cos(delta_angle))  # Normalize
                        delta_angle = torch.clamp(delta_angle, -max_steering_angle, max_steering_angle)
                        current_direction += delta_angle
                    # new_direction = current_direction
                    # Smoothly adjust the current direction
                    max_steering_angle = 0.15  # Maximum change in direction per step
                    

                    # Retrieve the route points
                    if self.is_evaluation_mode == False:
                        route_points = self.route_point_list
                    else:
                        route_points = self.precomputed_route_point_list[self.evaluation_index]

                    # Track visited route points to avoid revisiting
                    if not hasattr(self, "visited_route_points"):
                        self.visited_route_points = []
                    # print("visited:{}".format(self.visited_route_points))
                    # Remove already visited points from the list
                    available_route_points = [pt for pt in route_points if pt.tolist() not in self.visited_route_points]
                    # print("available route:{}".format(available_route_points))
                    # Find the nearest route point
                    if available_route_points:
                        
                        # print("current pos device:{}".format(current_pos.device))
                        distances = [torch.norm(current_pos - pt) for pt in available_route_points]
                        self.nearest_point = available_route_points[torch.argmin(torch.tensor(distances))]
                        # print("dist:{}".format(torch.norm(current_pos - nearest_point)))
                        if torch.norm(current_pos - self.nearest_point) < 0.7:
                            # Add the nearest point to the visited list
                            self.visited_route_points.append(self.nearest_point.tolist())
                        # print("nearest pt:{}".format(self.nearest_point))
                        # Compute the direction to the nearest route point
                        direction_to_point = self.nearest_point - current_pos
                        step_direction = direction_to_point / torch.norm(direction_to_point)  # Normalize
                        # step_direction = step_direction * self.velocity_limit  # Scale to the velocity limit
                    else:
                        # If no available points, continue based on the current direction
                        step_direction = torch.tensor([torch.cos(current_direction), torch.sin(current_direction)], device=self.device)
                        # step_direction = step_direction * self.velocity_limit  # Scale to the velocity limit

                    # Compute the tentative next position
                    tentative_next_pos = current_pos + step_direction * self.velocity_limit
                    if self.env_type == "tunnel" or self.env_type  == "narrow" or self.env_type == "mixed_in_distribution" or self.env_type == "door_and_narrow":
                        delta_angle = torch.atan2(step_direction[1], step_direction[0]) - current_direction
                        if delta_angle > 0:
                            self.angle_toward_route = 1
                        elif delta_angle < 0:
                            self.angle_toward_route = -1
                        else:
                            self.angle_toward_route = 0
                        delta_angle = torch.atan2(torch.sin(delta_angle), torch.cos(delta_angle))  # Normalize

        # Limit the change in angle
                        delta_angle = torch.clamp(delta_angle, -max_steering_angle, max_steering_angle)
                        current_direction += delta_angle


                    # Boundary avoidance
                    tentative_next_pos, current_direction = self.avoid_boundaries(
                        tentative_next_pos,
                        current_direction,
                        current_pos,
                        max_steering_angle
                    )

                    # Update positions and rotations
                    self.leader_robot.set_pos(tentative_next_pos, batch_index=dim)
                    self.leader_robot.set_rot(current_direction, batch_index=dim)
                    self.leader_agent.set_pos(tentative_next_pos, batch_index=dim)
                    self.leader_agent.set_rot(current_direction, batch_index=dim)
                    self.formation_center.set_pos(tentative_next_pos, batch_index=dim)
                    self.formation_center.set_rot(current_direction, batch_index=dim)
                    self.formation_center_pos[dim, :2] = tentative_next_pos
                    self.formation_center_pos[dim, 2] = current_direction
            elif formation_movement == "horizental":
                ###move from left to right, test formation's ability to cross through tunnel
                
                for dim in range(self.world.batch_dim):
                    t = self.t / 30
                    if self.t < 1:
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
                        self.last_leader_robot[0] = ((self.t-1)-1)/30*0.8 - 4
                        self.last_leader_robot[1] = 0
                        self.leader_robot.set_pos(
                            torch.tensor(
                                [
                                    (self.t-1)/30*0.8 - 4,
                                    0,
                                ],
                                device=self.world.device,
                            ),
                            batch_index=dim,
                        )
                        self.formation_center.set_pos(
                            torch.tensor(
                                [
                                    (self.t-1)/30*0.8 - 4,
                                    0,
                                ],
                                device=self.world.device,
                            ),
                            batch_index=dim,
                        )
                        self.formation_center_pos[dim, 0] = ((self.t-1)/30*0.8 - 4)
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
                    self.leader_agent.set_pos(
                        torch.tensor(
                            [
                                (self.t-1)/30*0.8 - 4,
                                    0,
                            ],
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )
                    self.leader_agent.set_rot(
                        torch.tensor(
                            0,
                            device=self.world.device,
                        ),
                        batch_index=dim,
                    )


                
            # self.agent_target_pos_global = self.transform_local_to_global(agent.action.u, self.leader_robot.state.pos, self.leader_robot.state.rot, self.device)
            # print("agent_target_pos_global:{}".format(self.agent_target_pos_global))
            # self.agent_target_pos_global = self.agent_target_pos_global.squeeze(dim=0)
        # angles = [-135/180.0*math.pi, 135/180.0*math.pi, -135/180.0*math.pi,  135/180.0*math.pi]
        # dists = [-0.8, -0.8, -1.6, -1.6]
        angles = [0.0, -26/180.0*math.pi, 26/180.0*math.pi, -26/180.0*math.pi,  26/180.0*math.pi]

        dists = [0.0, -1.34, -1.34, -2.68, -2.68]
        for i, world_agent in enumerate(self.world.agents):
            if agent == world_agent:

                # formation_type = "line"
                # formation_type = "rectangle"
                
                # if self.t < 200: 
                #     formation_type = "rectangle"
                # else:
                #     formation_type = "ren_shape"
                # print("agent actionu shape:{}".format(agent.action.u.shape))
                # if self.working_mode == "RL":
                    # agent_target_pose_global = self.transform_local_to_global(agent.action.u, self.leader_robot.state.pos, self.leader_robot.state.rot, self.device)
                if self.current_formation_type == "ren_shape":
                    #大雁人字形
                    # angles = [-135/180.0*math.pi, 135/180.0*math.pi, -135/180.0*math.pi,  135/180.0*math.pi]
                    angle = angles[i]  # Get the angle for the current agent
                    dist = dists[i]    # Get the distance for the current agent

                    # Calculate the angle for the current formation
                    # print("formatION-centttter_pos:{}".format(self.formation_center_pos[:, 2]))
                    formation_angle = self.formation_center_pos[:, 2] + angle  # Shape [10, 1] + scalar
                    # formation_angle = angle
                    # print("formation_gnale:{}".format(formation_angle))
                    # formation_angle = angle - self.formation_center_pos[:, 2]
                    # print("agent u:{}".format(agent.action.u))
                    # Update the goals using torch.cos and torch.sin
                    if self.working_mode == "RL": 
                        agent_target_pose_global = self.transform_local_to_global(agent.action.u, self.leader_robot.state.pos, self.leader_robot.state.rot, self.device)

                        if i == 0:
                            self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + torch.cos(formation_angle) * dist
                            self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + torch.sin(formation_angle) * dist
                        else:
                            self.formation_goals[i][:, 0] = agent_target_pose_global[:,0]
                            self.formation_goals[i][:, 1] = agent_target_pose_global[:,1]
                            self.formation_goals[i][:, 2] = agent_target_pose_global[:,2]

                    if self.working_mode == "imitation":
                        self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + torch.cos(formation_angle) * dist
                        self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + torch.sin(formation_angle) * dist
                        self.formation_goals[i][:, 2] = self.formation_center_pos[:, 2] 
                        if self.obstacle_pattern == 2:
                            self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + torch.cos(formation_angle) * (dist/2)
                            self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + torch.sin(formation_angle) * (dist/2)


                        threshold = 0.3
                        position_error_x = self.formation_goals[i][:, 0] - world_agent.state.pos[:, 0]
                        position_error_y = self.formation_goals[i][:, 1] - world_agent.state.pos[:, 1]

                    # Check if both position errors are smaller than the threshold for each agent in the batch
                        if self.env_type == "door" or self.env_type == "door_and_narrow":
                            threshold = 0.3
                            close_to_goal = (abs(position_error_x) < threshold) & (abs(position_error_y) < threshold)
                            far_to_door = (abs(world_agent.state.pos[:, 0] - self.door_x) > 1)
                            if i == 3 :

                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.65 + self.world.agents[0].state.pos[:, 0]*0.05)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.65 + self.world.agents[0].state.pos[:, 1]*0.05)
                                self.formation_goals[i][:, 0] = torch.where(far_to_door, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[4].state.pos[:, 0]*0.35)
                                self.formation_goals[i][:, 1] = torch.where(far_to_door, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[4].state.pos[:, 1]*0.35)
                            elif i == 4:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.65 + self.world.agents[0].state.pos[:, 0]*0.05)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.65 + self.world.agents[0].state.pos[:, 1]*0.05)
                                self.formation_goals[i][:, 0] = torch.where(far_to_door, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[3].state.pos[:, 0]*0.35)
                                self.formation_goals[i][:, 1] = torch.where(far_to_door, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[3].state.pos[:, 1]*0.35)
                            else:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.65 + self.world.agents[0].state.pos[:, 0]*0.05)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.65 + self.world.agents[0].state.pos[:, 1]*0.05)
                        elif self.env_type == "tunnel":
                            threshold = 0.3
                            close_to_goal = (abs(position_error_x) < threshold) & (abs(position_error_y) < threshold)
                            if i == 3 :

                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.45 + self.world.agents[4].state.pos[:, 0]*0.25)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.45 + self.world.agents[4].state.pos[:, 1]*0.25)
                            elif i == 4:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.45 + self.world.agents[3].state.pos[:, 0]*0.25)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.45 + self.world.agents[3].state.pos[:, 1]*0.25)
                            else:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.65 + self.world.agents[0].state.pos[:, 0]*0.05)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.65 + self.world.agents[0].state.pos[:, 1]*0.05)
                        elif self.env_type == "clutter":
                            threshold = 0.4
                            batch_size = self.batch_dim
                            
                            close_to_goal = (abs(position_error_x) < threshold) & (abs(position_error_y) < threshold)
                            target_index = 0
                            if i == 1:
                                target_index = 0
                            elif i == 2:
                                target_index = 0
                            elif i == 3:
                                target_index = 1
                            elif i == 4:
                                target_index = 2
                            target_position_error_x = self.formation_goals[target_index][:, 0] - self.world.agents[target_index].state.pos[:, 0]
                            target_position_error_y = self.formation_goals[target_index][:, 1] - self.world.agents[target_index].state.pos[:, 1]
                            
                            target_close_to_goal = (abs(target_position_error_x) < threshold) & (abs(target_position_error_y) < threshold)
                            
                            for dim in range(batch_size):
                                # Get obstacle manager for this environment
                                obstacle_manager = self.obstacle_manager_list[dim]
                                # Get obstacles near the agent
                                obstacles = obstacle_manager.get_near_obstacles(world_agent.state.pos[dim], 0.53)  # Adjust search radius as needed

                                if obstacles.numel() > 0:
                                    close_to_goal[dim] = (abs(position_error_x[dim]) < 0.1) & (abs(position_error_y[dim]) < 0.1)
                            if i == 3 :

                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.4 + self.world.agents[1].state.pos[:, 0]*0.3)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.4 + self.world.agents[1].state.pos[:, 1]*0.3)
                                self.formation_goals[i][:, 0] = torch.where(target_close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.1 + self.world.agents[1].state.pos[:, 0]*0.6)
                                self.formation_goals[i][:, 1] = torch.where(target_close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.1 + self.world.agents[1].state.pos[:, 1]*0.6)
                            
                            elif i == 4 :

                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.4 + self.world.agents[2].state.pos[:, 0]*0.3)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.4 + self.world.agents[2].state.pos[:, 1]*0.3)
                                self.formation_goals[i][:, 0] = torch.where(target_close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.1 + self.world.agents[2].state.pos[:, 0]*0.6)
                                self.formation_goals[i][:, 1] = torch.where(target_close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.1 + self.world.agents[2].state.pos[:, 1]*0.6)
                            elif i == 2 :

                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.4 + self.world.agents[0].state.pos[:, 0]*0.3)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.4 + self.world.agents[0].state.pos[:, 1]*0.3)
                                self.formation_goals[i][:, 0] = torch.where(target_close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.1 + self.world.agents[0].state.pos[:, 0]*0.6)
                                self.formation_goals[i][:, 1] = torch.where(target_close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.1 + self.world.agents[0].state.pos[:, 1]*0.6)
                            elif i == 1 :

                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.4 + self.world.agents[0].state.pos[:, 0]*0.3)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.4 + self.world.agents[0].state.pos[:, 1]*0.3)    
                        else:
                            close_to_goal = (abs(position_error_x) < threshold) & (abs(position_error_y) < threshold)
                            self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.65 + self.world.agents[0].state.pos[:, 0]*0.05)
                            self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.65 + self.world.agents[0].state.pos[:, 1]*0.05)
                        # self.formation_goals[i][:, 0] = self.formation_center_pos:, 0] + math.cos(self.formation_center_pos[:, 2] + angles[i]) * dists[i]
                        # self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + math.sin(self.formation_center_pos[:, 2] + angles[i]) * dists[i]
                    if self.working_mode == "potential_field":
                        self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + torch.cos(formation_angle) * dist
                        self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + torch.sin(formation_angle) * dist
                        self.formation_goals[i][:, 2] = self.formation_center_pos[:, 2] 
                        if self.obstacle_pattern == 2:
                            self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + torch.cos(formation_angle) * (dist/2)
                            self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + torch.sin(formation_angle) * (dist/2)


                    
                    
                    
                    
                    self.formation_normal_width = math.sin(45/180.0*math.pi)*0.5* 4
                    if self.obstacle_pattern == 1:
                        if self.working_mode == "imitation":
                            agent.set_vel(
                            self.compute_agent_velocity(agent, i),
                            batch_index=None,
                            )
                        elif self.working_mode == "RL":
                            agent.set_vel(
                                    torch.stack([3*(self.formation_goals[i][:, 0] - agent.state.pos[:, 0]), 3*(self.formation_goals[i][:, 1] - agent.state.pos[:, 1])], dim=-1) ,
                                batch_index=None,
                            )
                        elif self.working_mode == "potential_field":
                            agent.set_vel(
                            self.compute_potential_field_velocity(agent, i),
                            batch_index=None,
                            )
                    elif self.obstacle_pattern == 2:
                        agent.set_vel(
                                torch.stack([(self.formation_goals[i][:, 0] - agent.state.pos[:, 0]), (self.formation_goals[i][:, 1] - agent.state.pos[:, 1])], dim=-1) ,
                            batch_index=None,
                        )


                    
                    agent.set_ang_vel(
                            torch.stack([0.5*(self.formation_goals[i][:, 2] - agent.state.rot[:,0])], dim=-1),
                        batch_index=None,
                    )
                    
             
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
        
    
    def compute_agent_velocity(self, agent, agent_index):
        # Get the agent's current position (batch_size x 2)
        current_pos = agent.state.pos  # Shape: [batch_size, 2]

        # Get the agent's formation goal position
        goal_pos = self.formation_goals[agent_index][:, :2]  # Shape: [batch_size, 2]

        # Compute the goal direction vector
        goal_direction = goal_pos - current_pos  # Shape: [batch_size, 2]
        goal_distance = torch.norm(goal_direction, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        goal_direction_normalized = goal_direction / (goal_distance + 1e-6)  # Avoid division by zero
        if self.env_type == "clutter":
            target_index = 0
            if agent_index == 1:
                target_index = 0
            elif agent_index == 2:
                target_index = 0
            elif agent_index == 3:
                target_index = 1
            elif agent_index == 4:
                target_index = 2 
            leader_direction = self.formation_goals[target_index][:, :2] - current_pos
        else:
            leader_direction = self.formation_goals[0][:, :2] - current_pos
        leader_distance = torch.norm(leader_direction, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        leader_direction_normalized = leader_direction / (leader_distance + 1e-6)  # Avoid division by zero

        # Set the goal attraction strength
        k_goal = 5.0
        if self.env_type == "narrow" or self.env_type == "door" or self.env_type == "door_and_narrow":
            k_leader = 6.0
        elif self.env_type == "clutter":
            k_leader = 1.0
        else:
            k_leader = 2.0

        # Initialize the total force vector
        total_force = k_goal * goal_direction_normalized  # Shape: [batch_size, 2]

        # Initialize repulsive forces
        repulsive_forces = torch.zeros_like(total_force)  # Shape: [batch_size, 2]
        toward_leader_forces = torch.zeros_like(total_force)
        toward_leader_forces = k_leader* leader_direction_normalized
        # For each batch dimension
        batch_size = current_pos.shape[0]
        for dim in range(batch_size):
            # Get obstacle manager for this environment
            obstacle_manager = self.obstacle_manager_list[dim]
            # Get obstacles near the agent
            obstacles = obstacle_manager.get_near_obstacles(current_pos[dim], 0.8)  # Adjust search radius as needed

            if obstacles.numel() > 0:
                # Convert obstacle positions to torch tensor
                # obs_positions = torch.stack(obstacles).to(self.device)  # Shape: [num_obstacles, 2]

                # Compute vectors from agent to obstacles
                obs_vectors = current_pos[dim].unsqueeze(0) - obstacles  # Shape: [num_obstacles, 2]

                # Compute distances
                obs_distances = torch.norm(obs_vectors, dim=1, keepdim=True)  # Shape: [num_obstacles, 1]

                # Avoid division by zero
                obs_distances = torch.clamp(obs_distances, min=0.1)

                # Compute repulsive forces
                k_rep = 0.15  # Adjust repulsion strength as needed
                influence_range = 0.8  # Obstacles within this range influence the agent
                rep_force_magnitudes = k_rep * (1.0 / obs_distances - 1.0 / influence_range) / (obs_distances ** 2)
                # Only consider obstacles within influence range
                influence = (obs_distances < influence_range).float()
                rep_force_magnitudes = rep_force_magnitudes * influence

                # Compute repulsive force vectors
                repulsive_force_vectors = (obs_vectors / obs_distances) * rep_force_magnitudes  # Shape: [num_obstacles, 2]

                # Sum up repulsive forces
                total_repulsive_force = repulsive_force_vectors.sum(dim=0)  # Shape: [2]

                # Add to total force
                repulsive_forces[dim] = total_repulsive_force
            else:
                toward_leader_forces[dim] = 0 
        # Add repulsive forces to total force
        total_force += repulsive_forces
        total_force += toward_leader_forces
        # Normalize the total force to get the velocity direction
        total_force_norm = torch.norm(total_force, dim=1, keepdim=True)
        velocity_direction = total_force / (total_force_norm + 1e-6)

        # Set the agent's speed (you can adjust the speed as needed)
        max_speed = 3  # Limit speed to prevent excessive velocities
        agent_speed = torch.clamp(3*goal_distance, max=max_speed)  # Shape: [batch_size, 1]

        # Compute the velocity
        velocity = velocity_direction * agent_speed

        return velocity  # Shape: [batch_size, 2]

    def compute_potential_field_velocity(self, agent, agent_index):
        # Get the agent's current position (batch_size x 2)
        current_pos = agent.state.pos  # Shape: [batch_size, 2]

        # Get the agent's formation goal position
        goal_pos = self.formation_goals[agent_index][:, :2]  # Shape: [batch_size, 2]

        # Compute the goal direction vector
        goal_direction = goal_pos - current_pos  # Shape: [batch_size, 2]
        goal_distance = torch.norm(goal_direction, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        goal_direction_normalized = goal_direction / (goal_distance + 1e-6)  # Avoid division by zero
        if self.env_type == "clutter":
            target_index = 0
            if agent_index == 1:
                target_index = 0
            elif agent_index == 2:
                target_index = 0
            elif agent_index == 3:
                target_index = 1
            elif agent_index == 4:
                target_index = 2 
            leader_direction = self.formation_goals[target_index][:, :2] - current_pos
        else:
            leader_direction = self.formation_goals[0][:, :2] - current_pos
        
        

        all_positions = [agent.state.pos for agent in self.world.agents]
        center_pos = torch.stack(all_positions).mean(dim=0)  # Shape: [batch_size, 2]
        center_direction =  center_pos - current_pos
        center_distance = torch.norm(center_direction, dim = 1, keepdim=True)
        center_direction_normalized = center_direction / (center_distance + 1e-6)
        leader_distance = torch.norm(leader_direction, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        leader_direction_normalized = leader_direction / (leader_distance + 1e-6)  # Avoid division by zero

        # Set the goal attraction strength
        k_goal = 10.0
        if self.env_type == "narrow" or self.env_type == "door" or self.env_type == "door_and_narrow":
            k_leader = 6.0
        elif self.env_type == "clutter":
            k_leader = 1.0
        else:
            k_leader = 2.0

        # Initialize the total force vector
        total_force = k_goal *goal_distance* goal_direction_normalized  # Shape: [batch_size, 2]

        k_center = 2.0
        toward_center_forces = torch.zeros_like(total_force)
        toward_center_forces = k_center* center_distance * center_direction_normalized
        # Initialize repulsive forces
        repulsive_forces = torch.zeros_like(total_force)  # Shape: [batch_size, 2]
        toward_leader_forces = torch.zeros_like(total_force)
        toward_leader_forces = k_leader*leader_distance* leader_direction_normalized
        # For each batch dimension
        batch_size = current_pos.shape[0]
        for dim in range(batch_size):
            # Get obstacle manager for this environment
            obstacle_manager = self.obstacle_manager_list[dim]
            # Get obstacles near the agent
            obstacles = obstacle_manager.get_near_obstacles(current_pos[dim], 0.8)  # Adjust search radius as needed

            if obstacles.numel() > 0:
                # Convert obstacle positions to torch tensor
                # obs_positions = torch.stack(obstacles).to(self.device)  # Shape: [num_obstacles, 2]

                # Compute vectors from agent to obstacles
                obs_vectors = current_pos[dim].unsqueeze(0) - obstacles  # Shape: [num_obstacles, 2]

                # Compute distances
                obs_distances = torch.norm(obs_vectors, dim=1, keepdim=True)  # Shape: [num_obstacles, 1]

                # Avoid division by zero
                obs_distances = torch.clamp(obs_distances, min=0.1)

                # Compute repulsive forces
                k_rep = 2  # Adjust repulsion strength as needed
                influence_range = 0.5  # Obstacles within this range influence the agent
                rep_force_magnitudes = k_rep * (1.0 / obs_distances - 1.0 / influence_range) / (obs_distances ** 2)
                # Only consider obstacles within influence range
                influence = (obs_distances < influence_range).float()
                rep_force_magnitudes = rep_force_magnitudes * influence

                # Compute repulsive force vectors
                repulsive_force_vectors = (obs_vectors / obs_distances) * rep_force_magnitudes  # Shape: [num_obstacles, 2]

                # Sum up repulsive forces
                total_repulsive_force = repulsive_force_vectors.sum(dim=0)  # Shape: [2]

                # Add to total force
                repulsive_forces[dim] = total_repulsive_force
            else:
                toward_leader_forces[dim] = 0 
        
        k_agent_rep = 0.5             # Repulsion strength for agents (adjust as needed)
        influence_range_agent = 0.2   # Influence range for agent repulsion
        agent_rep_force = torch.zeros_like(total_force)  # Initialize repulsive force from agents

        for dim in range(batch_size):
            min_distance = float('inf')
            rep_force_vector = torch.zeros(2).to(current_pos.device)
            # Iterate through all agents in the world
            for idx, other_agent in enumerate(self.world.agents):
                if idx == agent_index:
                    continue  # Skip the current agent
                # Assuming each agent's state.pos is of shape [batch_size, 2]
                other_pos = other_agent.state.pos[dim]
                diff = current_pos[dim] - other_pos
                distance = torch.norm(diff)
                if distance < min_distance:
                    min_distance = distance
                    rep_force_vector = diff
            # Only apply the repulsive force if the closest agent is within the influence range
            if min_distance < influence_range_agent:
                rep_force_norm = torch.norm(rep_force_vector)
                rep_force_direction = rep_force_vector / (rep_force_norm + 1e-6)
                rep_magnitude = k_agent_rep * (1.0 / (min_distance + 1e-6) - 1.0 / influence_range_agent) / (min_distance**2 + 1e-6)
                agent_rep_force[dim] = rep_force_direction * rep_magnitude
            else:
                agent_rep_force[dim] = torch.zeros(2).to(current_pos.device)
        
        
        
        
        # Add repulsive forces to total force 
        total_force += repulsive_forces       
        # total_force += toward_leader_forces 
        total_force += toward_center_forces
        total_force += agent_rep_force    
        # Normalize the total force to get the velocity direction
        total_force_norm = torch.norm(total_force, dim=1, keepdim=True)
        velocity_direction = total_force / (total_force_norm + 1e-6)

        # Set the agent's speed (you can adjust the speed as needed)
        max_speed = 3  # Limit speed to prevent excessive velocities
        agent_speed = torch.clamp(3*goal_distance, max=max_speed)  # Shape: [batch_size, 1]

        new_velocity = velocity_direction * agent_speed

        # --- Damping / Smoothing Step ---
        # Blend the new velocity with the previous one to smooth out oscillations.
        # Here, we assume agent.state has an attribute `prev_velocity`. If it doesn't exist,
        # we initialize it to zero.
        if hasattr(agent, "prev_velocity"):
            smoothed_velocity = 0.7 * new_velocity + 0.3 * agent.prev_velocity
        else:
            smoothed_velocity = new_velocity
        # Update the previous velocity for the next iteration
        agent.prev_velocity = smoothed_velocity.clone()
    # --- End of Damping Step ---

        return smoothed_velocity
        
        
        
        
        
        # # Compute the velocity
        # velocity = velocity_direction * agent_speed

        # return velocity  # Shape: [batch_size, 2]

    def get_formation_minimal_through_width(self, formation_type, follower_formation_num, d, d_obs, robot_radius):
        if formation_type == "ren_shape":
            minimal_width = follower_formation_num * ((d  + 2*robot_radius)* math.sin(45/180.0*math.pi)) 
            
        elif formation_type == "rectangle":
            minimal_width = d + robot_radius + 2 * d_obs
        return minimal_width

    
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
            self.formation_goals[i][0] = new_goal_x
            self.formation_goals[i][1] = new_goal_y
            

        return not collision_detected


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
        
    def transform_global_to_local(self, global_pos, leader_pos, leader_rot, device):
        """
        Transforms positions from the global frame to the local frame of the leader.
        """
        global_pos = global_pos.to(device)
        leader_pos = leader_pos.to(device)
        leader_rot = leader_rot.to(device)

        # Translate to leader's local origin
        translated_pos = global_pos - leader_pos  # [batch_dim, 2]

        # Compute rotation matrix for the negative angle
        cos_theta = torch.cos(-leader_rot).unsqueeze(-1)
        sin_theta = torch.sin(-leader_rot).unsqueeze(-1)
        rotation_matrix = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).reshape(-1, 2, 2)  # [batch_dim, 2, 2]

        # Rotate to align with local frame
        local_pos = torch.matmul(rotation_matrix, translated_pos.unsqueeze(-1)).squeeze(-1)  # [batch_dim, 2]
        return local_pos
    


    def agent_formation_reward(self, agent: Agent):
        current_agent_index = self.world.agents.index(agent)

        # Nominal positions in the local frame
        nominal_positions_x = torch.tensor([0.0, -0.5534992, -0.5534992, -0.95179105, -0.95179105], device=self.device)
        nominal_positions_y = torch.tensor([0.0, 0.35284618, -0.35284618, 0.7946659, -0.7946659], device=self.device)
        nominal_positions = torch.stack([nominal_positions_x, nominal_positions_y], dim=-1)  # [5, 2]

        # Global positions of all agents
        agent_positions = torch.stack([a.state.pos for a in self.world.agents])  # [num_agents, num_envs, 2]
        num_envs = agent_positions.shape[1]  # Number of environments (e.g., 1200)

        # Leader (0th robot) position and orientation
        leader_robot_pos = agent_positions[0]  # [num_envs, 2]
        leader_robot_rot = self.world.agents[0].state.rot  # [num_envs]

        # Transform global positions to the local frame of the 0th robot
        # print("agent_positions shape:{}".format(agent_positions.shape))
        relative_positions_global = agent_positions[current_agent_index,:,:].squeeze(dim=0) - leader_robot_pos # [num_agents, num_envs, 2]
        # print("global shape:{}".format(relative_positions_global.shape))
        relative_positions_local = self.transform_global_to_local(
            relative_positions_global,  # Flatten for batched transformation
            leader_robot_pos,
            leader_robot_rot,
            self.device
        ) # Reshape back to [num_agents, num_envs, 2]

        # Calculate deviation from nominal positions
        nominal_position = nominal_positions[current_agent_index]  # [2]
        deviation = relative_positions_local - nominal_position  # [num_envs, 2]

        # Formation reward is the negative norm of the deviation
        formation_rew = -torch.linalg.vector_norm(deviation, dim=-1)  # [num_envs]
        # print("formatION-rew:{}".format(formation_rew))
        # Return the formation reward for this agent
        return 0.05*formation_rew

    def agent_velocity_target_direction_alignment_reward(self, agent: Agent):
        current_agent_index = self.world.agents.index(agent)

        # Get the target direction towards the formation goal (landmark)
        target_dir = self.formation_goals_landmark[current_agent_index].state.rot  # Scalar (angle, in radians)

        # Convert target_dir (scalar) to a unit vector using torch functions
        target_dir_vector = torch.stack([torch.cos(target_dir), torch.sin(target_dir)], dim=-1).squeeze(dim=1)  # 2D unit vector

        # Get the agent's current position and direction
        agent_pos = agent.state.pos
        goal_pos = self.formation_goals_landmark[current_agent_index].state.pos

        # Calculate the vector from agent to the goal and normalize it
        agent_to_goal_vector = goal_pos - agent_pos
        agent_to_goal_direction = agent_to_goal_vector / agent_to_goal_vector.norm(p=2, dim=-1, keepdim=True)  # Normalize the vector
        # print("agent_to_goal_vector shape:{}".format(agent_to_goal_direction.shape))
        # print("target)dir)vector shapeeeee:{}".format(target_dir_vector.shape))
        # Calculate the cosine of the angle between the agent's direction and the target direction
        dot_product = torch.sum(agent_to_goal_direction * target_dir_vector, dim=-1)  # Dot product
        # print("dot product shape:{}".format(dot_product.shape))
        # Clip the dot product to avoid numerical errors (cosine should be between -1 and 1)
        dot_product = torch.clamp(dot_product, min=-1.0, max=1.0)
        angle_cosine = dot_product

        # The reward is the negative cosine of the angle to penalize discrepancies
        reward = -self.agent_velocity_target_direction_alignment_reward_weight * angle_cosine  # Higher values indicate better alignment
        # print("alignment reward shape:{}".format(reward.shape))
        return reward


    def agent_target_collision_reward(self, agent: Agent):
        """
        Calculates and updates the collision-related rewards for the given agent.

        Args:
            agent (Agent): The agent for which to calculate the collision reward.
        """
        current_agent_index = self.world.agents.index(agent)

        # Assuming batch_dim corresponds to the number of parallel environments or agents
        batch_size = self.batch_dim
        closest_dist = 10.0 * torch.ones(batch_size, device=self.device)  # Initialize with a high distance

        # Agent's current position and target position
        agent_pos = agent.state.pos  # Shape: [batch_size, 3]
        target_pos = self.formation_goals_landmark[current_agent_index].state.pos  # Shape: [batch_size, 3]

        # Initialize a mask for obstacles too close to the connecting line
        line_collision_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for obs in self.obstacles:
            # Calculate the distance from the target to the obstacle
            distance = self.world.get_distance(self.formation_goals_landmark[current_agent_index], obs)  # Shape: [batch_size]
            closest_dist = torch.min(closest_dist, distance)  # Update the closest distance

            # Obstacle's position
            obstacle_pos = obs.state.pos  # Shape: [batch_size, 3]

            # Vectors from agent to target and agent to obstacle
            AB = target_pos - agent_pos  # Shape: [batch_size, 3]
            AC = obstacle_pos - agent_pos  # Shape: [batch_size, 3]

            # Calculate the squared magnitude of AB
            AB_sq = torch.sum(AB ** 2, dim=1)  # Shape: [batch_size]
            AB_sq = torch.clamp(AB_sq, min=1e-6)  # Prevent division by zero

            # Projection factor
            t = torch.sum(AC * AB, dim=1) / AB_sq  # Shape: [batch_size]
            t_clamped = torch.clamp(t, 0.0, 1.0)  # Clamp between 0 and 1

            # Projection point on AB
            P = agent_pos + t_clamped.unsqueeze(1) * AB  # Shape: [batch_size, 3]

            # Distance from obstacle to the line segment AB
            distance_to_line = torch.norm(obstacle_pos - P, dim=1)  # Shape: [batch_size]

            # Define threshold for line collision
            threshold_line = 0.1
            obstacle_too_close = distance_to_line < threshold_line  # Shape: [batch_size]

            # Update the collision mask
            line_collision_mask = line_collision_mask | obstacle_too_close  # Logical OR across obstacles

        # Reward adjustments based on closest distance to target
        collision_mask = closest_dist < 0.2  # Agents too close to obstacles at the target
        collision_free_mask = closest_dist > 0.5  # Agents sufficiently far from obstacles at the target

        if torch.any(collision_mask):
            # Penalize agents based on how close they are to the obstacles at the target
            agent.target_collision_rew[collision_mask] += -0.4 * (1 - closest_dist[collision_mask] / 0.2)

        # Reward adjustments based on proximity of obstacles to the connecting line
        if torch.any(line_collision_mask):
            large_negative_rew = -1.0  # Define a large negative reward
            agent.target_collision_rew[line_collision_mask] += large_negative_rew

        # Optional: Additional rewards or penalties can be added here
        # For example, encouraging agents to stay within certain bounds, follow formations, etc.

            #agent positions: agent.state.pos
        return agent.target_collision_rew

    
    def compute_agent_connectivity_reward(self, agent):
        current_agent_index = self.world.agents.index(agent)
        target_index = 0
        if current_agent_index == 0:
            return agent.connection_rew
        elif current_agent_index == 1:
            target_index = 0
        elif current_agent_index == 2:
            target_index = 0
        elif current_agent_index == 3:
            target_index = 1
        elif current_agent_index == 4:
            target_index = 2
        target_agent = self.world.agents[target_index] 
        pos_i = agent.state.pos  # [batch_size, 2]
        pos_j = target_agent.state.pos  # [batch_size, 2]
        rot_i = agent.state.rot  # [batch_size]
        
        

        # Relative position and distance: [batch_size, 2], [batch_size]
        rel_pos = pos_j - pos_i  # [batch_size, 2]
        d_ij = torch.norm(rel_pos, dim=1)  # [batch_size]
        
        # Check if within distance

        within_distance = d_ij <= self.max_connection_distance  # [batch_size]
        # print("within_distance shape:{}".format(within_distance.shape))
        # Compute relative angles: [batch_size]
        # print("rel_pos shape:{}".format(torch.atan2(rel_pos[:, 1], rel_pos[:, 0]).shape))
        # print("rot_i shape:{}".format(rot_i.squeeze(dim=1).shape))
        theta_ij = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]) - rot_i.squeeze(dim=1)  # [batch_size]
        # Normalize angles to [-pi, pi]
        theta_ij = torch.atan2(torch.sin(theta_ij), torch.cos(theta_ij))  # [batch_size]
        # print("theta_ij shape:{}".format(theta_ij.shape))
        # Check if within FOV
        within_fov = (theta_ij >= self.FOV_min) & (theta_ij <= self.FOV_max)  # [batch_size]
        # print("within_fov shape:{}".format(within_fov.shape))
        # Connection exists if within distance and FOV
        connection = within_distance & within_fov  # [batch_size]
        
        # Ensure connection is [batch_size]
        connection = connection.view(-1)  # [batch_size]
        # print("connection:{}".format(connection[0].float() ))
        batch_size = self.batch_dim
        connectivity_reward = torch.zeros(batch_size, device=self.device)
        for env_index in range(batch_size):
            if connection[env_index].float() == 1.0:
                is_line_clear = self.is_line_clear(agent.state.pos[env_index, :], target_agent.state.pos[env_index, :], self.obstacle_manager_list[env_index])
                if is_line_clear == True:
                    connectivity_reward[env_index] = self.connection_reward_positive
                    self.eva_connection_num[current_agent_index-1 ,env_index] = 1
                    # print("ev_con_num:{}".format(self.eva_connection_num))
            
            else:

                # If connection criterion is not met, apply reward shaping based on distance and angle
                # Compute penalty based on distance to target
                distance_penalty = (d_ij[env_index] - self.max_connection_distance).clamp(min=0)  # Positive penalty if outside max distance
                
                # Compute angular penalty (absolute angular difference to the field of view)
                angular_diff = torch.abs(theta_ij[env_index])

                # Calculate the total field of view width
                fov_width = self.FOV_max - self.FOV_min

                # If the angular difference exceeds half the field of view width, apply the penalty
                if angular_diff > fov_width / 2:
                    angular_penalty = angular_diff - fov_width / 2
                else:
                    angular_penalty = 0  # No penalty if within the FOV range

                # angular_penalty = angular_penalty.clamp(min=0)  # Positive penalty if angle is outside the FOV range
                angular_penalty = torch.tensor(angular_penalty, dtype=torch.float).clamp(min=0)
                # Combine distance and angular penalties (you can weigh them as desired)
                shaping_penalty = distance_penalty + angular_penalty

                # Set the reward to be negative of the shaping penalty
                connectivity_reward[env_index] = -shaping_penalty * self.connection_reward_negative


                # connectivity_reward[env_index] = self.connection_reward_negative
        return connectivity_reward

    def compute_connectivity_reward(self):
        """
        Computes the connectivity reward for each environment in the batch.
        Returns a tensor of shape [batch_size] with connectivity rewards.
        """
        num_agents = len(self.world.agents)
        batch_size = self.batch_dim  # Number of parallel environments
        
        # Initialize adjacency matrices: [batch_size, num_agents, num_agents]
        adjacency = torch.zeros(batch_size, num_agents, num_agents, device=self.device)
        
        # Compute pairwise connections based on FOV and distance
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if i == j:
                    continue
                # Get positions and rotations: [batch_size, 2], [batch_size]
                pos_i = agent1.state.pos  # [batch_size, 2]
                pos_j = agent2.state.pos  # [batch_size, 2]
                rot_i = agent1.state.rot  # [batch_size]
                
                

                # Relative position and distance: [batch_size, 2], [batch_size]
                rel_pos = pos_j - pos_i  # [batch_size, 2]
                d_ij = torch.norm(rel_pos, dim=1)  # [batch_size]
                
                # Check if within distance
                within_distance = d_ij <= self.max_connection_distance  # [batch_size]
                # print("within_distance shape:{}".format(within_distance.shape))
                # Compute relative angles: [batch_size]
                # print("rel_pos shape:{}".format(torch.atan2(rel_pos[:, 1], rel_pos[:, 0]).shape))
                # print("rot_i shape:{}".format(rot_i.squeeze(dim=1).shape))
                theta_ij = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]) - rot_i.squeeze(dim=1)  # [batch_size]
                # Normalize angles to [-pi, pi]
                theta_ij = torch.atan2(torch.sin(theta_ij), torch.cos(theta_ij))  # [batch_size]
                # print("theta_ij shape:{}".format(theta_ij.shape))
                # Check if within FOV
                within_fov = (theta_ij >= self.FOV_min) & (theta_ij <= self.FOV_max)  # [batch_size]
                # print("within_fov shape:{}".format(within_fov.shape))
                # Connection exists if within distance and FOV
                connection = within_distance & within_fov  # [batch_size]
                
                # Ensure connection is [batch_size]
                connection = connection.view(-1)  # [batch_size]
                # print("connection shape:{}".format(connection.shape))
                # Update adjacency matrices: set to 1 if connected
                adjacency[:, i, j] = connection.float()
                adjacency[:, j, i] = connection.float()
                    
                for dim in range(self.world.batch_dim):
                    if adjacency[dim, i, j] == 1.0:
                        is_line_clear = self.is_line_clear(agent1.state.pos[dim, :], agent2.state.pos[dim, :], self.obstacle_manager_list[dim])
                        if is_line_clear == False:
                            adjacency[dim, i, j] == 0.0
                            adjacency[dim, j, i] = 0.0
        # Initialize connectivity rewards: [batch_size]
        connectivity_reward = torch.zeros(batch_size, device=self.device)
        # print("adj:{}".format(adjacency))
        # Check connectivity for each environment using BFS
        for env in range(batch_size):
            adj = adjacency[env].cpu().numpy()  # [num_agents, num_agents]
            visited = set()
            queue = deque()
            queue.append(0)  # Start BFS from the first agent
            visited.add(0)
            
            while queue:
                current = queue.popleft()
                for neighbor in range(num_agents):
                    if adj[current, neighbor] == 1 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Assign rewards based on connectivity
            if len(visited) == num_agents:
                # Graph is connected
                # print("graph is connected")
                connectivity_reward[env] = self.connection_reward_positive
                self.eva_connection_num[:, env] = 1
            else:
                # Graph is disconnected
                # print("graph is not connected")
                connectivity_reward[env] = self.connection_reward_negative
        
        # Optionally scale the connectivity rewards
        # connectivity_reward *= self.connection_reward_coefficient
        
        return connectivity_reward  # [batch_size]
    
    def reward(self, agent: Agent):
        current_agent_index = self.world.agents.index(agent)
        is_first = agent == self.world.agents[0] 
        # agent.agent_collision_rew[:] = 0
        if is_first:

            self.eva_collision_num = torch.zeros(self.n_agents - 1, self.batch_dim, device=self.device)
        #number of agents that are connected to leader
            self.eva_connection_num = torch.zeros(self.n_agents - 1, self.batch_dim, device=self.device)
            self.t += 1
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            self.angle_rew[:] = 0
            self.formation_maintain_rew[:] = 0
            for i, a in enumerate(self.world.agents):
                a.agent_collision_rew[:] = 0
                a.pos_rew[:] = 0
                a.formation_rew[:] = 0
                a.target_collision_rew[:] = 0
                a.connection_rew[:] = 0
                a.angle_diff_rew[:] = 0
                # print("angle_diff:{}".format(a.angle_diff_with_leader_rew))
                a.angle_diff_with_leader_rew[:] = 0
                a.action_diff_rew[:] = 0
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i >= j:
                        continue
                    # if self.world.collides(a, b):
                    # print("a.state.pos:{}".format(a.state.pos))
                    # print("b.state.pos:{}".format(b.state.pos))

                    distance = torch.norm(a.state.pos - b.state.pos)
                    # print("dist:{}".format(distance))
                    # distance = self.world.get_distance(a, b)
                    # if distance <= self.min_collision_distance:
                        # print("collision between {}, {}".format(i, j))
                    # print("collision index:{}".format(distance <= self.min_collision_distance))
                    if i > 0:
                        self.eva_collision_num[i-1 ,distance <= self.min_collision_distance] = 1
                    if j > 0:
                        self.eva_collision_num[j-1 ,distance <= self.min_collision_distance] = 1
                    a.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty
                    # print("a.agent_collision_rew:{}".format(a.agent_collision_rew))
                    b.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty
                    # print("b.agent_collision_rew:{}".format(b.agent_collision_rew))
                    # input("1")
            for i, a in enumerate(self.world.agents):
                closest_dist = 10 * torch.ones(self.batch_dim, device=self.device)
                # agent_formation_rew = self.agent_formation_reward(agent)
                # print("obstacle_tenspr shape:{}".format(self.obstacle_manager_list[0].obstacles_tensor.shape))
                distances = torch.norm(a.state.pos.unsqueeze(1) - self.obstacle_manager_list[0].obstacles_tensor.unsqueeze(0), dim=2)

                closest_dist = torch.min(distances, dim=1).values
    
                # 后续碰撞判断逻辑保持不变
                # collision_mask = closest_dist < 0.2
                # Iterate through each obstacle
                # for obs in self.obstacles:
                #     # Calculate the distance to the obstacle
                #     # distance = self.world.get_distance(a, obs)
                #     distance = torch.norm(a.state.pos - obs.state.pos)
                #     # print("distance:{}".format(distance))

                #     # Compare and update the closest distance
                #     closest_dist = torch.min(closest_dist, distance)

                # Check for collision and update the reward based on closest distance
                collision_mask = closest_dist < 0.2
                if torch.any(collision_mask):  # Check if there are any collisions
                    if i > 0:
                        self.eva_collision_num[i-1 ,collision_mask] = 1
                    a.agent_collision_rew[collision_mask] += -0.4 * (1 - closest_dist[collision_mask] / 0.2)
                # if torch.any(collision_free_mask):
                    # a.formation_rew[collision_free_mask] = agent_formation_rew[collision_free_mask]
        # agent.pos_rew = self.single_agent_reward_graph_formation_maintained(agent)
            self.connection_rew = self.compute_connectivity_reward()
            # print("connection time:{}".format(time.time() - connection_time))
        
        #leader robot do not contribute to the reward
        if is_first:
            # print("single reward timme:{}, index:{}".format(time.time() - reward_time,current_agent_index))
            return agent.target_collision_rew

        if self.env_type == "clutter":
            agent.connection_rew = self.compute_agent_connectivity_reward(agent)
        else:
            agent.connection_rew = self.connection_rew.clone()

        # agent.target_collision_rew = self.agent_target_collision_reward(agent)
        # for i, a in enumerate(self.world.agents):
        #     for j, b in enumerate(self.world.agents):
        #         if i >= j:
        #             continue
        #         # if self.world.collides(a, b):
        #         distance = self.world.get_distance(self.formation_goals_landmark[i], self.formation_goals_landmark[j])
        #         # if distance <= self.min_collision_distance:
        #             # print("collision between {}, {}".format(i, j))
        #         # print("collision index:{}".format(distance <= self.min_collision_distance))
        #         a.target_collision_rew[
        #             distance <= self.min_target_collision_distance
        #         ] += self.agent_collision_penalty
        #         # print("a.agent_collision_rew:{}".format(a.agent_collision_rew))
        #         b.target_collision_rew[
        #             distance <= self.min_target_collision_distance
        #         ] += self.agent_collision_penalty
                    # print("b.agent_collision_rew:{}".format(b.agent_collision_rew))
                    # input("1")
        
        
        # pos_reward =  agent.pos_rew
        if current_agent_index != 0:
            if (current_agent_index in self.last_action_u) == False:
                self.last_action_u[current_agent_index] = agent.action.u
            agent.action_diff_rew = -(torch.norm(agent.action.u - self.last_action_u[current_agent_index], dim=1) -0.05)
            agent.angle_diff_with_leader_rew = -torch.norm(agent.state.rot - self.leader_agent.state.rot, dim=1)
        # agent_direction_align_reward = self.agent_velocity_target_direction_alignment_reward(agent)
        # print("agent agent_collision_rew shape:{}".format(agent.agent_collision_rew.shape))
        # angle_reward = self.angle_rew if self.shared_rew else agent.angle_rew
        # return 5*pos_reward + self.final_rew + agent.agent_collision_rew + angle_reward
        # return self.formation_maintain_rew + agent.agent_collision_rew
        # print("agent {} connection reward:{} collsition reward:{}, connection_rew:{}, target collision rew:{}".format(current_agent_index, agent.connection_rew, agent.agent_collision_rew, agent.connection_rew,agent.target_collision_rew))
        # return agent.agent_collision_rew + agent.connection_rew + agent.action_diff_rew + agent.formation_rew
        agent.agent_collision_rew = 10*agent.agent_collision_rew
        agent.connection_rew = 2*agent.connection_rew
        agent.action_diff_rew = 0.4*agent.action_diff_rew
        agent.angle_diff_with_leader_rew = 0.8*agent.angle_diff_with_leader_rew
        # agent.target_collision_rew = 10*agent.target_collision_rew
        # print("single reward timme:{}, index:{}".format(time.time() - reward_time,current_agent_index))
        # return agent.angle_diff_with_leader_rew + agent.agent_collision_rew + agent.connection_rew + agent.action_diff_rew + agent.target_collision_rew 
        return agent.angle_diff_with_leader_rew + agent.agent_collision_rew + agent.connection_rew + agent.action_diff_rew 
    
        # return agent.agent_collision_rew
        # return agent.connection_rew + agent.action_diff_rew
        # return agent.action_diff_rew


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

 

    


    def single_env_observation_graph_from_data(self, d, nominal_formation_tensor, near_obstacles_in_leader_frame, threshold_distance=2.5):
        # print("nominal_formation_tensor shape:{}".format(nominal_formation_tensor.shape))
        
        nominal_formation_category = torch.zeros((nominal_formation_tensor.size(0), 1), device=self.device)  # [5, 1] with category 0
        action_u_tensor = torch.zeros((nominal_formation_tensor.size(0), 3), device=self.device) 
        if self.working_mode == "RL":

            
            # print("RL")
            for i, world_agent in enumerate(self.world.agents):
                if world_agent.action.u != None:
                    # print("1")
                    # if world_agent.action.u[d, 0] == 0.0 and world_agent.action.u[d, 1] == 0.0:
                    leader_robot_pos = self.world.agents[0].state.pos[d,:2]  # [num_envs, 2]
                    leader_robot_rot = self.world.agents[0].state.rot[d]  # [num_envs]

                    # Transform global positions to the local frame of the 0th robot
                    # print("agent_positions shape:{}".format(agent_positions.shape))
                    # relative_positions_global = world_agent.state.pos[d, :2] - leader_robot_pos # [num_agents, num_envs, 2]
                    # print("global shape:{}".format(relative_positions_global))
                    relative_positions_local = self.transform_global_to_local(
                        world_agent.state.pos[d, :2],  # Flatten for batched transformation
                        leader_robot_pos,
                        leader_robot_rot,
                        self.device
                    ) 
                    # print("local position:{}".format(relative_positions_local))
                    action_u_tensor[i, :2] = relative_positions_local
                    action_u_tensor[i, 2] = 0
                    # else:
                    #     action_u_tensor[i, :] = world_agent.action.u[d, :]
                else:
                    leader_robot_pos = self.world.agents[0].state.pos[d,:2]  # [num_envs, 2]
                    leader_robot_rot = self.world.agents[0].state.rot[d]  # [num_envs]

                    # Transform global positions to the local frame of the 0th robot
                    # print("agent_positions shape:{}".format(agent_positions.shape))
                    # relative_positions_global = world_agent.state.pos[d, :2] - leader_robot_pos # [num_agents, num_envs, 2]
                    # print("global shape:{}".format(relative_positions_global))
                    relative_positions_local = self.transform_global_to_local(
                        world_agent.state.pos[d, :2],  # Flatten for batched transformation
                        leader_robot_pos,
                        leader_robot_rot,
                        self.device
                    ) 
                    # print("local position:{}".format(relative_positions_local))
                    action_u_tensor[i, :2] = relative_positions_local
                    action_u_tensor[i, 2] = 0
        if self.working_mode == "imitation" or self.working_mode == "potential_field":
            # print("imitation")
            for i, world_agent in enumerate(self.world.agents):
                # if world_agent.action.u != None:

                leader_robot_pos = self.world.agents[0].state.pos[d,:2]  # [num_envs, 2]
                leader_robot_rot = self.world.agents[0].state.rot[d]  # [num_envs]

                # Transform global positions to the local frame of the 0th robot
                # print("agent_positions shape:{}".format(agent_positions.shape))
                # relative_positions_global = world_agent.state.pos[d, :2] - leader_robot_pos # [num_agents, num_envs, 2]
                # print("global shape:{}".format(relative_positions_global))
                relative_positions_local = self.transform_global_to_local(
                    world_agent.state.pos[d, :2],  # Flatten for batched transformation
                    leader_robot_pos,
                    leader_robot_rot,
                    self.device
                ) 
                # print("local position:{}".format(relative_positions_local))
                action_u_tensor[i, :2] = relative_positions_local
                action_u_tensor[i, 2] = 0
        nominal_formation_tensor = torch.cat((nominal_formation_tensor, nominal_formation_category), dim=1)  # Shape: [5, 3]
        # nominal_formation_tensor = torch.cat((nominal_formation_tensor, action_u_tensor), dim=1)
        if near_obstacles_in_leader_frame is not None:
            # print("near_obstacles shape:{}".format(near_obstacles_in_leader_frame.shape))
        
            near_obstacles_in_leader_frame_add_dim = torch.zeros((near_obstacles_in_leader_frame.size(0), 1), device=self.device)  
            near_obstacles_category = torch.ones((near_obstacles_in_leader_frame.size(0), 1), device=self.device)  # [obstacle_num, 1] with category 1
            action_u_placeholder = torch.ones((near_obstacles_in_leader_frame.size(0), 3), device=self.device)
            near_obstacles_in_leader_frame_tensor = torch.cat((near_obstacles_in_leader_frame, near_obstacles_in_leader_frame_add_dim), dim=1)
            near_obstacles_in_leader_frame = torch.cat((near_obstacles_in_leader_frame_tensor, near_obstacles_category), dim=1)  # Shape: [obstacle_num, 3]
            # near_obstacles_in_leader_frame = torch.cat((near_obstacles_in_leader_frame, action_u_placeholder), dim=1)
            num_obstacles = near_obstacles_in_leader_frame.size(0)
        


        # Add a category feature to each node (0 for nominal formation, 1 for obstacles)
        
        # Concatenate category feature to position tensors
        x = None
        # Combine all node features into a single tensor
        if near_obstacles_in_leader_frame != None:
            x = torch.cat((nominal_formation_tensor, near_obstacles_in_leader_frame), dim=0)  # Shape: [5 + obstacle_num, 3]
        else:
            x = nominal_formation_tensor
        # Initialize edge index and edge attributes
        edge_index = []
        edge_attr = []

        # Number of agents and obstacles
        num_agents = nominal_formation_tensor.size(0)
        
        # Connect each nominal formation agent with near obstacles
        if near_obstacles_in_leader_frame is not None:
            for agent_index in range(num_agents):
                agent_pos = nominal_formation_tensor[agent_index, :2]  # Get the position part
                for obstacle_index in range(num_obstacles):
                    obstacle_pos = near_obstacles_in_leader_frame[obstacle_index, :2]  # Get the position part
                    distance = torch.norm(agent_pos - obstacle_pos)
                    if distance <= self.max_obstacle_edge_range:  # Check if within threshold distance
                        # Add edges from agent to obstacle
                        edge_index.append([agent_index, num_agents + obstacle_index])  # Agent to obstacleinfo
                        edge_index.append([num_agents + obstacle_index, agent_index])  # Obstacle to agent
                        edge_attr.append([distance.item()])  # Edge type 1 for agent-obstacle
                        edge_attr.append([distance.item()])  # Edge type 1 for agent-obstacle

        # Connect each pair of nominal formation agents
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
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
        
        return data

    
    def single_graph_from_data(self, d, nominal_formation_tensor, near_obstacles_in_leader_frame, threshold_distance=2.5):
        # print("nominal_formation_tensor shape:{}".format(nominal_formation_tensor.shape))
        
        nominal_formation_category = torch.zeros((nominal_formation_tensor.size(0), 1), device=self.device)  # [5, 1] with category 0
        action_u_tensor = torch.zeros((nominal_formation_tensor.size(0), 3), device=self.device) 
        if self.working_mode == "RL":

            
            # print("RL")
            for i, world_agent in enumerate(self.world.agents):
                if world_agent.action.u != None:
                    # print("1")
                    # if world_agent.action.u[d, 0] == 0.0 and world_agent.action.u[d, 1] == 0.0:
                    leader_robot_pos = self.world.agents[0].state.pos[d,:2]  # [num_envs, 2]
                    leader_robot_rot = self.world.agents[0].state.rot[d]  # [num_envs]

                    # Transform global positions to the local frame of the 0th robot
                    # print("agent_positions shape:{}".format(agent_positions.shape))
                    # relative_positions_global = world_agent.state.pos[d, :2] - leader_robot_pos # [num_agents, num_envs, 2]
                    # print("global shape:{}".format(relative_positions_global))
                    relative_positions_local = self.transform_global_to_local(
                        world_agent.state.pos[d, :2],  # Flatten for batched transformation
                        leader_robot_pos,
                        leader_robot_rot,
                        self.device
                    ) 
                    # print("local position:{}".format(relative_positions_local))
                    action_u_tensor[i, :2] = relative_positions_local
                    action_u_tensor[i, 2] = 0
                    # else:
                    #     action_u_tensor[i, :] = world_agent.action.u[d, :]
                else:
                    leader_robot_pos = self.world.agents[0].state.pos[d,:2]  # [num_envs, 2]
                    leader_robot_rot = self.world.agents[0].state.rot[d]  # [num_envs]

                    # Transform global positions to the local frame of the 0th robot
                    # print("agent_positions shape:{}".format(agent_positions.shape))
                    # relative_positions_global = world_agent.state.pos[d, :2] - leader_robot_pos # [num_agents, num_envs, 2]
                    # print("global shape:{}".format(relative_positions_global))
                    relative_positions_local = self.transform_global_to_local(
                        world_agent.state.pos[d, :2],  # Flatten for batched transformation
                        leader_robot_pos,
                        leader_robot_rot,
                        self.device
                    ) 
                    # print("local position:{}".format(relative_positions_local))
                    action_u_tensor[i, :2] = relative_positions_local
                    action_u_tensor[i, 2] = 0
        if self.working_mode == "imitation" or self.working_mode == "potential_field":
            # print("imitation")
            for i, world_agent in enumerate(self.world.agents):
                # if world_agent.action.u != None:

                leader_robot_pos = self.world.agents[0].state.pos[d,:2]  # [num_envs, 2]
                leader_robot_rot = self.world.agents[0].state.rot[d]  # [num_envs]

                # Transform global positions to the local frame of the 0th robot
                # print("agent_positions shape:{}".format(agent_positions.shape))
                # relative_positions_global = world_agent.state.pos[d, :2] - leader_robot_pos # [num_agents, num_envs, 2]
                # print("global shape:{}".format(relative_positions_global))
                relative_positions_local = self.transform_global_to_local(
                    world_agent.state.pos[d, :2],  # Flatten for batched transformation
                    leader_robot_pos,
                    leader_robot_rot,
                    self.device
                ) 
                # print("local position:{}".format(relative_positions_local))
                action_u_tensor[i, :2] = relative_positions_local
                action_u_tensor[i, 2] = 0
        nominal_formation_tensor = torch.cat((nominal_formation_tensor, nominal_formation_category), dim=1)  # Shape: [5, 3]
        nominal_formation_tensor = torch.cat((nominal_formation_tensor, action_u_tensor), dim=1)
        def add_noise(tensor, noise_level=0.1):
            noise = torch.rand_like(tensor) - 0.5
            # Scale the noise by noise_level
            noise = noise * noise_level
            # Add the noise to the original tensor
            return tensor + noise
        if near_obstacles_in_leader_frame is not None:
            # print("near_obstacles shape:{}".format(near_obstacles_in_leader_frame.shape))
            # print("near_obstacles_in_leader_frame shape:{}".format(near_obstacles_in_leader_frame.shape))
            near_obstacles_in_leader_frame = add_noise(near_obstacles_in_leader_frame, self.evaluation_noise)
            near_obstacles_in_leader_frame_add_dim = torch.zeros((near_obstacles_in_leader_frame.size(0), 1), device=self.device)  
            near_obstacles_category = torch.ones((near_obstacles_in_leader_frame.size(0), 1), device=self.device)  # [obstacle_num, 1] with category 1
            action_u_placeholder = torch.ones((near_obstacles_in_leader_frame.size(0), 3), device=self.device)
            near_obstacles_in_leader_frame_tensor = torch.cat((near_obstacles_in_leader_frame, near_obstacles_in_leader_frame_add_dim), dim=1)
            near_obstacles_in_leader_frame = torch.cat((near_obstacles_in_leader_frame_tensor, near_obstacles_category), dim=1)  # Shape: [obstacle_num, 3]
            near_obstacles_in_leader_frame = torch.cat((near_obstacles_in_leader_frame, action_u_placeholder), dim=1)
            num_obstacles = near_obstacles_in_leader_frame.size(0)
        


        # Add a category feature to each node (0 for nominal formation, 1 for obstacles)
        
        # Concatenate category feature to position tensors
        x = None
        # Combine all node features into a single tensor
        if near_obstacles_in_leader_frame != None:
            x = torch.cat((nominal_formation_tensor, near_obstacles_in_leader_frame), dim=0)  # Shape: [5 + obstacle_num, 3]
        else:
            x = nominal_formation_tensor
        # Initialize edge index and edge attributes
        edge_index = []
        edge_attr = []

        # Number of agents and obstacles
        num_agents = nominal_formation_tensor.size(0)
        
        # Connect each nominal formation agent with near obstacles
        if near_obstacles_in_leader_frame is not None:
            for agent_index in range(num_agents):
                agent_pos = nominal_formation_tensor[agent_index, :2]  # Get the position part
                for obstacle_index in range(num_obstacles):
                    if self.evaluation_noise > 0.2:
                        if random.random() < self.evaluation_noise:
                            continue
                    obstacle_pos = near_obstacles_in_leader_frame[obstacle_index, :2]  # Get the position part
                    distance = torch.norm(agent_pos - obstacle_pos)
                    if distance <= self.max_obstacle_edge_range:  # Check if within threshold distance
                        # Add edges from agent to obstacle
                        edge_index.append([agent_index, num_agents + obstacle_index])  # Agent to obstacle
                        edge_index.append([num_agents + obstacle_index, agent_index])  # Obstacle to agent
                        edge_attr.append([distance.item()])  # Edge type 1 for agent-obstacle
                        edge_attr.append([distance.item()])  # Edge type 1 for agent-obstacle

        # Connect each pair of nominal formation agents
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
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
        
        return data



    def set_last_policy_output(self, output):
        self.last_policy_output = copy.deepcopy(output)

    def observation(self, agent: Agent):
        # get_obs_time = time.time()
        goal_poses = []
        goal_rot_poses = []
        current_agent_index = self.world.agents.index(agent)
        agent_num = len(self.world.agents)
        current_agent_index = self.world.agents.index(agent)
        graph_list = []
        max_obstacles =100  # Define this attribute in your class
        num_node_features = 3  # Assuming x and y coordinates
        num_edge_features = 1  # Assuming distance as edge attribute
        num_agents = agent_num  # Number of agents per graph
        max_nodes = num_agents + max_obstacles  # Agents + Obstacles
        max_edges = num_agents * max_obstacles * 2 + (num_agents * (num_agents - 1)) * 2  # Approximate
        
        # Define feature_length:
        # - Node features: max_nodes * num_node_features
        # - Edge indices: max_edges * 2
        # - Edge attributes: max_edges * num_edge_features
        # Total feature_length = max_nodes*num_node_features + max_edges*(2 + num_edge_features)
        
        feature_length = max_nodes * num_node_features + max_edges * (2 + num_edge_features)
        # feature_length = (num_agents + max_obstacles) * num_node_features + max_obstacles * 2 + max_obstacles * num_edge_features
        if current_agent_index == 0:
            self.env_observation = []
            agent_num = len(self.world.agents)
            nominal_formation_tensor = torch.zeros((self.world.batch_dim, agent_num, 3), device=self.device)
            
            # Define nominal positions for agents (assuming 5 agents)
            # nominal_positions_x = [0.0, -0.3536, -0.3536, -0.7071, -0.7071]
            # nominal_positions_y = [0.0, 0.35366, -0.3536, 0.7071, -0.7071]
            nominal_positions_x = [0.0, -1.2, -1.2, -2.4, -2.4]
            nominal_positions_y = [0.0, 0.6, -0.6, 1.2, -1.2]
            for i, nomi_agent in enumerate(self.world.agents):
                nominal_formation_tensor[:, i, 0] = nominal_positions_x[i]
                nominal_formation_tensor[:, i, 1] = nominal_positions_y[i]
                nominal_formation_tensor[:, i, 2] = 0.0
            previous_positions = self.agent_history.get_previous_positions()  # [num_agents, history_length, 2]


            for d in range(self.world.batch_dim):
                # print("obs manager:{}".format(self.obstacle_manager_list[d]))
                obstacle_manager = self.obstacle_manager_list[d]
                leader_pos = self.leader_robot.state.pos[d, :]  # Shape: [2]
                leader_rot = self.leader_robot.state.rot[d]      # Scalar (radians)
                
                # agent_prev_positions = previous_positions[d, :, :, :] 
                # Transform previous positions relative to leader's frame
                # Translation: agent_prev_positions - leader_pos
                # translated_prev_positions = agent_prev_positions - leader_pos.unsqueeze(0).unsqueeze(1)  # [agent_num, history_length, 2]
                
                # Rotation: Apply rotation matrix
                cos_theta = torch.cos(leader_rot)
                sin_theta = torch.sin(leader_rot)
                # print("sin_theta shape:{}".format(sin_theta.shape))
                rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                                [sin_theta, cos_theta]], device=self.device)  # [2, 2]
                
                # Apply rotation to each agent's previous positions
                # Reshape for batch matrix multiplication
                # agent_prev_positions_flat = translated_prev_positions.view(-1, 2)  # [agent_num * history_length, 2]
                # transformed_prev_positions = torch.matmul(agent_prev_positions_flat, rotation_matrix)  # [agent_num * history_length, 2]
                # transformed_prev_positions = transformed_prev_positions.view(agent_num, -1, 2)  # [agent_num, history_length, 2]
                
                # # Retrieve transformed previous positions for all agents
                # transformed_prev_positions_batch = transformed_prev_positions  # [agent_num, history_length, 2]

                NUM_BINS = 90
                BIN_SIZE = 2 * math.pi / NUM_BINS  # Size of each bin in radians

                # Assume this code is inside a method where `obstacle_manager`, `leader_pos`, 
                # `rotation_matrix`, `self.max_obstacle_include_range`, and other variables are defined.

                # Retrieve nearby obstacles
                
                near_obstacles_tensor = obstacle_manager.get_near_obstacles(
                    leader_pos, 
                    self.max_obstacle_include_range
                )
                # Check if there are any nearby obstacles
                if near_obstacles_tensor.shape[0] != 0:
                    # Stack the list of obstacle tensors into a single tensor of shape [num_obstacles, 2]
                    # near_obstacles_tensor = torch.stack(near_obstacles)  # Shape: [num_obstacles, 2]
                    
                    # Translate obstacles relative to the leader's position
                    translated_obstacles = near_obstacles_tensor - leader_pos  # Shape: [num_obstacles, 2]
                    
                    # Compute polar coordinates: angles and distances
                    angles = torch.atan2(translated_obstacles[:, 1], translated_obstacles[:, 0])  # [num_obstacles]
                    angles = angles % (2 * math.pi)  # Normalize angles to [0, 2π)
                    distances = torch.norm(translated_obstacles, dim=1)  # [num_obstacles]
                    
                    # Assign each obstacle to an angular bin
                    bin_indices = (angles / BIN_SIZE).long()  # [num_obstacles]
                    bin_indices = torch.clamp(bin_indices, max=NUM_BINS - 1)  # Ensure indices are within [0, NUM_BINS-1]
                    
                    # Combine bin indices and distances to sort: primary by bin, secondary by distance
                    # Multiplying bin_indices by a large constant ensures bin sorting before distance sorting
                    combined_sort_key = bin_indices.float() * 1e6 + distances  # [num_obstacles]
                    sorted_order = torch.argsort(combined_sort_key)  # Indices that would sort the array
                    sorted_bin_indices = bin_indices[sorted_order]  # [num_obstacles] sorted by bin and distance
                    sorted_obstacles = translated_obstacles[sorted_order]  # [num_obstacles, 2]
                     
                    # Identify the first occurrence of each unique bin
                    # Since sorted_bin_indices is sorted, the first occurrence of each bin is unique
                    # Create a mask where a bin index is different from the previous one
                    if sorted_bin_indices.numel() > 0:
                        # Initialize mask with all False except the first element
                        mask = torch.zeros_like(sorted_bin_indices, dtype=torch.bool)
                        mask[0] = True  # First obstacle is always selected
                        # Compare each bin index with the previous one
                        mask[1:] = sorted_bin_indices[1:] != sorted_bin_indices[:-1]
                    else:
                        mask = torch.tensor([], dtype=torch.bool, device=sorted_bin_indices.device)
                    
                    # Select the closest obstacle in each unique bin
                    selected_obstacles = sorted_obstacles[mask]  # Shape: [num_unique_bins, 2]
                    # print("origi obs num:{}".format(len(near_obstacles)))
                    # print("selected num:{}".format(len(selected_obstacles)))
                    # Apply rotation to transform obstacles into the leader's frame
                    near_obstacles_in_leader_frame = torch.matmul(
                        selected_obstacles, 
                        rotation_matrix
                    )  # Shape: [num_unique_bins, 2]
                    
                    # Create the current graph using the filtered obstacles
                    current_graph = self.single_graph_from_data(
                        d, 
                        nominal_formation_tensor[d, :, :], 
                        near_obstacles_in_leader_frame
                    )

                    # env_obs_graph = self.single_env_observation_graph_from_data(d, 
                        # nominal_formation_tensor[d, :, :], 
                        # near_obstacles_in_leader_frame)
                    # self.env_observation.append(env_obs_graph)
                else:
                    # If no obstacles are nearby, create the graph without obstacle data
                    current_graph = self.single_graph_from_data(
                        d, 
                        nominal_formation_tensor[d, :, :], 
                        None
                    )

                
                graph_list.append(current_graph)
            # print("get obs time:{}".format(time.time() - get_obs_time))
            
            return graph_list

        
        else:
           
            observation_tensor = torch.zeros((self.world.batch_dim, feature_length), dtype=torch.float, device=self.device)
            # print("other dim shape:{}".format(observation_tensor.shape))
            return observation_tensor

    
      
    def done(self):
        
        return torch.zeros(self.batch_dim, device=self.device, dtype=torch.bool)
        # return done_status

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        Collects and returns information for the given agent, including optimized target positions
        and graph representations of the environment relative to the leader robot's frame.

        Args:
            agent (Agent): The agent for which to collect information.

        Returns:
            Dict[str, Tensor]: A dictionary containing the optimized target positions and graph data.
        """
        current_agent_index = self.world.agents.index(agent)
        graph_list = []
        
        

        # Translate agent positions relative to leader's position
        translated_agent_pos = agent.state.pos - self.leader_robot.state.pos  # Shape: [batch_dim, 2]
        translated_agent_rot = torch.zeros_like(agent.state.rot)
        # # Rotate agent positions into leader's frame
        # # Assume self.leader_robot.state.rot is a tensor of shape [batch_dim]
        cos_theta = torch.cos(self.leader_robot.state.rot).unsqueeze(-1)  # Shape: [batch_dim, 1]
        sin_theta = torch.sin(self.leader_robot.state.rot).unsqueeze(-1)  # Shape: [batch_dim, 1]
        
        # # Define rotation matrices for each batch
        rotation_matrices = torch.cat([cos_theta, sin_theta, -sin_theta, cos_theta], dim=1).reshape(-1, 2, 2)  # Shape: [batch_dim, 2, 2]
        
        # # Apply rotation: [batch_dim, 2, 2] x [batch_dim, 2, 1] -> [batch_dim, 2, 1]
        optimized_target_pos = torch.bmm(rotation_matrices, translated_agent_pos.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_dim, 2]
        optimized_target_pose = torch.cat([optimized_target_pos, translated_agent_rot], dim=1)
        # print("optimized_target_pose:{}".format(optimized_target_pose.shape))
        if current_agent_index == 0:
            return {
                # "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
                # "final_rew": self.final_rew,
                "eva_collision_num": self.eva_collision_num,
                "eva_connection_num": self.eva_connection_num,   # 1 means connected, 0 means not connected
                "env_observation": self.env_observation,
                "agent_collisions": agent.agent_collision_rew,
                "leader_x": agent.state.pos[:, 0],
                # "formation_goal": agent.goal.state.pos,
                # "formation_main_rew":self.formation_maintain_rew,
                "optimized_target_pos": optimized_target_pose,  # Transformed to leader's frame
                # "graph_list": graph_list,
                "agent_connection_rew": agent.connection_rew,
                "agent_diff_rew": agent.action_diff_rew,
                "agent_target_collision": agent.target_collision_rew,
            }
        else:
            return {
                # "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
                # "final_rew": self.final_rew,
                "agent_collisions": agent.agent_collision_rew,
                # "formation_goal": agent.goal.state.pos,
                # "formation_main_rew":self.formation_maintain_rew,
                "optimized_target_pos": optimized_target_pose,  # Transformed to leader's frame
                # "graph_list": graph_list,
                "agent_connection_rew": agent.connection_rew,
                "agent_diff_rew": agent.action_diff_rew,
                "agent_target_collision": agent.target_collision_rew,
            }
    
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        
        alpha = 1.0  # Weight for distance difference
        beta = 1.0   # Weight for angle difference
        #start

        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if i != j:
                    pos_i = agent1.state.pos[env_index]
                    pos_j = agent2.state.pos[env_index]
                    rot_i = agent1.state.rot[env_index]
                    
                    rel_pos = pos_j - pos_i
                    d_ij = torch.norm(rel_pos).item()
                    
                    if d_ij <= self.max_connection_distance:
                        # Calculate the relative angle using PyTorch operations only
                        theta_ij = torch.atan2(rel_pos[1], rel_pos[0]) - rot_i

                        # Normalize the angle to be within the range [-π, π] using PyTorch
                        theta_ij = torch.atan2(torch.sin(theta_ij), torch.cos(theta_ij))

                        if self.FOV_min <= theta_ij <= self.FOV_max:
                            # print("connect {} and {}".format(i, j))
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

        #end
        return geoms

if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
