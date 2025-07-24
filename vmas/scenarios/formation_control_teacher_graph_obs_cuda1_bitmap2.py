#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import copy
import math
import os
from PIL import Image
import time
import typing
from typing import Callable, Dict, List, Tuple
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
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box, Line, BitmapObstacle
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
        if self.is_imitation:
            self.working_mode = "imitation"
        else:
            self.working_mode = "RL"
        self.obstacle_pattern = 1
        self.env_type = kwargs.get("env_type", "narrow")
        # self.env_type = "bitmap"
        self.is_evaluation_mode = kwargs.get("is_evaluation_mode", False)
        self.evaluation_index = kwargs.get("evaluation_index", 0)

        self.lidar_range = kwargs.get("lidar_range", 5.0)
        if self.working_mode == "imitation":
            self.agent_radius = kwargs.get("agent_radius", 0.5)
        else:
            self.agent_radius = kwargs.get("agent_radius", 0.2)


        self.train_map_directory = kwargs.get("train_map_directory", "train_maps_0_clutter")
        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", False)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.2)
        self.evaluation_noise = kwargs.get("evaluation_noise", 0.0)
        self.has_laser = kwargs.get("has_laser", True)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 7
        if self.env_type == "mixed_in_distribution":
            self.world_semidim_x = 40
            self.world_semidim_y = 7
        else:
            self.world_semidim_x = 20
            self.world_semidim_y = 20
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
        self.max_connection_distance = 1.7  # Example distance threshold
        
        self.FOV_min = -0.3 * torch.pi
        self.FOV_max = 0.3 * torch.pi
        self.min_collision_distance = 0.7
        self.min_target_collision_distance = 0.3
        self.connection_reward_positive = 0.5  # Positive reward
        self.connection_reward_negative = -0.5  # Negative reward
        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -0.1)
        self.agent_velocity_target_direction_alignment_reward_weight = 0.3
        self.obstacle_center_list = []
        self.obstacle_radius_list = []
        self.route_point_list = []
        self.precomputed_route_point_list = {}
        self.env_observation = []

        self.precompute_obs_dict = {}
        self.evaluation_num = 10

        self.reached_leader_path_end =torch.zeros(batch_dim, device=self.device, dtype=torch.bool)
        self.formation_out_of_shape = torch.zeros(batch_dim, device=self.device, dtype=torch.bool)
        self.max_distance_from_follower_to_leader = 12.0
        self.current_lidar_reading = None




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
        self.current_action_mean = {}
        self.formation_goals = {}
        self.success_reconfigure_goals = {}
        self.formation_goals_landmark = {}
        self.formation_nominal_goals = {}
        for i in range(self.n_agents):   
            self.formation_goals[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            self.formation_nominal_goals[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            self.success_reconfigure_goals[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            self.last_action_u[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            self.current_action_mean[i] = torch.zeros(
                (batch_dim, 3),
                device=device
            )
            self.formation_goals_landmark[i] = Landmark(
                name=f"formation goal{i}",
                collide=False,
                movable=True,
                rotatable=True,
                color=Color.GREEN,
                renderable=True,
            )
            # self.formation_goals_landmark[i].renderable = False
            world.add_landmark(self.formation_goals_landmark[i])
            
        world.add_landmark(self.formation_center)
        world.add_landmark(self.leader_robot)

        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the parent directory and then to train_maps
        # train_maps_path = os.path.join(script_dir, "..", "train_tunnel_maps")
        if self.env_type == "bitmap":
            train_maps_path = os.path.join(script_dir, "..", self.train_map_directory)
        elif self.env_type == "bitmap_tunnel":
            train_maps_path = os.path.join(script_dir, "..", self.train_map_directory)

        # self.create_obstacles(self.obstacle_pattern, world)
        self.bitmap = self.create_bitmap_obstacle(train_maps_path, world, batch_dim)
        # print("bitmap:{}".format(self.bitmap.shape))
        def detect_obstacles(x):
            return x.name.startswith("obs_") or x.name.startswith("map") or x.name.startswith("agent_") or x.name.startswith("wall")




        #add leader agent
        self.leader_agent = Agent(
                name=f"agent_0",
                collide=self.collisions,
                color=Color.RED,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                dynamics=HolonomicWithRotation(), 
                linear_friction=0,
                max_t = 10,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=20,
                            max_range=self.lidar_range,
                            entity_filter=detect_obstacles,
                        ),
                    ]
                    if self.has_laser
                    else None
                ),
            )
        self.leader_agent.pos_rew = torch.zeros(batch_dim, device=device)
        self.leader_agent.angle_rew = torch.zeros(batch_dim, device=device)
        self.leader_agent.group_center_rew = self.leader_agent.pos_rew.clone()
        self.leader_agent.collision_obstacle_rew = self.leader_agent.pos_rew.clone()
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
                render_action=False,
                dynamics=HolonomicWithRotation(), 
                linear_friction=0,
                max_t = 10,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=20,
                            max_range=self.lidar_range / 2.0,
                            entity_filter=detect_obstacles,
                        ),
                    ]
                    if self.has_laser
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.angle_rew = torch.zeros(batch_dim, device=device)
            agent.group_center_rew = agent.pos_rew.clone()
            agent.agent_collision_rew = agent.pos_rew.clone()
            agent.collision_obstacle_rew = agent.pos_rew.clone()
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
        self.group_center_diff_rew = torch.zeros(batch_dim, device=device)
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

        self.left_opening = 0.0
        self.right_opening = 0.0
        self.LIDAR_OPENING_SMOOTHING_ALPHA = 0.15 # HYPERPARAMETER: Lower value = more smoothing. Start with 0.1-0.3

        self.smoothed_left_opening = torch.zeros(self.batch_dim, device=self.device)
        self.smoothed_right_opening = torch.zeros(self.batch_dim, device=self.device)

        self.eva_collision_num = torch.zeros(self.n_agents - 1, self.batch_dim, device=device)
        #number of agents that are connected to leader
        self.eva_connection_num = torch.zeros(self.n_agents - 1, self.batch_dim, device=device)
        self.precompute_evaluation_scene(self.env_type)


        self.TUNNEL_COMMIT_STEPS = 30 # Hyperparameter: tune based on robot speed and environment scale

        # History buffer to store the last N environment types for each parallel env.
        # Shape: [batch_dim, history_length]
        # We initialize it with a non-tunnel type (e.g., 1 for clutter).
        self.env_type_history = torch.full(
            (self.batch_dim, self.TUNNEL_COMMIT_STEPS), 
            fill_value=0, # Default to "clutter"
            device=self.device, 
            dtype=torch.long
        )
        self.env_type_history_enter_tunnel = torch.full(
            (self.batch_dim, 10), 
            fill_value=1, # Default to "clutter"
            device=self.device, 
            dtype=torch.long
        )
        self.OA_PREDICT_TIME = 0.7              # seconds, how far to look ahead
        self.OA_SIMULATION_TIMESTEP = 0.1       # seconds, resolution of the simulated path
        self.OA_ROBOT_RADIUS = 0.25             # meters, for collision checking
        self.OA_NUM_ANGULAR_SEARCH_STEPS = 10   # Number of alternative angles to check on each side (left/right)
        
        # --- Robot Dynamic Constraints ---
        self.OA_MAX_SPEED_X = 0.7  # m/s
        self.OA_MAX_SPEED_Y = 0.2  # m/s
        self.OA_MAX_SPEED_YAW = 0.6 # rad/s
        self.footprint_local = self._get_robot_footprint()
        self.OA_PREDICT_TIME = 0.7              # seconds, how far to look ahead
        self.OA_NUM_VELOCITY_SEARCH_STEPS = 3   # Number of steps to reduce forward velocity
        
        # --- Robot Dynamic Constraints ---
        self.OA_MAX_SPEED_X = 0.5  # m/s
        self.OA_MAX_SPEED_Y = 0.3  # m/s
        self.OA_MAX_SPEED_YAW = 1.0 # rad/s

        return world

    def load_map(self, file_path: str, obstacle_threshold: int = 128) -> torch.Tensor:
        """
        Load a bitmap image and convert it to a binary obstacle grid.
        
        Args:
            file_path: Path to the image file
            obstacle_threshold: Pixel value threshold for considering a cell blocked (0-255)
        
        Returns:
            Tensor: Binary matrix where 1 represents obstacles
        """
        # Load image and convert to grayscale
        img = Image.open(file_path).convert('L')
        
        # Convert to numpy array and threshold
        map_array = np.array(img)
        binary_map = (map_array >= obstacle_threshold).astype(np.float32)
        
        # Convert to PyTorch tensor and add dimension for channels
        return torch.from_numpy(binary_map).to(torch.float32)

    def create_bitmap_obstacle(self, bitmap_folder, world, map_num=1):
        """
        Creates a single BitmapObstacle by sampling multiple maps from a folder
        and batching them together.
        
        Args:
            bitmap_folder: Path to the folder containing bitmap maps
            world: The world where obstacle will be created
            map_num: Number of different maps to sample
            
        Returns:
            A BitmapObstacle instance with batched maps
        """
        import os
        import random
        
        # Check if the folder exists
        if not os.path.exists(bitmap_folder):
            raise FileNotFoundError(f"Bitmap folder not found: {bitmap_folder}")
        
        # Get list of all image files in the folder
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        bitmap_files = [
            os.path.join(bitmap_folder, f) for f in os.listdir(bitmap_folder)
            if os.path.isfile(os.path.join(bitmap_folder, f)) and 
            any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        
        # Check if there are enough maps
        if len(bitmap_files) < map_num:
            raise ValueError(f"Not enough maps in folder. Requested {map_num}, but only found {len(bitmap_files)}")
        
        # Randomly sample map_num maps
        selected_maps = random.sample(bitmap_files, map_num)
        
        # Process all selected maps into binary arrays
        binary_bitmaps = []
        
        for map_file in selected_maps:
            # Load the bitmap as a grayscale image

            torch.tensor(self.load_map(map_file))

            bitmap_img = Image.open(map_file).convert('L')
            bitmap_array = np.array(bitmap_img)
            binary_map = (bitmap_array >= 128).astype(np.float32)
            # Convert to binary (0 for free space, 1 for obstacles)
            binary_bitmaps.append(binary_map)
        
        # Check if all bitmaps have the same shape
        first_shape = binary_bitmaps[0].shape
        if not all(bitmap.shape == first_shape for bitmap in binary_bitmaps):
            # Resize all bitmaps to the shape of the first one if needed
            raise ValueError(f"Not all maps same size")
        
        # Convert to pytorch tensor
        binary_bitmaps_np = np.stack(binary_bitmaps, axis=0)
        bitmap_tensor = torch.tensor(binary_bitmaps_np, dtype=torch.float, device=world.device)
        
        # If map_num < batch_dim, repeat the maps to match batch_dim
        if map_num < world.batch_dim:
            # Calculate how many full repetitions we need
            full_repeats = world.batch_dim // map_num
            remainder = world.batch_dim % map_num
            
            # Repeat the tensor for full repetitions
            bitmap_tensor_repeated = bitmap_tensor.repeat(full_repeats, 1, 1)
            
            # Add the remainder if necessary
            if remainder > 0:
                bitmap_tensor_remainder = bitmap_tensor[:remainder]
                bitmap_tensor = torch.cat([bitmap_tensor_repeated, bitmap_tensor_remainder], dim=0)
        elif map_num > world.batch_dim:
            # If we have more maps than batch_dim, just use the first batch_dim maps
            bitmap_tensor = bitmap_tensor[:world.batch_dim]
        # print("bitmap_tensor shape:{}".format(bitmap_tensor.shape))
        # Create a single bitmap obstacle with the batched tensor
        origin = [-12.8, -12.8]  # Example origin coordinates (bottom-left corner)
        resolution = 0.1       # Example resolution (meters per pixel)
        
        bitmap_obstacle = BitmapObstacle(
            name="map",
            bitmap=bitmap_tensor,
            resolution=resolution,
            origin=origin
        )
        world.add_landmark(bitmap_obstacle)
        
        return bitmap_obstacle

    # def create_bitmap_obstacle(self, bitmap_folder, world: World):
    #     batch_dim = 1
    #     bitmaps = torch.stack([
    #         torch.tensor(self.load_map(bitmap_filename))
    #             # Your loading function
    #         # torch.tensor(self.load_map("map2.png"))
    #     ], dim=0)
    #     print("bitmaps shape:{}".format(bitmaps.shape))
    #     # input("1")
    #     obstacle = BitmapObstacle(
    #         name="map",
    #         bitmap=bitmaps,
    #         resolution=0.1,  # 10cm per pixel
    #         origin=(-12.8, -12.8)  # Map covers [-5,5] in both axes
    #     )
    #     # input("2")
    #     world.add_landmark(obstacle)
    #     return obstacle

    

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
                # print("door_Y:{}".format(door_y))
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
                    # print("center device:{}".format(center.device))
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
                # print("obs num:{}".format(len(self.obstacles)))

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
                # print("obs num:{}".format(len(self.obstacles)))

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


    def reset_world_at_with_init_direction(self, env_index, init_direction):
        """
        Reset the world state for a specific environment index with a specified initial direction.
        
        Args:
            env_index (int): The index of the environment to reset
            init_direction (float or torch.Tensor): The initial heading direction in radians
        """
        # Ensure the direction is a tensor with the proper device
        if not torch.is_tensor(init_direction):
            init_direction = torch.tensor(init_direction, device=self.device)
        
        # Reset timestep counter for this environment
        if hasattr(self, 't'):
            self.t = 0
        
        # Position the leader agent at the origin
        self.leader_agent.set_pos(
            torch.tensor([0, 0], device=self.device),
            batch_index=env_index,
        )
        
        # Set the formation center to match the leader's position
        self.formation_center.set_pos(
            torch.tensor([0, 0], device=self.device),
            batch_index=env_index,
        )
        
        # Set the formation center position record
        self.formation_center_pos[env_index, 0] = 0
        self.formation_center_pos[env_index, 1] = 0
        
        # Set the leader's rotation to the initial direction
        self.leader_agent.set_rot(init_direction, batch_index=env_index)
        self.formation_center.set_rot(init_direction, batch_index=env_index)
        self.formation_center_pos[env_index, 2] = init_direction
        
        # Calculate velocity components based on the initial direction
        # vel_x = torch.cos(init_direction) * 0.8  # Maintain speed of 0.8
        # vel_y = torch.sin(init_direction) * 0.8
        
        # # Set the leader's velocity in the direction of init_direction
        # self.leader_agent.set_vel(
        #     torch.tensor([vel_x, vel_y], device=self.world.device),
        #     batch_index=env_index,
        # )
        
        # # Set angular velocity to zero
        # self.leader_agent.set_ang_vel(
        #     torch.tensor(0, device=self.world.device),
        #     batch_index=env_index,
        # )
        nominal_positions_x = [0.0, -1.2, -1.2, -2.4, -2.4]
        nominal_positions_y = [0.0, 0.6, -0.6, 1.2, -1.2]
        # Reset follower agents to their initial positions in the formation
        # These positions need to be rotated according to the leader's new orientation
        for i, agent in enumerate(self.world.agents):
            if i == 0:  # Skip the leader agent (already set)
                continue
                
            # Get the nominal formation position for this agent (relative to leader)
            nominal_x = nominal_positions_x[i]
            nominal_y = nominal_positions_y[i]
            
            # Rotate the nominal position by init_direction
            cos_rot = torch.cos(init_direction)
            sin_rot = torch.sin(init_direction)
            rotated_x = nominal_x * cos_rot - nominal_y * sin_rot
            rotated_y = nominal_x * sin_rot + nominal_y * cos_rot
            
            # Set the agent's position (leader at origin + rotated offset)
            agent.set_pos(
                torch.tensor([rotated_x, rotated_y], device=self.device),
                batch_index=env_index,
            )
            
            # Set the agent's rotation to match the leader's rotation
            agent.set_rot(init_direction, batch_index=env_index)
            
            # Initialize velocity and angular velocity to zero
            agent.set_vel(
                torch.tensor([0.0, 0.0], device=self.device),
                batch_index=env_index,
            )
            agent.set_ang_vel(
                torch.tensor(0, device=self.device),
                batch_index=env_index,
            )
        
  


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
        
        self.leader_robot.set_pos(
                    torch.tensor(
                        [
                            0.0, 
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
                            0.0, 
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
                            0.0, 
                            0.0,
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif i == 1:
                agent.set_pos(
                    torch.tensor(
                        [
                            -0.7, 
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
                            -0.7,  
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
                            -0.9,
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
                            -0.9,
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

        self.get_leader_paths()

        
        # self.spawn_obstacles(self.obstacle_pattern, env_index)
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

    def set_action_mean(self, actions):
        print("actions mean shape:{}".format(actions.shape))
        #actions mean shape:torch.Size([20, 5, 3])
        for i in range(self.n_agents):   
            self.current_action_mean[i] = actions[:, i, :]
            print("self last_actionu shape:{}, {}".format(i, self.current_action_mean[i].shape))
        # self.last_action_u

    def _is_path_near_obstacles(self, path, bitmap, check_radius_pixels, min_near_points_ratio=0.1):
        """
        Checks if a path has a sufficient portion of its points near an obstacle.

        Args:
            path (list): The path as a list of (y, x) coordinates.
            bitmap (np.array): The environment bitmap where 0 represents an obstacle.
            check_radius_pixels (int): The radius (in pixels) around each waypoint to check for obstacles.
            min_near_points_ratio (float): The minimum ratio of waypoints that must be "near" an obstacle
                                           for the path to be considered valid.

        Returns:
            bool: True if the path is sufficiently near obstacles, False otherwise.
        """
        if not path:
            return False

        near_obstacle_point_count = 0
        height, width = bitmap.shape

        for y, x in path:
            # Define a square window to check around the current waypoint
            y_min = max(0, y - check_radius_pixels)
            y_max = min(height, y + check_radius_pixels + 1)
            x_min = max(0, x - check_radius_pixels)
            x_max = min(width, x + check_radius_pixels + 1)

            window = bitmap[y_min:y_max, x_min:x_max]
            
            # If any obstacle (value 0) is found in the window, this point is "near"
            if np.any(window == 0):
                near_obstacle_point_count += 1
        
        # Check if the ratio of "near" points meets the minimum requirement
        if len(path) > 0 and (near_obstacle_point_count / len(path)) >= min_near_points_ratio:
            return True
        
        return False

    def get_forward_env_type(self):
        instantaneous_env_type = torch.ones(self.batch_dim, device=self.device, dtype=torch.long)

        is_wide_open = (self.smoothed_left_opening > 1.1) & (self.smoothed_right_opening > 1.1)
        is_narrow = (self.smoothed_left_opening < 0.6) & (self.smoothed_right_opening < 0.6)

        instantaneous_env_type[is_wide_open] = 0
        instantaneous_env_type[is_narrow] = 2
        
        # --- Step 2: Check history to identify "Deep Tunnel" state ---
        # Check if the last `TUNNEL_COMMIT_STEPS` in the history were all type 2.
        # This creates a boolean mask of shape [batch_dim].
        is_deep_in_tunnel = (self.env_type_history == 2).all(dim=1)
        is_short_in_tunnel = (self.env_type_history_enter_tunnel == 2).all(dim=1)
        # --- Step 3: Generate the final output type ---
        # Start with the instantaneous type.
        final_env_type = instantaneous_env_type.clone()
        
        # Where the "deep in tunnel" condition is met, override the type to 3.
        # This handles the transition from 2 -> 3.
        final_env_type[is_short_in_tunnel] = 2
        final_env_type[is_deep_in_tunnel] = 3
        
        # --- Step 4: Update the history buffer for the next time step ---
        # Shift the history one step to the left (discarding the oldest entry).
        self.env_type_history = torch.roll(self.env_type_history, shifts=-1, dims=1)
        self.env_type_history_enter_tunnel = torch.roll(self.env_type_history, shifts=-1, dims=1)

        
        # Add the new *instantaneous* type to the end of the history.
        # We use the instantaneous type so the history reflects the raw perception,
        # allowing the "is_deep_in_tunnel" condition to trigger correctly in the future.
        self.env_type_history[:, -1] = instantaneous_env_type
        self.env_type_history_enter_tunnel[:, -1] = instantaneous_env_type

        return final_env_type






        
    def get_leader_paths(self, max_trials=1000):
        """
        Find a path for the leader agent to a randomly selected target point in the bitmap.
        
        Args:
            leader_agent: The leader agent entity
            bitmap: The bitmap representing the environment (1 for obstacles, 0 for free space)
            origin: The (x,y) world coordinates corresponding to bitmap[0,0]
            scale: The size of each bitmap cell in world units
            max_trials: Maximum number of attempts to find a valid path
            
        Returns:
            list: A list of waypoints forming a path, or empty list if no path found
        """
        import numpy as np
        import random
        import torch
        self.batch_leader_paths = []
        
        for dim in range(self.batch_dim):
            # print("compute path {}".format(dim))
            print("get_leader path dim:{}".format(dim))
            bitmap = self.bitmap.bitmap[dim, :]
            origin = [-12.8, -12.8]  # Example origin coordinates (bottom-left corner)
            scale = 0.1
            # Convert bitmap to numpy if it's a torch tensor
            if isinstance(bitmap, torch.Tensor):
                bitmap = bitmap.cpu().numpy()

            # print("bitmap shape:{}".format(bitmap.shape))
            # input("get_paths")
            # Get initial position of the leader agent (from the first batch)
            start_pos = self.leader_agent.state.pos[dim].cpu().numpy()
            # print("start_pos shape:{}".format(start_pos.shape))
            # print("start_pos:{}".format(start_pos))
            agent_radius = self.leader_agent.shape.radius
            
            # Convert to bitmap coordinates
            bitmap_height, bitmap_width = bitmap.shape
            
            # Convert world position to bitmap coordinates
            start_x = int((start_pos[0] - origin[0]) / scale)
            start_y = int((start_pos[1] - origin[1]) / scale)
            start_coord = (start_y, start_x)
            
            # Make sure start position is valid
            if not self._is_valid_point(start_coord, bitmap) or not self._is_safe_point(start_coord, bitmap, agent_radius / scale):
            
                # print("Leader's starting position is invalid or too close to obstacles")
                # input("Leader's starting position is invalid or too close to obstacles")
                return []
            

            inflation_radius = 0.0
            if self.train_map_directory == "train_maps_0_clutter":
                inflation_radius = 3.0
            elif self.train_map_directory == "train_maps_1_clutter":
                inflation_radius = 2.5
            elif self.train_map_directory == "train_maps_2_clutter":
                inflation_radius = 2.5
            elif self.train_map_directory == "train_maps_3_clutter":
                inflation_radius = 2.5
            elif self.train_map_directory == "train_maps_4_clutter":
                inflation_radius = 2.0
            elif self.train_map_directory == "train_maps_5_clutter":
                inflation_radius = 0.5

            # if self.train_map_directory == "train_maps_0_clutter":
            #     inflation_radius = 0.5
            # elif self.train_map_directory == "train_maps_1_clutter":
            #     inflation_radius = 0.5
            # elif self.train_map_directory == "train_maps_2_clutter":
            #     inflation_radius = 0.5
            # elif self.train_map_directory == "train_maps_3_clutter":
            #     inflation_radius = 0.5
            # elif self.train_map_directory == "train_maps_4_clutter":
            #     inflation_radius = 0.5
            # elif self.train_map_directory == "train_maps_5_clutter":
            #     inflation_radius = 0.5



            # Try finding a path for max_trials attempts
            for trial in range(max_trials):
                # Randomly pick a target point
                print("trial:{}".format(trial))
                target_found = False
                for _ in range(1000):  # Try 100 random points before giving up on this trial
                    if self.env_type == "bitmap_tunnel":
                        target_x = int(bitmap_width - 10)
                        target_y = int(bitmap_height / 2.0)
                    else:
                        target_x = random.randint(0, bitmap_width - 1)
                        target_y = random.randint(0, bitmap_height - 1)

                    if target_x > int(bitmap_width / 3.0) and target_x < int(bitmap_width / 3.0 * 2.0) and target_y > int(bitmap_height / 3.0) and target_y < int(bitmap_height / 3.0 * 2.0):
                        continue
                    # if random.random() < 0.5:
                    #     target_x = random.randint(0, int(bitmap_width / 3.0))
                    # else:
                    #     target_x = random.randint(int(bitmap_width / 3.0 * 2.0), bitmap_width - 1)
                    
                    # if random.random() < 0.5:
                    #     target_y = random.randint(0, int(bitmap_width / 3.0))
                    # else:
                    #     target_y = random.randint(int(bitmap_width / 3.0 * 2.0), bitmap_width - 1)

                    # target_x = random.randint(0, bitmap_width - 1)
                    # target_y = random.randint(0, bitmap_height - 1)
                    target_coord = (target_y, target_x)
                    
                    # Ensure target point is valid and safe


                    if self._is_valid_point(target_coord, bitmap) and self._is_safe_point(target_coord, bitmap, agent_radius / scale + inflation_radius):
                        target_found = True
                        break
                        
                if not target_found:
                    # No valid target found in 100 attempts, try next trial
                    continue
                # print("safe buffer:{}".format(agent_radius / scale))
                # Find path using A* algorithm with safety buffer for agent radius
                path = self._find_path_a_star(start_coord, target_coord, bitmap, safety_radius=(agent_radius / scale + inflation_radius))
                
                # If path found, simplify it and convert back to world coordinates
                if path:
                    if self._is_path_near_obstacles(path, bitmap, check_radius_pixels=30, min_near_points_ratio=0.03):
                        # Simplify path (remove unnecessary waypoints)
                        simplified_path = self._simplify_path(path, bitmap, agent_radius / scale)
                        
                        # Convert to world coordinates
                        world_path = []
                        path_idx = 0
                        for y, x in path:
                            if path_idx > 10:
                                world_x = x * scale + origin[0] - 0.03 +  random.random()*0.06
                                world_y = y * scale + origin[1] - 0.03 +  random.random()*0.06
                            else:
                                world_x = x * scale + origin[0] 
                                world_y = y * scale + origin[1] 
                            world_path.append(torch.tensor([world_x, world_y], device=self.leader_agent.state.pos.device))
                            path_idx += 1
                        # print("world_path:{}".format(world_path))
                        # input("got path for dim {}".format(dim))
                        self.batch_leader_paths.append(world_path)

                        target_found = True
                        print("path found")
                        break
                    else:
                        continue
            if target_found == False:
            # If reached here, no path found after max_trials
                print(f"No valid path found after {max_trials} attempts")
                # input("No valid path found")
                self.batch_leader_paths.append([])
        
        
        # print("batch_leader_paths size:{}".format(len(self.batch_leader_paths)))
        for dim in range(self.batch_dim):
            # print("dim:{}".format(dim))
            init_direction = self.compute_leader_init_direction(self.batch_leader_paths[dim])
            self.reset_world_at_with_init_direction(dim, init_direction)
        

        return self.batch_leader_paths
    

    def compute_leader_init_direction(self, world_path):
        """
        Compute the initial direction for the leader agent based on the first segment of the path.
        
        Args:
            world_path: List of waypoints in world coordinates
            
        Returns:
            torch.Tensor: Initial heading angle in radians in the world frame
        """
        # Default direction if we can't compute a proper one
        default_angle = torch.tensor(0.0, device=self.device)
        
        # Check if we have at least two waypoints to determine direction
        if not world_path or len(world_path) < 2:
            return default_angle
        
        # Get the first two waypoints
        first_waypoint = world_path[0]
        second_waypoint = world_path[1]
        
        # Calculate the vector from first to second waypoint
        dx = second_waypoint[0] - first_waypoint[0]
        dy = second_waypoint[1] - first_waypoint[1]
        
        # Calculate the angle in world frame (atan2 gives angle in range [-π, π])
        # Use torch.atan2 to maintain compatibility with tensors
        if torch.is_tensor(dx) and torch.is_tensor(dy):
            angle = torch.atan2(dy, dx)
        else:
            angle = torch.tensor(math.atan2(dy, dx), device=self.device)
        
        return angle

    def _is_valid_point(self, coord, bitmap):
        """Check if point is valid (in bounds and not on obstacle)"""
        y, x = coord
        # print("valid point check:y:{}, x:{}, value:{}".format(y, x, bitmap[y,x]))
        if 0 <= y < bitmap.shape[0] and 0 <= x < bitmap.shape[1]:
            return bitmap[y, x] == 1.0  # 1 means no obstacle
        return False

    def _is_safe_point(self, coord, bitmap, safety_radius):
        """Check if point is far enough from obstacles"""
        import numpy as np
        
        y, x = coord
        radius_int = int(np.ceil(safety_radius))
        
        # Check all points within safety_radius
        for dy in range(-radius_int, radius_int + 1):
            for dx in range(-radius_int, radius_int + 1):
                # Skip if outside the distance threshold
                if dx**2 + dy**2 > safety_radius**2:
                    continue
                    
                check_y, check_x = y + dy, x + dx
                # print("value:{}".format(bitmap[check_y, check_x]))
                if 0 <= check_y < bitmap.shape[0] and 0 <= check_x < bitmap.shape[1]:
                    if bitmap[check_y, check_x] == 0.0:  # Found obstacle within radius
                        return False
        
        return True

    def _find_path_a_star(self, start, goal, bitmap, safety_radius=0):
        """A* pathfinding algorithm to find path from start to goal, avoiding obstacles by safety_radius"""
        import heapq
        import numpy as np
        
        def heuristic(a, b):
            # Euclidean distance
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        # Define possible movements (8 directions)
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # The set of visited nodes that need not be expanded
        closed_set = set()
        
        # The set of tentative nodes to be evaluated
        open_set = []
        heapq.heappush(open_set, (0, start))
        open_set_dict = {start: 0}  # For quick look-up
        
        # Dictionary to track most efficient previous step
        came_from = {}
        
        # Cost from start along best known path
        g_score = {start: 0}
        
        # Estimated total cost from start to goal through y
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            for dy, dx in neighbors:
                neighbor = (current[0] + dy, current[1] + dx)
                
                if neighbor in closed_set:
                    continue

                    
                if not self._is_valid_point(neighbor, bitmap) or not self._is_safe_point(neighbor, bitmap, safety_radius):
                    continue
                    
                # Distance between current and neighbor (use Euclidean distance)
                movement_cost = np.sqrt(dx*dx + dy*dy)
                tentative_g_score = g_score[current] + movement_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score_value = g_score[neighbor] + heuristic(neighbor, goal)
                    f_score[neighbor] = f_score_value
                    
                    # Add to open set if not already there
                    if neighbor not in open_set_dict or open_set_dict[neighbor] > f_score_value:
                        heapq.heappush(open_set, (f_score_value, neighbor))
                        open_set_dict[neighbor] = f_score_value
                        
        # No path found
        return []

    def _simplify_path(self, path, bitmap, safety_radius=0):
        """Simplify path by removing redundant waypoints while maintaining safety"""
        if len(path) <= 2:
            return path
            
        simplified = [path[0]]  # Start with the first point
        
        i = 0
        while i < len(path) - 1:
            # Try to find the furthest point that can be reached directly from current point
            max_reachable = i + 1
            
            for j in range(i + 2, len(path)):
                if self._line_of_sight(path[i], path[j], bitmap, safety_radius):
                    max_reachable = j
                else:
                    break
                    
            simplified.append(path[max_reachable])
            i = max_reachable
            
        return simplified

    def _line_of_sight(self, p1, p2, bitmap, safety_radius=0):
        """Check if there is a clear line of sight between p1 and p2"""
        y1, x1 = p1
        y2, x2 = p2
        
        # Bresenham's line algorithm to check all cells along the line
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while x1 != x2 or y1 != y2:
            if not self._is_valid_point((y1, x1), bitmap) or not self._is_safe_point((y1, x1), bitmap, safety_radius):
                return False
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
        return True




    def compute_path_following_controls(self, agent):
        """
        Compute control signals for the leader agent to follow its path.
        
        Returns:
            vel_x: Velocity in x direction (world frame)
            vel_y: Velocity in y direction (world frame)
            ang_vel: Angular velocity
        """
        # Initialize output tensors for batch processing
        batch_size = agent.state.pos.shape[0]
        vel_x = torch.zeros(batch_size, device=self.device)
        vel_y = torch.zeros(batch_size, device=self.device)
        ang_vel = torch.zeros(batch_size, device=self.device)
        
        # Initialize waypoint tracking if not already done
        if not hasattr(self, 'current_waypoint_indices'):
            self.current_waypoint_indices = [0] * batch_size
        
        # Process each environment in the batch
        for env_idx in range(batch_size):
            # Check if we have a path for this environment
            if env_idx < len(self.batch_leader_paths) and self.batch_leader_paths[env_idx]:
                # Get the current path for this environment
                path = self.batch_leader_paths[env_idx]
                
                # Skip if path is empty
                if not path:
                    continue
                    
                # Get current waypoint index
                waypoint_idx = self.current_waypoint_indices[env_idx]
                # print("waypoint_idx:{}, path length:{}".format(waypoint_idx, len(path)))
                # If we've reached the end of the path, stop
                if waypoint_idx >= (len(path)-2):
                    self.reached_leader_path_end[env_idx] = True
                    print("reached end of the path")
                    continue
                
                # Get the current waypoint
                target_waypoint = path[waypoint_idx]
                if torch.is_tensor(target_waypoint):
                    target_pos = target_waypoint[:2]
                else:
                    target_pos = torch.tensor(target_waypoint[:2], device=self.device)
                
                # Get the leader's current position and orientation
                leader_pos = agent.state.pos[env_idx, :2]
                leader_rot = agent.state.rot[env_idx]
                # print("target_pos:{}".format(target_pos))
                # print("leader_pos:{}".format(leader_pos))
                # print("leader_rot:{}".format(leader_rot))

                # Calculate vector to target
                vec_to_target = target_pos - leader_pos
                distance_to_target = torch.norm(vec_to_target)
                
                # If close enough to the current waypoint, move to the next one
                waypoint_threshold = 0.2  # Distance to consider waypoint reached
                # print("distance_to_target:{}".format(distance_to_target))
                if distance_to_target < waypoint_threshold:
                    self.current_waypoint_indices[env_idx] = min(waypoint_idx + 1, len(path) - 1)
                    
                    # If we just moved to the last waypoint, check if we should stop
                    if self.current_waypoint_indices[env_idx] == len(path) - 1:
                        final_waypoint = path[-1]
                        if torch.is_tensor(final_waypoint):
                            final_pos = final_waypoint[:2]
                        else:
                            final_pos = torch.tensor(final_waypoint[:2], device=self.device)
                        
                        # If very close to final position, stop completely
                        final_distance = torch.norm(final_pos - leader_pos)
                        if final_distance < waypoint_threshold / 2:
                            print("very close to final position")
                            vel_x[env_idx] = 0.0
                            vel_y[env_idx] = 0.0
                            ang_vel[env_idx] = 0.0
                            continue
                    
                    # Continue to the next waypoint in the next iteration
                    continue
                
                # Calculate desired heading angle to the waypoint
                desired_angle = torch.atan2(vec_to_target[1], vec_to_target[0])
                
                # Calculate the angle difference (error)
                angle_diff = self.angle_normalize(desired_angle - leader_rot)
                
                # Set angular velocity proportional to the angle difference
                angular_gain = 1.5  # Proportional gain for turning
                ang_vel[env_idx] = angular_gain * angle_diff
                
                # Set linear velocity based on distance and angle alignment
                max_speed = 0.4  # Maximum speed
                
                # Reduce speed when not well-aligned with the target
                alignment_factor = torch.cos(angle_diff)
                alignment_factor = torch.clamp(alignment_factor, 0.3, 1.0)  # Between 0.3 and 1
                
                # When the angle difference is large, focus on turning first
                if abs(angle_diff) > 0.5:  # About 30 degrees
                    speed = max_speed * 0.5  # Move slower when turning sharply
                else:
                    speed = max_speed * alignment_factor
                random_factor = random.random()
                # Set velocity components in world frame
                vel_x[env_idx] = speed *random_factor* torch.cos(leader_rot + angle_diff * 0.5)
                vel_y[env_idx] = speed *random_factor* torch.sin(leader_rot + angle_diff * 0.5)
                
                # vel_x[env_idx] = 0.01
                # vel_y[env_idx] = 0.0
                # print("vel_x:{}".format(vel_x[env_idx]))
                # print("vel_y:{}".format(vel_y[env_idx]))
                # print("ang vel:{}".format(ang_vel[env_idx]))
        return vel_x, vel_y, ang_vel

    def angle_normalize(self, angle):
        """
        Normalize an angle to [-pi, pi]
        """
        return torch.atan2(torch.sin(angle), torch.cos(angle))




    def process_action(self, agent: Agent):
        self.velocity_limit = 0.8 # Adjusted speed for smoother movement
        is_first = agent == self.world.agents[0]
        if is_first:
            current_positions = torch.stack([agent.state.pos for agent in self.world.agents])  # [num_agents, batch_dim, 2]
            current_positions = current_positions.permute(1, 0, 2)  # [batch_dim, num_agents, 2]
            self.agent_history.update(current_positions)
         
          
            vel_x, vel_y, ang_vel = self.compute_path_following_controls(agent)
            vel = torch.stack([vel_x, vel_y], dim=-1)  # [batch_dim, num_agents, 2]
            
            # Apply the computed controls to the action
            
            
            # print("vel:{}".format(vel))
            for dim in range(self.world.batch_dim):
                # print("vel dim:{}".format(vel.shape))
                # print("ang_vel dim:{}".format(ang_vel.shape))

                self.leader_agent.set_vel(
                    vel[dim, :],
                    batch_index=dim,
                )
                self.leader_agent.set_ang_vel(
                    ang_vel[dim],
                    batch_index=dim,
                )
                self.formation_center_pos[dim, 0] = self.leader_agent.state.pos[dim, 0]
                self.formation_center_pos[dim, 1] = self.leader_agent.state.pos[dim, 1]
                self.formation_center_pos[dim, 2] = self.leader_agent.state.rot[dim, 0]

        

            

                
            # self.agent_target_pos_global = self.transform_local_to_global(agent.action.u, self.leader_robot.state.pos, self.leader_robot.state.rot, self.device)
            # print("agent_target_pos_global:{}".format(self.agent_target_pos_global))
            # self.agent_target_pos_global = self.agent_target_pos_global.squeeze(dim=0)
        # angles = [-135/180.0*math.pi, 135/180.0*math.pi, -135/180.0*math.pi,  135/180.0*math.pi]
        # dists = [-0.8, -0.8, -1.6, -1.6]
        # angles = [0.0, 0.0, 0.0, 0.0, 0.0]
        angles = [0.0, -26/180.0*math.pi, 26/180.0*math.pi, -26/180.0*math.pi,  26/180.0*math.pi]
        # angles = [0.0, -18/180.0*math.pi, 18/180.0*math.pi, -18/180.0*math.pi,  18/180.0*math.pi]

        # dists = [0.0, -1.34, -1.34, -2.68, -2.68]
        dists = [0.0, -1.34, -1.34, -2.68, -2.68]

        for i, world_agent in enumerate(self.world.agents):

            if agent == world_agent:
                if i == 0:
                    continue
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

                    self.formation_nominal_goals[i][:, 0] = self.formation_center_pos[:, 0] + torch.cos(formation_angle) * dist
                    self.formation_nominal_goals[i][:, 1] = self.formation_center_pos[:, 1] + torch.sin(formation_angle) * dist
                    self.formation_nominal_goals[i][:, 2] = self.formation_center_pos[:, 2] 

                    if self.working_mode == "RL": 
                        agent_target_pose_global = self.transform_local_to_global(agent.action.u, agent.state.pos, agent.state.rot, self.device)

                        # agent_target_pose_global = self.transform_local_to_global(agent.action.u, self.leader_agent.state.pos, self.leader_agent.state.rot, self.device)
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
                        if self.env_type == "door" or self.env_type == "door_and_narrow" or self.env_type == "bitmap1":
                            threshold = 0.3
                            close_to_goal = (abs(position_error_x) < threshold) & (abs(position_error_y) < threshold)
                            # far_to_door = (abs(world_agent.state.pos[:, 0] - self.door_x) > 1)
                            if i == 3 :

                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[4].state.pos[:, 0]*0.35)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[4].state.pos[:, 1]*0.35)
                                # self.formation_goals[i][:, 0] = torch.where(far_to_door, self.formation_goals[i][:, 0], 
                                #                 self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[4].state.pos[:, 0]*0.35)
                                # self.formation_goals[i][:, 1] = torch.where(far_to_door, self.formation_goals[i][:, 1], 
                                #                 self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[4].state.pos[:, 1]*0.35)
                            elif i == 4:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[3].state.pos[:, 0]*0.35)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[3].state.pos[:, 1]*0.35)
                                # self.formation_goals[i][:, 0] = torch.where(far_to_door, self.formation_goals[i][:, 0], 
                                #                 self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[3].state.pos[:, 0]*0.35)
                                # self.formation_goals[i][:, 1] = torch.where(far_to_door, self.formation_goals[i][:, 1], 
                                #                 self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[3].state.pos[:, 1]*0.35)
                            elif i == 1:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[2].state.pos[:, 0]*0.35)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[2].state.pos[:, 1]*0.35)
                            elif i == 2:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.35 + self.world.agents[1].state.pos[:, 0]*0.35)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.35 + self.world.agents[1].state.pos[:, 1]*0.35)
                            else:
                                self.formation_goals[i][:, 0] = torch.where(close_to_goal, self.formation_goals[i][:, 0], 
                                                self.formation_goals[i][:, 0]*0.3 + world_agent.state.pos[:, 0]*0.65 + self.world.agents[0].state.pos[:, 0]*0.05)
                                self.formation_goals[i][:, 1] = torch.where(close_to_goal, self.formation_goals[i][:, 1], 
                                                self.formation_goals[i][:, 1]*0.3 + world_agent.state.pos[:, 1]*0.65 + self.world.agents[0].state.pos[:, 1]*0.05)
                        
                        elif self.env_type == "bitmap" or self.env_type == "bitmap_tunnel":
                            pass
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
                            expected_vel = self.compute_agent_velocity(agent, i)
                            # agent.set_vel(
                            # self.compute_agent_velocity(agent, i),
                            # batch_index=None,
                            # )
                            agent.state.force = 3*(expected_vel - agent.state.vel[:, :2])

                            # limit due to accelaration of real robot. mass = 1kg, 
                            agent.state.force = torch.clamp(agent.state.force, min=-1, max=1)
                            
                            raw_angle_difference = self.formation_goals[i][:, 2] - agent.state.rot[:,0]

                            # 2. Normalize the angle difference to the range [-pi, pi]
                            # Use torch.pi if your PyTorch version supports it (>= 1.7.0), otherwise math.pi
                            pi_tensor = torch.tensor(torch.pi if hasattr(torch, 'pi') else math.pi, device=raw_angle_difference.device, dtype=raw_angle_difference.dtype)
                            normalized_angle_difference = torch.remainder(raw_angle_difference + pi_tensor, 2 * pi_tensor) - pi_tensor

                            # 3. Calculate expected_ang_vel using the normalized difference
                            expected_ang_vel = torch.stack([0.2 * normalized_angle_difference], dim=-1)

                            # 4. Calculate torque (and optionally clamp it as in the previous request)
                            agent.state.torque = 0.3 * (expected_ang_vel - agent.state.ang_vel[:, :1])
                            
                            
                            # expected_ang_vel = torch.stack([0.2*(self.formation_goals[i][:, 2] - agent.state.rot[:,0])], dim=-1)
                            # agent.state.torque = 1.0*(expected_ang_vel - agent.state.ang_vel[:, :1])
                            # print("torque:{}".format(agent.state.torque))
                            # agent.state.torque = torch.clamp(agent.state.torque, min=-1, max=1)
                        #     agent.set_ang_vel(
                        #     torch.stack([0.5*(self.formation_goals[i][:, 2] - agent.state.rot[:,0])], dim=-1),
                        # batch_index=None,
                        #     )
                        elif self.working_mode == "RL":
                            # agent.state.force = 2*(agent.action.u[:, :2] - agent.state.vel[:, :2])
                            # agent.state.torque = agent.action.u[:, 2].unsqueeze(-1) - agent.state.ang_vel[:, :1]
                            p_gain_pos = 5.0
                            p_gain_rot = 1.5
                            
                            # Get the error between the current pose and the high-level goal pose
                            pos_error_x = self.formation_goals[i][:, 0] - agent.state.pos[:, 0]
                            pos_error_y = self.formation_goals[i][:, 1] - agent.state.pos[:, 1]
                            # IMPORTANT: Normalize angle error to handle wrap-around (e.g., from pi to -pi)
                            rot_error = self._normalize_angle(self.formation_goals[i][:, 2] - agent.state.rot[:,0])

                            # Calculate the raw, unclamped velocity commands
                            vx_desired_unclamped = p_gain_pos * pos_error_x
                            vy_desired_unclamped = p_gain_pos * pos_error_y
                            w_desired_unclamped = p_gain_rot * rot_error

                            # --- 2. Clamp the desired velocities to respect physical limits ---
                            # This uses the maximum speed parameters defined in your ObstacleAvoidanceControllerMixin
                            vx_clamped = torch.clamp(vx_desired_unclamped, -self.OA_MAX_SPEED_X, self.OA_MAX_SPEED_X)
                            vy_clamped = torch.clamp(vy_desired_unclamped, -self.OA_MAX_SPEED_Y, self.OA_MAX_SPEED_Y)
                            w_clamped = torch.clamp(w_desired_unclamped, -self.OA_MAX_SPEED_YAW, self.OA_MAX_SPEED_YAW)
                            
                            # This is the realistic velocity command that the high-level policy "wants"
                            original_velocity_clamped = torch.stack([vx_clamped, vy_clamped, w_clamped], dim=-1)

                            oa_velocity = self.get_oa_velocity(agent, i, original_velocity_clamped)
                            
                            agent.set_vel(
                                   oa_velocity[:, :2],
                                batch_index=None,
                            )
                            # print("dwa vel:{}".format(apf_vel.shape))
                        #     agent.set_ang_vel(
                        #         torch.stack([1.5*(self.formation_goals[i][:, 2] - agent.state.rot[:,0])], dim=-1),
                        # batch_index=None,
                        #     )       
                            agent.set_ang_vel(
                                oa_velocity[:, 2].unsqueeze(dim=-1),
                        batch_index=None,
                            )       
                            # agent.set_vel(
                                    # torch.stack([3*(self.formation_goals[i][:, 0] - agent.state.pos[:, 0]), 3*(self.formation_goals[i][:, 1] - agent.state.pos[:, 1])], dim=-1) ,
                                # batch_index=None,
                            # )
                        elif self.working_mode == "potential_field":
                            agent.set_vel(
                            self.compute_potential_field_velocity(agent, i),
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
        
    def _simulate_holonomic_trajectory(self, vx: Tensor, vy: Tensor, w: Tensor) -> Tensor:
        """
        Simulates a short trajectory for a holonomic robot given a velocity command.
        
        Args:
            vx, vy, w (Tensor): Tensors of shape [batch_dim] for linear and angular velocities.
        
        Returns:
            Tensor: Simulated trajectory points of shape [batch_dim, num_sim_steps, 2]
                    in the robot's local frame.
        """
        batch_dim = vx.shape[0]
        num_sim_steps = int(self.OA_PREDICT_TIME / self.OA_SIMULATION_TIMESTEP)
        
        # Initialize trajectory points at the origin (robot's local frame)
        trajectory = torch.zeros(batch_dim, num_sim_steps, 2, device=self.device)
        
        # Initial local pose is (0, 0, 0)
        x = torch.zeros(batch_dim, device=self.device)
        y = torch.zeros(batch_dim, device=self.device)
        theta = torch.zeros(batch_dim, device=self.device)

        for t in range(num_sim_steps):
            # For a holonomic robot, vx and vy are directly in the local frame
            x += vx * self.OA_SIMULATION_TIMESTEP
            y += vy * self.OA_SIMULATION_TIMESTEP
            # Note: For a diff-drive robot, this would be different:
            # theta += w * self.OA_SIMULATION_TIMESTEP
            # x += vx * torch.cos(theta) * self.OA_SIMULATION_TIMESTEP
            # y += vx * torch.sin(theta) * self.OA_SIMULATION_TIMESTEP
            
            trajectory[:, t, 0] = x
            trajectory[:, t, 1] = y
            
        return trajectory

    def _get_robot_footprint(self) -> Tensor:
        """Creates a set of points representing the robot's circular footprint."""
        footprint = [
            [0.0, 0.0], [self.OA_ROBOT_RADIUS, 0.0], [-self.OA_ROBOT_RADIUS, 0.0],
            [0.0, self.OA_ROBOT_RADIUS], [0.0, -self.OA_ROBOT_RADIUS]
        ]
        return torch.tensor(footprint, device=self.device, dtype=torch.float32)

    def _check_trajectory_collision(self, trajectory_local: Tensor, current_pos: Tensor, current_rot: Tensor) -> Tensor:
        """Checks if the robot's circular footprint collides with obstacles along a trajectory."""
        batch_dim, num_sim_steps, _ = trajectory_local.shape
        num_footprint_points = self.footprint_local.shape[0]

        trajectory_with_footprint_local = trajectory_local.unsqueeze(2) + self.footprint_local.view(1, 1, -1, 2)
        
        cos_rot = torch.cos(current_rot).unsqueeze(1)
        sin_rot = torch.sin(current_rot).unsqueeze(1)
        current_pos_exp = current_pos.unsqueeze(1).unsqueeze(2)

        x_local = trajectory_with_footprint_local[..., 0]
        y_local = trajectory_with_footprint_local[..., 1]
        x_global_offset = x_local * cos_rot - y_local * sin_rot
        y_global_offset = x_local * sin_rot + y_local * cos_rot
        global_traj_points = torch.stack([x_global_offset, y_global_offset], dim=-1) + current_pos_exp
        
        global_traj_points_flat = global_traj_points.view(-1, 2)
        grid_indices = ((global_traj_points_flat - self.bitmap.origin) / self.bitmap.resolution).long()
        
        # grid_indices[:, 0] = torch.clamp(grid_indices[:, 0], 0, self.bitmap.grid_size - 1)
        # grid_indices[:, 1] = torch.clamp(grid_indices[:, 1], 0, self.bitmap.grid_size - 1)

        batch_indices = torch.arange(batch_dim, device=self.device).repeat_interleave(num_sim_steps * num_footprint_points)
        values = self.bitmap.bitmap[batch_indices, grid_indices[:, 1], grid_indices[:, 0]]
        is_collision_flat = (values == self.bitmap.obstacle_value)
        
        return torch.any(is_collision_flat.view(batch_dim, -1), dim=1)

    def get_oa_velocity(self, agent, agent_idx: int, original_velocity: Tensor) -> Tensor:
        """
        Checks the original velocity command for safety using a prioritized search.
        If unsafe, it tries to find a safe alternative by first reducing lateral velocity,
        then searching angular velocity, then reducing forward velocity.
        """
        # --- 0. Get Current State and Find the Bitmap Obstacle ---
        
        current_pos = agent.state.pos
        current_rot = agent.state.rot
        
        # Initialize final velocities with the original command
        final_vx = original_velocity[:, 0].clone()
        final_vy = original_velocity[:, 1].clone()
        final_w = original_velocity[:, 2].clone()

        # --- Search Level 1: Check original velocity ---
        original_traj = self._simulate_holonomic_trajectory(final_vx, final_vy, final_w)
        is_unsafe = self._check_trajectory_collision(original_traj, current_pos, current_rot)
        
        # If all trajectories in the batch are safe, we are done
        if not torch.any(is_unsafe):
            return original_velocity

        # --- Search Level 2: Try setting lateral velocity to zero ---
        # Only for the environments that were unsafe
        unsafe_indices = is_unsafe.nonzero(as_tuple=True)[0]
        vy_zeroed = torch.zeros_like(final_vy[unsafe_indices])
        
        traj_vy_zeroed = self._simulate_holonomic_trajectory(final_vx[unsafe_indices], vy_zeroed, final_w[unsafe_indices])
        is_still_unsafe_after_vy_zero = self._check_trajectory_collision(traj_vy_zeroed, current_pos[unsafe_indices], current_rot[unsafe_indices])
        
        # Update the velocities for those that are now safe
        newly_safe_mask = ~is_still_unsafe_after_vy_zero
        if newly_safe_mask.any():
            indices_now_safe = unsafe_indices[newly_safe_mask]
            final_vy[indices_now_safe] = 0.0
            # Update the main `is_unsafe` mask
            is_unsafe[indices_now_safe] = False
        
        if not torch.any(is_unsafe):
            return torch.stack([final_vx, final_vy, final_w], dim=1)

        # --- Search Level 3 & 4: Iteratively search angular and forward velocities ---
        # We only need to search for the envs that are still unsafe
        unsafe_indices = is_unsafe.nonzero(as_tuple=True)[0]
        
        # For these remaining unsafe trajectories, we will work with vy=0
        vx_to_search = final_vx[unsafe_indices]
        vy_to_search = torch.zeros_like(vx_to_search) # Lateral velocity is now zero
        w_to_search = final_w[unsafe_indices]

        found_safe_alternative = torch.zeros_like(unsafe_indices, dtype=torch.bool)
        
        # Velocity reduction steps (e.g., 100%, 75%, 50%)
        vel_reduction_factors = torch.linspace(1.0, 0.25, self.OA_NUM_VELOCITY_SEARCH_STEPS, device=self.device)
        angular_search_space = torch.linspace(0, self.OA_MAX_SPEED_YAW, self.OA_NUM_ANGULAR_SEARCH_STEPS, device=self.device)

        for vel_factor in vel_reduction_factors:
            if found_safe_alternative.all(): break
            
            current_vx_candidate = vx_to_search * vel_factor
            
            for i in range(1, self.OA_NUM_ANGULAR_SEARCH_STEPS):
                if found_safe_alternative.all(): break
                
                for turn_direction in [1, -1]:
                    if found_safe_alternative.all(): break
                    
                    indices_to_check = unsafe_indices[~found_safe_alternative]
                    if indices_to_check.numel() == 0: break
                    
                    # Get the subset of velocities for the envs we are still trying to fix
                    vx_subset = current_vx_candidate[~found_safe_alternative]
                    vy_subset = vy_to_search[~found_safe_alternative]
                    w_subset_original = w_to_search[~found_safe_alternative]

                    w_candidate = w_subset_original + turn_direction * angular_search_space[i]
                    w_candidate = torch.clamp(w_candidate, -self.OA_MAX_SPEED_YAW, self.OA_MAX_SPEED_YAW)

                    candidate_traj = self._simulate_holonomic_trajectory(vx_subset, vy_subset, w_candidate)
                    
                    is_candidate_unsafe = self._check_trajectory_collision(
                        candidate_traj, 
                        current_pos[indices_to_check], 
                        current_rot[indices_to_check]
                    )
                    
                    newly_safe_mask = ~is_candidate_unsafe
                    if newly_safe_mask.any():
                        indices_of_newly_safe = indices_to_check[newly_safe_mask]
                        # Update the final velocities for these envs
                        final_vx[indices_of_newly_safe] = vx_subset[newly_safe_mask]
                        final_vy[indices_of_newly_safe] = vy_subset[newly_safe_mask] # Stays 0
                        final_w[indices_of_newly_safe] = w_candidate[newly_safe_mask]
                        
                        # Update the master `found_safe_alternative` mask
                        temp_found_mask = torch.zeros_like(unsafe_indices, dtype=torch.bool)
                        temp_found_mask[~found_safe_alternative] = newly_safe_mask
                        found_safe_alternative |= temp_found_mask

        # --- Search Level 5: Safety Stop ---
        # For any trajectories that are still unsafe after all searches, stop the robot
        still_unsafe_indices = unsafe_indices[~found_safe_alternative]
        if still_unsafe_indices.numel() > 0:
            final_vx[still_unsafe_indices] = 0.0
            final_vy[still_unsafe_indices] = 0.0
            final_w[still_unsafe_indices] = 0.0

        return torch.stack([final_vx, final_vy, final_w], dim=1)
    
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
            k_leader = 0.2
        k_repulsive_lidar = 1.0  # Strength of the LiDAR repulsive force (tune this parameter)
        # Initialize the total force vector
        total_force = k_goal * goal_direction_normalized  # Shape: [batch_size, 2]

        # Initialize repulsive forces
        repulsive_forces_lidar = torch.zeros_like(total_force)  # Shape: [batch_size, 2]
        
        repulsive_forces = torch.zeros_like(total_force)  # Shape: [batch_size, 2]
        toward_leader_forces = torch.zeros_like(total_force)
        toward_leader_forces = k_leader* leader_direction_normalized




        lidar_sensor = agent.sensors[0]
        max_range = lidar_sensor._max_range
        lidar_readings = max_range - self.current_lidar_reading
         # Shape: [batch_size, num_beams] (e.g., [batch_size, 20])
        num_beams = lidar_readings.shape[1]
        batch_size = lidar_readings.shape[0]


        local_beam_angles_offsets = lidar_sensor._angles # Shape: [1, num_beams]
        # print("local_beam_angles_offset shape:{}".format(local_beam_angles_offsets.shape))
        # print("local_beam_angles_offset:{}".format(local_beam_angles_offsets))

        # Get agent's current world rotation
        # agent.state.rot is shape [batch_size, 1]

        agent_world_rotations = agent.state.rot 
        # print("agent_world_rations shape:{}".format(agent_world_rotations.shape))
        # print("agent_world_rations :{}".format(agent_world_rotations))

        # Calculate world-frame angles for each beam for each agent
        # Broadcasting local_beam_angles_offsets ([1, num_beams])
        # with agent_world_rotations ([batch_size, 1])
        # results in world_beam_angles ([batch_size, num_beams])
        world_beam_angles = local_beam_angles_offsets + agent_world_rotations # <--- ADDITION HAPPENS HERE
        # print("world_beam_angles shape:{}".format(world_beam_angles.shape))
        # print("world_beam_angles :{}".format(world_beam_angles))

        # Calculate the (x, y) direction vector FOR EACH RAY in the world frame
        # beam_ray_dir_x_world, beam_ray_dir_y_world will have shape [batch_size, num_beams]
        beam_ray_dir_x_world = torch.cos(world_beam_angles) # <--- THEN COSINE
        beam_ray_dir_y_world = torch.sin(world_beam_angles) # <--- THEN SINE



        # Expand beam directions for batch operations to [batch_size, num_beams]
        # beam_dir_x_batch = beam_ray_dir_x_world.unsqueeze(0).expand(batch_size, -1)
        # beam_dir_y_batch = beam_ray_dir_y_world.unsqueeze(0).expand(batch_size, -1)

        # Parameters for LiDAR repulsion
        
        # Consider readings up to max_range as potentially hitting an obstacle.
        # A small epsilon is used for floating point comparisons.
        obstacle_detection_threshold = max_range - 1.2

        # Identify beams hitting obstacles: True where an obstacle is detected
        # obstacle_mask: [batch_size, num_beams]
        obstacle_mask = lidar_readings < obstacle_detection_threshold

        # Calculate repulsive magnitude for all beams.
        # Magnitude is proportional to (max_range - distance) / max_range,
        # so it's strongest (k_repulsive_lidar) for very close obstacles (distance near 0)
        # and zero for obstacles at max_range or further.
        # magnitude_numerator: [batch_size, num_beams]
        magnitude_numerator = max_range - lidar_readings
        
        # Ensure magnitudes are non-negative and scaled.
        # repulsive_magnitudes_unmasked: [batch_size, num_beams]
        repulsive_magnitudes_unmasked = k_repulsive_lidar * torch.clamp(magnitude_numerator, min=0.0) / max_range
        
        # Apply mask: only consider magnitudes where an obstacle was actually hit.
        # effective_repulsive_magnitudes: [batch_size, num_beams]
        effective_repulsive_magnitudes = repulsive_magnitudes_unmasked * obstacle_mask.float()

        # Calculate the (x, y) components of the repulsive force for each beam.
        # The force direction is OPPOSITE to the beam's direction (i.e., from obstacle to agent).
        # rep_force_x_per_beam: [batch_size, num_beams]
        # rep_force_y_per_beam: [batch_size, num_beams]
        rep_force_x_per_beam = effective_repulsive_magnitudes * (-beam_ray_dir_x_world)
        rep_force_y_per_beam = effective_repulsive_magnitudes * (-beam_ray_dir_y_world)

        # Sum repulsive forces from all beams for each agent in the batch.
        # repulsive_forces_lidar: [batch_size, 2]
        repulsive_forces_lidar[:, 0] = torch.sum(rep_force_x_per_beam, dim=1)
        repulsive_forces_lidar[:, 1] = torch.sum(rep_force_y_per_beam, dim=1)









       
        # angles = torch.linspace(
                # 0, 2 * torch.pi, n_rays + 1, device=self.device
            # )[:n_rays]
        # For each batch dimension
        # batch_size = current_pos.shape[0]
        # for dim in range(batch_size):
            # pass
            
        # Add repulsive forces to total force
        total_force += repulsive_forces_lidar
        # print("repulsive:{}".format(repulsive_forces_lidar))
        # print("toward_leader:{}".format(toward_leader_forces))
        # print("toward goal:{}".format(k_goal * goal_direction_normalized))
        # print("total force:{}".format(total_force))
        total_force += toward_leader_forces
        # Normalize the total force to get the velocity direction
        total_force_norm = torch.norm(total_force, dim=1, keepdim=True)
        velocity_direction = total_force / (total_force_norm + 1e-6)

        # Set the agent's speed (you can adjust the speed as needed)
        max_speed = 1.2  # Limit speed to prevent excessive velocities
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
        # print("formation_normal_width:{}".format(self.formation_normal_width))
        # print("left scale:{}".format(left_scale_factor))
        # print("right scale:{}".format(right_scale_factor))
        
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
        # print("scale:0.9")
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
        # print("update formation positions scale:{}".format(scale))
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



    def single_agent_collision_obstacle_rew(self, agent):
        """
        Calculates a smooth, continuous, and stable obstacle avoidance reward.
        The penalty is based on an exponential function of the closest LiDAR reading.
        This function returns the reward value for the current step.

        Args:
            agent: The agent object, which must have an `obs` dictionary containing
                   a 'laser' tensor of shape [batch_dim, lidar_ray_num].

        Returns:
            torch.Tensor: A tensor of shape [batch_dim] with the calculated reward.
        """
        self.OBSTACLE_PENALTY_STEEPNESS = 5.0

        # This is the maximum penalty an agent can receive from this reward component.
        # Keeping this value from being excessively large is key to training stability.
        # Good starting range: -1.0 to -20.0
        self.MAX_OBSTACLE_PENALTY = -10.0
        
        # The distance at which the penalty starts. Should be greater than min_range.
        self.PENALTY_START_DISTANCE = 1.0 
        # Ensure agent observation is valid before proceeding
        

        lidar_readings = self.current_lidar_reading # Shape: [batch_dim, lidar_ray_num]
        batch_dim = lidar_readings.shape[0]

        if batch_dim == 0:
            return torch.zeros(0, device=self.device)
            
        # --- New Reward Logic ---

        # 1. Focus on the single closest obstacle detected by any ray.
        # This provides a much clearer signal than summing all penalties.
        # min() returns a tuple (values, indices)
        min_dist_per_env, _ = torch.min(lidar_readings, dim=1) # Shape: [batch_dim]

        # 2. Use a smooth exponential penalty function.
        # The penalty is 0 when min_dist >= PENALTY_START_DISTANCE.
        # It curves smoothly towards MAX_OBSTACLE_PENALTY as min_dist approaches 0.
        # Formula: Penalty = MAX_PENALTY * exp(-steepness * distance)
        
        # We only apply the penalty if the closest reading is within the threshold.
        # First, calculate the potential penalty for all instances in the batch.
        penalty = self.MAX_OBSTACLE_PENALTY * torch.exp(
            -self.OBSTACLE_PENALTY_STEEPNESS * min_dist_per_env
        )

        # Apply the penalty only where the condition is met.
        # Otherwise, the reward component is 0.
        reward_for_this_agent_batch = torch.where(
            min_dist_per_env < self.PENALTY_START_DISTANCE,
            penalty,
            torch.zeros_like(min_dist_per_env) # No penalty if all rays are clear
        )

        return reward_for_this_agent_batch
    
    # def single_agent_collision_obstacle_rew(self, agent):
    #     #agent.collision_obstacle_rew is of shape torch.zeros(batch_dim, device=device)
    #     # self.current_lidar_reading  is of shape [batch_size, lidar_ray_num] 
    #     self.LIDAR_DANGER_THRESHOLD = 0.8  # meters
    #     self.LIDAR_CRITICAL_DISTANCE = 0.25 # meters
    #     self.CRITICAL_DISTANCE_PENALTY = -50.0
    #     self.NORMAL_DANGER_PENALTY_SCALE = -10.0 # Adjusted scale for more impact
    #     current_agent_index = self.world.agents.index(agent)

    #     lidar_readings = self.current_lidar_reading   # Shape: [batch_dim, lidar_ray_num]
    #     print("lidar_readings:{}".format(lidar_readings))
    #     batch_dim = lidar_readings.shape[0]
        
    #     if batch_dim == 0: # No environments in the batch
    #         return

    #     # Mask for readings below the critical distance
    #     critical_mask = lidar_readings < self.LIDAR_CRITICAL_DISTANCE
    #     # Mask for readings in the "danger zone" (below danger threshold but not critical)
    #     danger_mask = (lidar_readings < self.LIDAR_DANGER_THRESHOLD) & (~critical_mask)
    #     # Calculate penalties for critical distances
    #     # Assign CRITICAL_DISTANCE_PENALTY where critical_mask is true, 0 otherwise
    #     critical_penalties = torch.where(
    #         critical_mask,
    #         torch.full_like(lidar_readings, self.CRITICAL_DISTANCE_PENALTY, device=lidar_readings.device),
    #         torch.zeros_like(lidar_readings, device=lidar_readings.device)
    #     )

    #     # Calculate penalties for normal danger zone
    #     # Penalty = SCALE * (THRESHOLD - reading)
    #     # This makes penalty more negative as reading gets smaller (closer to obstacle)
    #     normal_danger_penalties = torch.where(
    #         danger_mask,
    #         self.NORMAL_DANGER_PENALTY_SCALE * (self.LIDAR_DANGER_THRESHOLD - lidar_readings),
    #         torch.zeros_like(lidar_readings, device=lidar_readings.device)
    #     )
        
    #     # Combine penalties: critical penalties take precedence if a ray is in both (though our masks are exclusive)
    #     # Summing them works because only one mask can be true for a given reading for the penalty part.
    #     total_penalties_per_ray = critical_penalties + normal_danger_penalties
        
    #     # Sum penalties from all rays for each instance in the batch
    #     # This means if multiple rays detect obstacles, the penalty accumulates.
    #     # An alternative could be to take the min penalty (most severe single ray),
    #     # or average of non-zero penalties. Summing is often a strong signal.
    #     reward_for_this_agent_batch = total_penalties_per_ray.sum(dim=1) # Sum over lidar_ray_num dimension

    #     # Add this calculated reward component to the agent's reward attribute.
    #     # If agent.collision_obstacle_rew should *only* be this value, then use direct assignment:
    #     # agent.collision_obstacle_rew = reward_for_this_agent_batch
    #     # If it's an accumulator for various collision/obstacle related rewards:
         
    #     return reward_for_this_agent_batch
    
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

    def _normalize_angle(self, angle):
        """Normalize angles to the range [-pi, pi]."""
        angle = angle % (2 * torch.pi)
        angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)
        angle = torch.where(angle < -torch.pi, angle + 2 * torch.pi, angle)
        return angle

    
    def compute_group_center_reward(self):
        
        self.VIRTUAL_CENTER_RELATIVE_X = -1.8  # meters (target behind leader)
        self.VIRTUAL_CENTER_RELATIVE_Y = 0.0    # meters (target centered with leader)
        self.VIRTUAL_CENTER_RELATIVE_THETA = 0.0 # radians (target same orientation as leader)
        
        self.GROUP_CENTER_POS_REWARD_SCALE = -1.0
        self.GROUP_CENTER_ORIENT_REWARD_SCALE = -0.5
        
        self.RECTANGLE_CENTER_X_OFFSET = -3.0  # Longitudinal offset of rectangle center from leader (negative for behind)
        self.RECTANGLE_LATERAL_HALFWIDTH = 0.4   # Half-width (y-extent in leader's frame)
        self.RECTANGLE_LONGITUDINAL_HALFLENGTH = 3 # Half-length (x-extent in leader's frame, around its center_x_offset)
        self.OUT_OF_RECTANGLE_PENALTY_SCALE = -10.0 # Penalty scale for being outside

        self.OPEN_SPACE_CENTER_X = -1.8
        self.OPEN_SPACE_LATERAL_HALFWIDTH = 2.0  # Wide containment box
        self.OPEN_SPACE_LONGITUDINAL_HALFLENGTH = 1.5

        # --- Target Formation Parameters (for narrow, line formations) ---
        self.LINE_FORMATION_CENTER_X = -3.0  # Centroid is further back
        self.LINE_FORMATION_LATERAL_HALFWIDTH = 0.1 # Very narrow containment
        self.LINE_FORMATION_LONGITUDINAL_HALFLENGTH = 3.0

        # --- Reward Scaling ---
        self.GROUP_CENTER_POS_REWARD_SCALE = -1.0
        self.GROUP_CENTER_ORIENT_REWARD_SCALE = -0.5
        self.OUT_OF_RECTANGLE_PENALTY_SCALE = -10.0

        # --- Adaptation Control ---
        self.OPENING_TO_LINE_TRANSITION_STEEPNESS = 1.5 # Controls how fast the formation changes as opening narrows
        self.OPENING_TRANSITION_MIDPOINT = 3.0 # The total opening width (m) where transition is halfway
        self.LATERAL_ADAPTATION_SCALE = 0.5 # How much to shift laterally based on asymmetry (0 to 1)
        self.MAX_LATERAL_SHIFT = 0.5 # Max distance (m) the center can shift sideways
        follower_agents = []
        for i, agent in enumerate(self.world.agents):
            if i > 0:
                follower_agents.append(agent)

        """
        Computes a reward for maintaining group cohesion around a virtual center
        relative to the leader, and penalizes excessive spread.
        Assumes self.world.agents and self.leader_agent are set.
        Assumes agent.state.pos [batch_dim, 2] and agent.state.rot [batch_dim, 1] exist.
        """
        if len(follower_agents) <= 0: # No followers
            return torch.zeros(self.batch_dim, device=self.device)

        # --- Group Center and Orientation Cohesion (Existing Logic) ---
        all_follower_pos = torch.stack([agent.state.pos for agent in follower_agents], dim=1)
        all_follower_orient = torch.stack([agent.state.rot for agent in follower_agents], dim=1)
        
        current_group_centroid_pos = torch.mean(all_follower_pos, dim=1)
        
        follower_orient_x = torch.cos(all_follower_orient)
        follower_orient_y = torch.sin(all_follower_orient)
        avg_orient_x = torch.mean(follower_orient_x, dim=1)
        avg_orient_y = torch.mean(follower_orient_y, dim=1)
        current_group_avg_orient = torch.atan2(avg_orient_y, avg_orient_x)

        leader_pos = self.leader_agent.state.pos
        leader_orient = self.leader_agent.state.rot

        total_opening_width = self.smoothed_left_opening + self.smoothed_right_opening
        
        # Alpha is a transition factor from 0 (open space) to 1 (line formation)
        alpha = torch.sigmoid(
            self.OPENING_TO_LINE_TRANSITION_STEEPNESS * (self.OPENING_TRANSITION_MIDPOINT - total_opening_width)
        )
        # Interpolate the desired X offset for the group center
        dynamic_center_x = (1 - alpha) * self.OPEN_SPACE_CENTER_X + alpha * self.LINE_FORMATION_CENTER_X

        # Calculate the desired Y offset based on opening asymmetry
        asymmetry = self.smoothed_left_opening - self.smoothed_right_opening
        dynamic_center_y = self.MAX_LATERAL_SHIFT * torch.tanh(self.LATERAL_ADAPTATION_SCALE * asymmetry)
        
        # Interpolate the rectangle dimensions
        dynamic_rect_lat_halfwidth = (1 - alpha) * self.OPEN_SPACE_LATERAL_HALFWIDTH + alpha * self.LINE_FORMATION_LATERAL_HALFWIDTH
        dynamic_rect_long_halfwidth = (1 - alpha) * self.OPEN_SPACE_LONGITUDINAL_HALFLENGTH + alpha * self.LINE_FORMATION_LONGITUDINAL_HALFLENGTH

        rel_x_vc = self.VIRTUAL_CENTER_RELATIVE_X
        rel_y_vc = self.VIRTUAL_CENTER_RELATIVE_Y
        

        cos_l_orient = torch.cos(leader_orient).squeeze(-1)
        sin_l_orient = torch.sin(leader_orient).squeeze(-1)

        global_offset_x = dynamic_center_x * cos_l_orient - dynamic_center_y * sin_l_orient
        global_offset_y = dynamic_center_x * sin_l_orient + dynamic_center_y * cos_l_orient
        global_offset = torch.stack([global_offset_x, global_offset_y], dim=1)
        desired_virtual_pos_global = leader_pos + global_offset
        # global_offset_y = rel_x_vc * sin_l_orient + rel_y_vc * cos_l_orient
        desired_virtual_orient_global = self._normalize_angle(leader_orient + self.VIRTUAL_CENTER_RELATIVE_THETA)

        pos_error_vec = current_group_centroid_pos - desired_virtual_pos_global
        pos_error_dist = torch.norm(pos_error_vec, dim=1) # Get the simple Euclidean distance
        MAX_POS_PENALTY = -10.0 # Define a reasonable maximum penalty
# tanh will be 0 when error is 0, and approach 1 as error grows.
# So, the penalty will smoothly go from 0 down to -20.
        position_penalty = MAX_POS_PENALTY * torch.tanh(pos_error_dist) 
        orient_error = self._normalize_angle(current_group_avg_orient - desired_virtual_orient_global)
        orient_error_dist = torch.norm(orient_error, dim=1) # Get the simple Euclidean distance
        MAX_ORI_PENALTY = -10.0 # Define a reasonable maximum penalty
# tanh will be 0 when error is 0, and approach 1 as error grows.
# So, the penalty will smoothly go from 0 down to -20.
        position_penalty = MAX_POS_PENALTY * torch.tanh(pos_error_dist) 
        orientation_penalty = MAX_ORI_PENALTY * torch.tanh(orient_error_dist) 
        group_center_cohesion_reward = position_penalty + orientation_penalty

        # --- Follower Containment in Rectangle Penalty (New Logic) ---
        # 1. Transform follower positions to leader's local frame (already done for spread)
        translated_follower_pos = all_follower_pos - leader_pos.unsqueeze(1)
        neg_leader_orient = -leader_orient
        cos_neg_lo = torch.cos(neg_leader_orient)
        sin_neg_lo = torch.sin(neg_leader_orient)
        
        cos_neg_lo_exp = cos_neg_lo.unsqueeze(2)
        sin_neg_lo_exp = sin_neg_lo.unsqueeze(2)

        tfp_x = translated_follower_pos[:, :, 0].unsqueeze(2)
        tfp_y = translated_follower_pos[:, :, 1].unsqueeze(2)

        follower_pos_local_x = (tfp_x * cos_neg_lo_exp - tfp_y * sin_neg_lo_exp).squeeze(-1) # Shape: [batch_dim, num_followers]
        follower_pos_local_y = (tfp_x * sin_neg_lo_exp + tfp_y * cos_neg_lo_exp).squeeze(-1) # Shape: [batch_dim, num_followers]

        # 2. Define rectangle boundaries in leader's local frame
        # Rectangle center is at (self.RECTANGLE_CENTER_X_OFFSET, 0) in leader's local frame
        rect_local_center_x = dynamic_center_x
        rect_local_min_x = rect_local_center_x - dynamic_rect_long_halfwidth
        rect_local_max_x = rect_local_center_x + dynamic_rect_long_halfwidth
        
        # Also shift the lateral boundaries based on the asymmetry
        rect_local_center_y = dynamic_center_y
        rect_local_min_y = rect_local_center_y - dynamic_rect_lat_halfwidth
        rect_local_max_y = rect_local_center_y + dynamic_rect_lat_halfwidth

        # Calculate out-of-bounds distance for each follower
        dx_out = torch.relu(rect_local_min_x.unsqueeze(1) - follower_pos_local_x) + torch.relu(follower_pos_local_x - rect_local_max_x.unsqueeze(1))
        dy_out = torch.relu(rect_local_min_y.unsqueeze(1) - follower_pos_local_y) + torch.relu(follower_pos_local_y - rect_local_max_y.unsqueeze(1))
        
        # Use a bounded penalty function here as well
        out_of_bounds_dist = torch.sqrt(dx_out**2 + dy_out**2)
        per_follower_out_of_rect_penalty = self.OUT_OF_RECTANGLE_PENALTY_SCALE * torch.tanh(out_of_bounds_dist)
        
        total_out_of_rectangle_penalty = torch.sum(per_follower_out_of_rect_penalty, dim=1)

        
        
        
        
        # rect_local_min_x = self.RECTANGLE_CENTER_X_OFFSET - self.RECTANGLE_LONGITUDINAL_HALFLENGTH
        # rect_local_max_x = self.RECTANGLE_CENTER_X_OFFSET + self.RECTANGLE_LONGITUDINAL_HALFLENGTH
        
        # # desired_rect_local_max_y = 3.0
        # # desired_rect_local_min_y = -3.0
        # desired_rect_local_max_y = self.smoothed_left_opening   # Based on left perceived opening
        # desired_rect_local_min_y = -(self.smoothed_right_opening) # Based on right perceived opening
        
        # # Ensure rect_local_min_y is not positive, and rect_local_max_y is not negative
        # # Also ensure a minimum small opening if perceived opening is too small (e.g., < 0.5)
        # # These are element-wise operations for the batch
        # rect_local_min_y = torch.where(desired_rect_local_min_y >= -0.01, # If it's positive or too close to zero
        #                                torch.full_like(desired_rect_local_min_y, -0.01), 
        #                                desired_rect_local_min_y)
        # rect_local_max_y = torch.where(desired_rect_local_max_y <= 0.01,  # If it's negative or too close to zero
        #                                torch.full_like(desired_rect_local_max_y, 0.01),
        #                                desired_rect_local_max_y)
        
        
        # # 3. Calculate out-of-bounds distance for each follower
        # # For x-dimension (longitudinal)
        # dx_low = torch.relu(rect_local_min_x - follower_pos_local_x)  # How much below min_x
        # dx_high = torch.relu(follower_pos_local_x - rect_local_max_x) # How much above max_x
        # # dx_out_sq = (dx_low + dx_high)**2 # Squared longitudinal out-of-bounds distance per follower
        # dx_out_dist = torch.tanh(dx_low + dx_high)
        # # For y-dimension (lateral)
        # dy_low = torch.relu(rect_local_min_y.unsqueeze(1) - follower_pos_local_y)   # How much below min_y (more negative y_local)
        # dy_high = torch.relu(follower_pos_local_y - rect_local_max_y.unsqueeze(1))  # How much above max_y (more positive y_local)
        # # dy_out_sq = (dy_low + dy_high)**2 # Squared lateral out-of-bounds distance per follower
        # dy_out_dist = torch.tanh(dy_low + dy_high)
        # # 4. Total out-of-bounds penalty per follower
        # # Sum of squared distances for each follower, then sum over followers for each batch instance
        # per_follower_out_of_rect_penalty = self.OUT_OF_RECTANGLE_PENALTY_SCALE * (dx_out_dist + dy_out_dist)
        
        # # Sum penalties for all followers for each batch instance
        # total_out_of_rectangle_penalty = torch.sum(per_follower_out_of_rect_penalty, dim=1) # Shape: [batch_dim]

        # --- Combine all reward components ---
        final_group_reward = group_center_cohesion_reward + total_out_of_rectangle_penalty
        # final_group_reward = total_out_of_rectangle_penalty
        
        return final_group_reward

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
                    
        #         for dim in range(self.world.batch_dim):
        #             if adjacency[dim, i, j] == 1.0:
        #                 is_line_clear = self.is_line_clear(agent1.state.pos[dim, :], agent2.state.pos[dim, :], self.obstacle_manager_list[dim])
        #                 if is_line_clear == False:
        #                     adjacency[dim, i, j] == 0.0
        #                     adjacency[dim, j, i] = 0.0
        # # Initialize connectivity rewards: [batch_size]
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
                a.group_center_rew[:] = 0
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

                    distance = torch.norm(a.state.pos - b.state.pos, dim=1)

                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # print("dist between {}, {}:{}".format(i, j, distance))
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

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
        #     for i, a in enumerate(self.world.agents):
        #         closest_dist = 10 * torch.ones(self.batch_dim, device=self.device)
        #         # agent_formation_rew = self.agent_formation_reward(agent)
        #         # print("obstacle_tenspr shape:{}".format(self.obstacle_manager_list[0].obstacles_tensor.shape))
        #         distances = torch.norm(a.state.pos.unsqueeze(1) - self.obstacle_manager_list[0].obstacles_tensor.unsqueeze(0), dim=2)

        #         closest_dist = torch.min(distances, dim=1).values
    
        #         # 后续碰撞判断逻辑保持不变
        #         # collision_mask = closest_dist < 0.2
        #         # Iterate through each obstacle
        #         # for obs in self.obstacles:
        #         #     # Calculate the distance to the obstacle
        #         #     # distance = self.world.get_distance(a, obs)
        #         #     distance = torch.norm(a.state.pos - obs.state.pos)
        #         #     # print("distance:{}".format(distance))

        #         #     # Compare and update the closest distance
        #         #     closest_dist = torch.min(closest_dist, distance)

        #         # Check for collision and update the reward based on closest distance
        #         collision_mask = closest_dist < 0.2
        #         if torch.any(collision_mask):  # Check if there are any collisions
        #             if i > 0:
        #                 self.eva_collision_num[i-1 ,collision_mask] = 1
        #             a.agent_collision_rew[collision_mask] += -0.4 * (1 - closest_dist[collision_mask] / 0.2)
        #         # if torch.any(collision_free_mask):
        #             # a.formation_rew[collision_free_mask] = agent_formation_rew[collision_free_mask]
            self.connection_rew = self.compute_connectivity_reward()
            # print("connection time:{}".format(time.time() - connection_time))
            self.group_center_diff_rew = self.compute_group_center_reward()
        #leader robot do not contribute to the reward

        agent.collision_obstacle_rew = self.single_agent_collision_obstacle_rew(agent)
        if self.env_type == "bitmap":
            
            agent.pos_rew = self.single_agent_reward_graph_formation_maintained(agent)
        if is_first:
            # print("single reward timme:{}, index:{}".format(time.time() - reward_time,current_agent_index))
            return agent.target_collision_rew

        if self.env_type == "clutter":
            agent.connection_rew = self.compute_agent_connectivity_reward(agent)
        else:
            agent.connection_rew = self.connection_rew.clone()

        agent.group_center_rew = self.group_center_diff_rew.clone()
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
                self.last_action_u[current_agent_index] = copy.deepcopy(self.current_action_mean[current_agent_index])
            agent.action_diff_rew = -torch.norm(self.current_action_mean[current_agent_index] - self.last_action_u[current_agent_index], dim=1)

            self.last_action_u[current_agent_index] = copy.deepcopy(self.current_action_mean[current_agent_index])
            agent.angle_diff_with_leader_rew = -torch.norm(agent.state.rot - self.leader_agent.state.rot, dim=1)
        else:
            pass
            # self.last_action_u[current_agent_index] = copy.deepcopy(agent.action.u)
        # agent_direction_align_reward = self.agent_velocity_target_direction_alignment_reward(agent)
        # print("agent agent_collision_rew shape:{}".format(agent.agent_collision_rew.shape))
        # angle_reward = self.angle_rew if self.shared_rew else agent.angle_rew
        # return 5*pos_reward + self.final_rew + agent.agent_collision_rew + angle_reward
        # return self.formation_maintain_rew + agent.agent_collision_rew
        # print("agent {} connection reward:{} collsition reward:{}, connection_rew:{}, target collision rew:{}".format(current_agent_index, agent.connection_rew, agent.agent_collision_rew, agent.connection_rew,agent.target_collision_rew))
        # return agent.agent_collision_rew + agent.connection_rew + agent.action_diff_rew + agent.formation_rew
        agent.agent_collision_rew = 1000*agent.agent_collision_rew
        agent.connection_rew = 25*agent.connection_rew
        agent.action_diff_rew = 50*agent.action_diff_rew
        agent.angle_diff_with_leader_rew = 0.8*agent.angle_diff_with_leader_rew
        agent.collision_obstacle_rew = 3* agent.collision_obstacle_rew
        agent.pos_rew = 5* agent.pos_rew
        # agent.target_collision_rew = 10*agent.target_collision_rew
        # print("single reward timme:{}, index:{}".format(time.time() - reward_time,current_agent_index))
        # return agent.angle_diff_with_leader_rew + agent.agent_collision_rew + agent.connection_rew + agent.action_diff_rew + agent.target_collision_rew 
        # return agent.angle_diff_with_leader_rew + agent.agent_collision_rew + agent.connection_rew + agent.action_diff_rew 
        # return agent.pos_rew
        if self.env_type == "bitmap":
            # return agent.pos_rew
            # return agent.group_center_rew + agent.agent_collision_rew
            # return agent.connection_rew + agent.group_center_rew + agent.agent_collision_rew + agent.collision_obstacle_rew + agent.action_diff_rew
            # rewards_setup2 = agent.connection_rew + agent.group_center_rew + agent.agent_collision_rew + agent.action_diff_rew
            # rewards_setup1 = agent.pos_rew + agent.connection_rew + agent.group_center_rew + agent.agent_collision_rew + agent.action_diff_rew
            # is_clutter_condition = (self.smoothed_left_opening < 1.2) & (self.smoothed_right_opening < 1.2)
            # final_rewards = torch.where(
            #     is_clutter_condition, 
            #     rewards_setup2,       # Value if condition is True
            #     rewards_setup1        # Value if condition is False
            # ) 
            
            return  agent.connection_rew  + agent.group_center_rew + agent.agent_collision_rew + agent.action_diff_rew + agent.collision_obstacle_rew

            # return agent.connection_rew + agent.group_center_rew + agent.agent_collision_rew + agent.collision_obstacle_rew
        elif self.env_type == "bitmap_tunnel":
            return agent.connection_rew + agent.collision_obstacle_rew + agent.group_center_rew + agent.agent_collision_rew
        


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

    

    def single_agent_reward_graph_formation_maintained(self, agent, rotation_weight=0.4):
        current_agent_index = self.world.agents.index(agent)
        current_agent_positions_dict = {} # Renamed for clarity: current agent states
        current_agent_rotations_dict = {} # Renamed for clarity: current agent states

        for a_index, ag in enumerate(self.world.agents):
            current_agent_positions_dict[a_index] = ag.state.pos
            current_agent_rotations_dict[a_index] = ag.state.rot

        batch_size = agent.state.pos.shape[0]
        num_agents = len(self.world.agents)

        # Prepare list of all goal specifications (which include both position and rotation)
        # This list will contain tensors of shape [batch_size, num_features_per_goal]
        formation_goals_specs_list = []

        if not hasattr(self, 'formation_nominal_goals') or \
        len(self.formation_nominal_goals) != num_agents:
            raise ValueError("self.formation_nominal_goals is not properly defined or has incorrect length.")
        if self.formation_nominal_goals and self.formation_nominal_goals[0].shape[1] < 3:
            raise ValueError("self.formation_nominal_goals[i] must have at least 3 features (x, y, rotation).")


        for i in range(num_agents):
            formation_goals_specs_list.append(self.formation_nominal_goals[i])

        # Verbose printing, consider reducing for performance
        # print(f"Debug: formation_goals_specs_list length: {len(formation_goals_specs_list)}")
        # if formation_goals_specs_list:
        #     print(f"Debug: formation_goals_specs_list[0] shape: {formation_goals_specs_list[0].shape}")

        for batch_idx in range(batch_size):
           

            # Get the current agent's actual position for this batch item
            agent_actual_position = current_agent_positions_dict[current_agent_index][batch_idx, :2]
            # Get the position of the goal assigned to this agent (for positional reward)
            assigned_target_position = self.formation_nominal_goals[current_agent_index][batch_idx, :2]

            # --- Positional Reward ---
            distance_to_positional_goal = torch.norm(agent_actual_position - assigned_target_position)
            position_reward_component = -distance_to_positional_goal

            # --- Rotational Reward/Penalty ---
            # Get the current agent's actual rotation for this batch item
            agent_actual_rotation = current_agent_rotations_dict[current_agent_index][batch_idx, 0] # Shape e.g. [1]
            
            # Get the target rotation of the goal assigned to this agent
            assigned_target_rotation = self.formation_nominal_goals[current_agent_index][batch_idx, 2] # Shape e.g. [1]

            rot_diff = agent_actual_rotation - assigned_target_rotation
            normalized_rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))
            rotation_cost_component = normalized_rot_diff.abs()

            # --- Combine Rewards ---
            final_reward = position_reward_component - rotation_weight * rotation_cost_component
            if final_reward.numel() > 1:
                print(f"Warning: final_reward is not scalar: {final_reward}. Taking mean.")
                final_reward = final_reward.mean()

            # print(f"Agent {current_agent_index}, Batch {batch_idx}: PosDist {distance_to_positional_goal.item():.3f}, "
            #       f"RotDiff_abs {rotation_cost_component.item():.3f}, FinalRew {final_reward.item():.3f}")
            
            agent.pos_rew[batch_idx] = final_reward

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

    


    def set_last_policy_output(self, output):
        self.last_policy_output = copy.deepcopy(output)

    def normalize_angles_for_opening_calculation(self, angles_0_to_2pi: torch.Tensor) -> torch.Tensor:
        """
        Normalizes angles from [0, 2*pi) to approximately [-pi, pi).
        0 stays 0. Angles > pi are mapped to negative counterparts.
        e.g., pi/2 -> pi/2; pi -> -pi (or pi); 3*pi/2 -> -pi/2.
        """
        angles_mod = angles_0_to_2pi % (2 * torch.pi)
        # For angles in [pi, 2*pi), map them to [-pi, 0)
        # For angles in [0, pi), they remain positive.
        normalized_angles = torch.where(angles_mod >= torch.pi, angles_mod - 2 * torch.pi, angles_mod)
        return normalized_angles

    def compute_lidar_opening_raw(self, 
        lidar_readings: torch.Tensor,    # Shape: [batch_dim, n_rays]
        ray_angles_local_0_to_2pi: torch.Tensor,  # Shape: [batch_dim, n_rays], local frame, IN [0, 2*PI) RANGE
        max_half_opening_width: float,   # Max possible opening on one side (e.g., nominal_formation_width / 2)
        obstacle_distance_threshold: float, # Rays shorter than this are considered obstacles defining the opening edge
        max_abs_angle_for_opening_calc: float = torch.pi / 2  # Max *absolute* normalized angle to consider (e.g., pi/2 for +/- 90 deg)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the perceived left and right opening widths based on LiDAR readings.
        Handles local ray angles provided in the [0, 2*pi) range by normalizing them.

        Args:
            lidar_readings: Distances measured by LiDAR rays.
            ray_angles_local_0_to_2pi: Local angles of LiDAR rays (0 is front, CCW positive, range [0, 2*pi)).
            max_half_opening_width: Maximum possible width for one side of the opening (cap).
            obstacle_distance_threshold: LiDAR readings below this are treated as obstacles.
            max_abs_angle_for_opening_calc: Defines the angular cone (e.g., +/- pi/2 from front)
                                            to search for opening edges using normalized angles.

        Returns:
            A tuple (final_left_opening_abs_y, final_right_opening_abs_y):
                - final_left_opening_abs_y: Tensor of shape [batch_dim] with the perceived opening
                                        width to the left (positive y in local frame).
                - final_right_opening_abs_y: Tensor of shape [batch_dim] with the perceived opening
                                            width to the right (absolute value of y for right side).
        """
        batch_dim = lidar_readings.shape[0]
        device = lidar_readings.device

        if batch_dim == 0:
            return torch.zeros(0, device=device), torch.zeros(0, device=device)

        # Normalize angles to [-pi, pi) for easier left/right symmetric logic
        # Positive normalized angles are left (CCW), negative are right (CW)
        ray_angles_normalized = self.normalize_angles_for_opening_calculation(ray_angles_local_0_to_2pi)

        # Create masks for relevant rays
        # 1. Rays within the desired angular cone (using normalized angles)
        frontal_cone_mask = torch.abs(ray_angles_normalized) <= max_abs_angle_for_opening_calc
        
        # 2. Rays that hit something close enough to be considered an obstacle
        obstacle_hit_mask = lidar_readings < obstacle_distance_threshold
        
        # Combine masks: relevant obstacles are those within the frontal cone AND close enough
        relevant_obstacle_mask = frontal_cone_mask & obstacle_hit_mask

        # Calculate the local y-coordinate of all LiDAR hit points
        # Use original ray_angles_local_0_to_2pi for sin calculation as it matches the frame definition
        y_coords_of_hits = lidar_readings * torch.sin(ray_angles_local_0_to_2pi) # Shape: [batch_dim, n_rays]

        # --- Determine Left Opening ---
        # Left side corresponds to normalized angles > 0
        # We are looking for the smallest positive y-coordinate among relevant obstacles on the left.
        left_side_criteria = relevant_obstacle_mask & (ray_angles_normalized > 1e-4) # Epsilon for strictly positive

        left_obstacle_y_values = torch.where(
            left_side_criteria,
            y_coords_of_hits, # y_coords are based on original angles, sin correctly handles this
            torch.full_like(y_coords_of_hits, float('inf')) # Use 'inf' for non-obstacles on left
        )
        
        left_opening_y, _ = torch.min(left_obstacle_y_values, dim=1) # Shape: [batch_dim]
        
        left_opening_y = torch.where(
            torch.isinf(left_opening_y) | (left_opening_y < 0), # If no valid left obstacle or y became negative
            torch.full_like(left_opening_y, max_half_opening_width),
            left_opening_y
        )
        left_opening_y = torch.clamp(left_opening_y, 0, max_half_opening_width)

        print("left_opening_y:{}".format(left_opening_y))
        left_opening_y = left_opening_y - 0.7
        print("left_opening_y after minus0.7:{}".format(left_opening_y))
        left_opening_y = torch.where(
             (left_opening_y < 0), # If no valid left obstacle or y became negative
            torch.full_like(left_opening_y, 0.01),
            left_opening_y
        )
        final_left_opening_abs_y = torch.clamp(left_opening_y, 0, max_half_opening_width)

        # --- Determine Right Opening ---
        # Right side corresponds to normalized angles < 0
        # We are looking for the largest negative y-coordinate (closest to zero from negative side).
        right_side_criteria = relevant_obstacle_mask & (ray_angles_normalized < -1e-4) # Epsilon for strictly negative

        right_obstacle_y_values = torch.where(
            right_side_criteria,
            y_coords_of_hits, # y_coords are based on original angles
            torch.full_like(y_coords_of_hits, float('-inf')) # Use '-inf' for non-obstacles on right
        )

        right_opening_y, _ = torch.max(right_obstacle_y_values, dim=1) # Shape: [batch_dim], will be negative if obstacle found

        right_opening_y = torch.where(
            torch.isinf(right_opening_y) | (right_opening_y > 0), # If no valid right obstacle or y became positive
            torch.full_like(right_opening_y, -max_half_opening_width),
            right_opening_y
        )
        right_opening_y = torch.clamp(right_opening_y, -max_half_opening_width, 0)
        right_opening_y = right_opening_y + 0.7
        right_opening_y = torch.where(
             (right_opening_y > 0), # If no valid left obstacle or y became negative
            torch.full_like(right_opening_y, -0.01),
            right_opening_y
        )

        final_right_opening_abs_y = torch.clamp(torch.abs(right_opening_y), 0, max_half_opening_width)

        return final_left_opening_abs_y, final_right_opening_abs_y

    def update_and_get_smoothed_opening(
        self,
        raw_lidar_readings: torch.Tensor,
        ray_angles: torch.Tensor,
        max_half_opening_width: float,
        obstacle_distance_threshold: float,
        max_abs_angle_for_opening_calc: float = torch.pi / 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the raw opening, applies Exponential Moving Average smoothing,
        and returns the stable, smoothed opening values.
        """
        # Reset the smoothed values for environments that just finished an episode
        # self.reset(dones=dones)

        # 1. Get the current raw measurement
        raw_left, raw_right = self.compute_lidar_opening_raw(
            raw_lidar_readings,
            ray_angles,
            max_half_opening_width,
            obstacle_distance_threshold,
            max_abs_angle_for_opening_calc
        )

        # 2. Apply EMA filter
        # smoothed_value = alpha * new_value + (1 - alpha) * old_smoothed_value
        alpha = self.LIDAR_OPENING_SMOOTHING_ALPHA
        
        self.smoothed_left_opening = alpha * raw_left + (1 - alpha) * self.smoothed_left_opening
        self.smoothed_right_opening = alpha * raw_right + (1 - alpha) * self.smoothed_right_opening

        # Return the newly updated smoothed values
        return self.smoothed_left_opening, self.smoothed_right_opening

    def observation(self, agent: Agent):
        # get_obs_time = time.time()
        goal_poses = []
        goal_rot_poses = []
        current_agent_index = self.world.agents.index(agent)
        agent_num = len(self.world.agents)
        graph_list = []
        max_obstacles =100  # Define this attribute in your class
        num_node_features = 3  # Assuming x and y coordinates
        num_edge_features = 1  # Assuming distance as edge attribute
        num_agents = agent_num  # Number of agents per graph
        max_nodes = num_agents + max_obstacles  # Agents + Obstacles
        max_edges = num_agents * max_obstacles * 2 + (num_agents * (num_agents - 1)) * 2  # Approximate
        # if current_agent_index == 0:
        # print("index:{}".format(current_agent_index))
        if self.has_laser == True:
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raw_lidar_reading = agent.sensors[0].measure()
            lidar_observation_tensor = [(agent.sensors[0]._max_range - raw_lidar_reading) / agent.sensors[0]._max_range]
            self.current_lidar_reading = raw_lidar_reading
            if current_agent_index == 0:
                # agnet.sensors[0].measure()   shape: [batch_size, lidar_ray_num]
                # print("leader lidar reading shape:{}".format(agent.sensors[0].measure().shape))  
                # print("angles shape:{}".format(agent.sensors[0]._angles.shape))
                # print("angles:{}".format(agent.sensors[0]._angles ))
                # self.left_opening, self.right_opening = self.compute_lidar_opening(raw_lidar_reading, agent.sensors[0]._angles, 2.0, 3.5, torch.pi*9.0/10.0)
            # print("lidar_observation_tensor :{}".format(self.current_lidar_reading.shape))
                self.smoothed_left_opening, self.smoothed_right_opening = self.update_and_get_smoothed_opening(raw_lidar_reading, agent.sensors[0]._angles, 2.0, 3.5, torch.pi*9.0/10.0)
                print("left open:{}, right open:{}".format(self.smoothed_left_opening, self.smoothed_right_opening))
        # Define feature_length:
        # - Node features: max_nodes * num_node_features
        # - Edge indices: max_edges * 2
        # - Edge attributes: max_edges * num_edge_features
        # Total feature_length = max_nodes*num_node_features + max_edges*(2 + num_edge_features)
        
        feature_length = max_nodes * num_node_features + max_edges * (2 + num_edge_features)
        # feature_length = (num_agents + max_obstacles) * num_node_features + max_obstacles * 2 + max_obstacles * num_edge_features
        nominal_formation_tensor = torch.zeros((self.world.batch_dim, 3), device=self.device)
            
        # angles = [0.0, -18/180.0*math.pi, 18/180.0*math.pi, -15/180.0*math.pi,  15/180.0*math.pi]

        # dists = [0.0, -1.34, -1.34, -2.68, -2.68]
        # dists = [0.0, -1.34, -1.34, -2.68, -3.68]

        # nominal_positions_x = [0.0, -1.274, -1.274, -2.548, -2.548]
        # nominal_positions_y = [0.0, 0.414, -0.414, 0.828, -0.828]

        nominal_positions_x = [0.0, -1.2, -1.2, -2.4, -2.4]
        nominal_positions_y = [0.0, 0.6, -0.6, 1.2, -1.2]
        nominal_formation_tensor[:, 0] = nominal_positions_x[current_agent_index]
        nominal_formation_tensor[:, 1] = nominal_positions_y[current_agent_index]
        nominal_formation_tensor[:, 2] = 0.0
        previous_positions = self.agent_history.get_previous_positions()  # [num_agents, history_length, 2]

        relative_poses = []
        for d in range(self.world.batch_dim):
            # print("obs manager:{}".format(self.obstacle_manager_list[d]))
            leader_pos = self.leader_agent.state.pos[d, :]  # Shape: [2]
            leader_rot = self.leader_agent.state.rot[d]      # Scalar (radians)
            # print("leader_pos:{}".format(leader_pos))
         
            if current_agent_index == 0:
                relative_poses.append(torch.zeros(3, device=self.device).unsqueeze(dim=1))  # 新增这行：[0.0, 0.0, 0.0]
            else:
                rel_pos = agent.state.pos[d] - leader_pos

                raw_rel_theta = agent.state.rot[d] - leader_rot # Scalar tensor

           
                rel_theta_normalized = torch.atan2(torch.sin(raw_rel_theta), torch.cos(raw_rel_theta))
                relative_dist = torch.norm(rel_pos).item()
                # print("relative_duist:{}".format(relative_dist))
                if relative_dist > self.max_distance_from_follower_to_leader:
                    self.formation_out_of_shape[d] = True

                    # 坐标转换到领队坐标系
                cos_theta = torch.cos(leader_rot)
                sin_theta = torch.sin(leader_rot)
                rotated_pos = torch.stack([
                    cos_theta * rel_pos[0] + sin_theta * rel_pos[1],
                    -sin_theta * rel_pos[0] + cos_theta * rel_pos[1]
                ])
                    
                relative_poses.append(torch.cat([rotated_pos, rel_theta_normalized.unsqueeze(0)]))
        # print("relative_pos shape:{}".format(torch.stack(relative_poses).shape))
        opening_tensor = torch.stack([self.smoothed_left_opening, self.smoothed_right_opening], dim=0)
        print("opening_tensor shape:{}".format(opening_tensor.shape))
        if self.has_laser == True:
            return {
                'laser': torch.tensor(lidar_observation_tensor[0]),  # shape [B, 20]
                'relative_pos': torch.stack(relative_poses).squeeze(dim=-1),  # shape [B, 3]
                'nominal_pos_diff': nominal_formation_tensor - torch.stack(relative_poses).squeeze(dim=-1),
                'nominal_pos': nominal_formation_tensor, 
                'leader_vel': self.leader_agent.state.vel,
                'leader_ang_vel': self.leader_agent.state.ang_vel,
                'forward_opening': opening_tensor,
                'last_action_u': self.last_action_u[current_agent_index],
            }    
        else:
            return {
                # 'laser': torch.tensor(lidar_observation_tensor[0]),  # shape [B, 20]
                'relative_pos': torch.stack(relative_poses).squeeze(dim=-1),  # shape [B, 3]
                'nominal_pos_diff': nominal_formation_tensor - torch.stack(relative_poses).squeeze(dim=-1),
                'nominal_pos': nominal_formation_tensor, 
                'leader_vel': self.leader_agent.state.vel,
                'leader_ang_vel': self.leader_agent.state.ang_vel,
                'last_action_u': self.last_action_u[current_agent_index],

            }   
        
        

    
      
    def done(self):
        # Get number of agents in the world
        num_agents = len(self.world.agents)
        
        # Collect all agent positions into a single tensor [batch_dim, agent_num, 2]
        agent_poses = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        
        # Initialize done tensor with all False values
        done_status = torch.zeros(self.batch_dim, device=self.device, dtype=torch.bool)
        
   
        # collisions = self.bitmap.check_collision_with_poses(agent_poses)
        # done_status = done_status | collisions
        done_status = done_status | self.reached_leader_path_end

        done_status = done_status | self.formation_out_of_shape
        # print("done:{}".format(done_status))
        return done_status

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
        translated_agent_pos = agent.state.pos - self.leader_agent.state.pos  # Shape: [batch_dim, 2]
        translated_agent_rot = torch.zeros_like(agent.state.rot)
        # # Rotate agent positions into leader's frame
        # # Assume self.leader_robot.state.rot is a tensor of shape [batch_dim]
        cos_theta = torch.cos(self.leader_agent.state.rot).unsqueeze(-1)  # Shape: [batch_dim, 1]
        sin_theta = torch.sin(self.leader_agent.state.rot).unsqueeze(-1)  # Shape: [batch_dim, 1]
        
        # # Define rotation matrices for each batch
        # rotation_matrices = torch.cat([cos_theta, sin_theta, -sin_theta, cos_theta], dim=1).reshape(-1, 2, 2)  # Shape: [batch_dim, 2, 2]
        
        # # Apply rotation: [batch_dim, 2, 2] x [batch_dim, 2, 1] -> [batch_dim, 2, 1]
        # optimized_target_pos = torch.bmm(rotation_matrices, translated_agent_pos.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_dim, 2]
        # optimized_target_pose = torch.cat([optimized_target_pos, translated_agent_rot], dim=1)
        # print("optimized_target_pose:{}".format(optimized_target_pose.shape))
        env_type = self.get_forward_env_type()
        
        if current_agent_index == 0:
            opening_tensor = torch.stack([self.smoothed_left_opening, self.smoothed_right_opening], dim=1)
            opening_list = list(opening_tensor)
            print("opening_list:{}".format(opening_list))
            return {
                # "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
                # "final_rew": self.final_rew,
                "forward_opening": opening_list,
                "eva_collision_num": self.eva_collision_num,
                "eva_connection_num": self.eva_connection_num,   # 1 means connected, 0 means not connected
                "env_observation": self.env_observation,
                "agent_collisions": agent.agent_collision_rew,
                "leader_x": agent.state.pos[:, 0],
                # "formation_goal": agent.goal.state.pos,
                # "formation_main_rew":self.formation_maintain_rew,
                # "optimized_target_pos": optimized_target_pose,  # Transformed to leader's frame
                # "graph_list": graph_list,
                "agent_connection_rew": agent.connection_rew,
                "agent_diff_rew": agent.action_diff_rew,
                "agent_target_collision": agent.target_collision_rew,
                "agent_vel": agent.state.vel,
                "agent_ang_vel": agent.state.ang_vel,
                "group_center_rew": agent.group_center_rew,
                "collision_obstacle_rew": agent.collision_obstacle_rew,
                "agent_pos_rew": agent.pos_rew,
                "env_type": env_type,

            }
        else:
            return {
                # "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
                # "final_rew": self.final_rew,
                "group_center_rew": agent.group_center_rew,
                "agent_collisions": agent.agent_collision_rew,
                "collision_obstacle_rew": agent.collision_obstacle_rew,
                # "formation_goal": agent.goal.state.pos,
                # "formation_main_rew":self.formation_maintain_rew,
                # "optimized_target_pos": optimized_target_pose,  # Transformed to leader's frame
                # "graph_list": graph_list,
                "agent_connection_rew": agent.connection_rew,
                "agent_diff_rew": agent.action_diff_rew,
                "agent_target_collision": agent.target_collision_rew,
                "agent_vel": agent.state.vel,
                "agent_ang_vel": agent.state.ang_vel,
                "agent_pos_rew": agent.pos_rew,

            }
    


    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Render connections between agents (original code)
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
                                width=0.3,
                            )
                            xform = rendering.Transform()
                            line.add_attr(xform)
                            line.set_color(*color)
                            geoms.append(line)
        
        # NEW CODE: Render the leader's path if available
        try:
            if hasattr(self, 'batch_leader_paths') and self.batch_leader_paths is not None:
                if env_index < len(self.batch_leader_paths) and self.batch_leader_paths[env_index]:
                    path = self.batch_leader_paths[env_index]
                    # print("render leader path")
                    # print("path:{}".format(path))
                    # input("render leader path")
                    # Draw lines connecting path waypoints
                    for i in range(len(path) - 1):
                        # Extract coordinates, handling tensor values
                        if torch.is_tensor(path[i]):
                            x1, y1 = path[i][0].item(), path[i][1].item()
                            x2, y2 = path[i+1][0].item(), path[i+1][1].item()
                        else:
                            x1, y1 = path[i][0], path[i][1]
                            x2, y2 = path[i+1][0], path[i+1][1]
                        
                        # Draw path segment as a green line
                        line = rendering.Line(
                            (x1, y1),
                            (x2, y2),
                            width=0.1,  # Thicker than agent connections
                        )
                        xform = rendering.Transform()
                        line.add_attr(xform)
                        line.set_color(*Color.GREEN.value)  # Green path
                        geoms.append(line)
                    
                    # Draw waypoint markers
                    for point in path:
                        # Extract coordinates
                        if torch.is_tensor(point):
                            x, y = point[0].item(), point[1].item()
                        else:
                            x, y = point[0], point[1]
                            
                        # Draw blue circle at each waypoint
                        circle = rendering.make_circle(radius=0.03)
                        xform = rendering.Transform()
                        xform.set_translation(x, y)
                        circle.add_attr(xform)
                        circle.set_color(*Color.BLUE.value)  # Blue waypoints
                        geoms.append(circle)
        except Exception as e:
            # If there's an error in rendering the path, log it but don't crash
            print(f"Error rendering leader path: {e}")

        left_opening_text = rendering.TextLine("left:{}".format(self.smoothed_left_opening.item()), x = 0.5, y=0.2)
        right_opening_text = rendering.TextLine("right:{}".format(self.smoothed_right_opening.item()), x = 275.7, y=0.2)
        geoms.append(left_opening_text)
        geoms.append(right_opening_text)

        return geoms





if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
