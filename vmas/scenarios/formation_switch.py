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
        self.plot_grid = True
        self.grid_spacing = 1
        self.device =device
        # self.agents_with_same_goal = kwargs.get("agents_with_same_goal", 1)
        # self.split_goals = kwargs.get("split_goals", False)
        self.observe_all_goals = kwargs.get("observe_all_goals", True)

        self.lidar_range = kwargs.get("lidar_range", 1.8)
        self.agent_radius = kwargs.get("agent_radius", 0.35)
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
  
        world = World(batch_dim, device, substeps=2)
        world._x_semidim = self.world_semidim
        world._y_semidim = self.world_semidim
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
                color=Color.GRAY,
            )
            # self.formation_goals_landmark[i].renderable = False
            world.add_landmark(self.formation_goals_landmark[i])
            
        world.add_landmark(self.formation_center)
        world.add_landmark(self.leader_robot)

        self.obstacle_pattern = 1
        self.create_obstacles(self.obstacle_pattern, world)

        def detect_obstacles(x):
            return x.name.startswith("obs_") or x.name.startswith("agent_") or x.name.startswith("wall")




        #add leader agent
        self.leader_agent = Agent(
                name=f"agent_0",
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
        self.leader_agent.pos_rew = torch.zeros(batch_dim, device=device)
        self.leader_agent.angle_rew = torch.zeros(batch_dim, device=device)
        self.leader_agent.agent_collision_rew = self.leader_agent.pos_rew.clone()
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
                color=color,
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
        self.last_policy_output = None
        history_length = 1  # Example: store last 5 positions
        self.agent_history = AgentHistory(
            batch_dim=batch_dim,
            num_agents=self.n_agents,
            history_length=history_length,
            device=device
        )
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
    def spawn_obstacles(self, obstacle_pattern, env_index):
        print("env_index:{}".format(env_index))
        passage_indexes = []
        j = self.n_boxes // 2
        line_segments = []  # Store line segments to maintain continuity
        polygon_list = []
        def create_polygon(
            existing_centers,
            num_vertices_min=8, 
            num_vertices_max=12, 
            sphere_radius=0.1, 
            max_spheres_per_polygon=50, 
            world_semidim=5.0, 
            device='cuda'
        ):
            # Random number of vertices between min and max
            num_vertices = np.random.randint(num_vertices_min, num_vertices_max)
            
            # Determine the minimum distance between centers to prevent overlap
            min_center_distance = 1.5  # Adjust based on polygon size and desired spacing
            
            # Loop until a suitable center is found
            max_attempts = 100
            for attempt in range(max_attempts):
                # Generate a random center
                center = torch.tensor(
                    [
                        np.random.uniform(-world_semidim + 2, world_semidim - 2),
                        np.random.uniform(-world_semidim + 2, world_semidim - 2),
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
            
            min_radius = 0.5  # Minimum distance from the center
            max_radius = 0.8   # Maximum distance from the center
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

            return positions, existing_centers
                

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
                        positions, outer_existing_centers = create_polygon(
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
                            0,
                            0.9,
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
                            0,
                            -0.8,
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
                        noise = torch.randn(line_segments[i_value].shape, device=self.device) * 0.02 # Scale noise as needed
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
                
                # print("i shape:{}".format(i.shape))
                # print("i:{}".format(i))
                # if i.item() < len(line_segments):
                #     return line_segments[i.item()]
                # else:
                #     # Handle cases where i exceeds the number of pre-generated segments
                #     return torch.tensor(
                #         [
                #             np.random.uniform(-self.world_semidim, self.world_semidim),
                #             np.random.uniform(-self.world_semidim, self.world_semidim),
                #         ],
                #         dtype=torch.float32,
                #         device=self.world.device,
                #     )
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
                    print("x_positions:{}".format(x_positions))
                    input("1")
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


        if obstacle_pattern != 1:
            # Initialize positions and assign to obstacles
            i = torch.zeros(
                (self.world.batch_dim,) if env_index is None else (1,),
                dtype=torch.int,
                device=self.world.device,
            )

            for obs in self.obstacles:
                pos = get_pos(i)
                if pos != None:
                    obs.set_pos(get_pos(i), batch_index=env_index)
                    i += 1

            # Create obstacle managers for each batch
            for d in range(self.world.batch_dim):
                single_batch_obstacles = [obs.state.pos[d,:].squeeze() for obs in self.obstacles]
                manager = ObstacleManager(single_batch_obstacles)
                self.obstacle_manager_list.append(manager)
        else:
            if len(line_segments) == 0:

                num_polygons = np.random.randint(5, 8)
                outer_existing_centers = []
                polygon_dict = {}
                # Generate polygons and collect positions
                for poly_idx in range(num_polygons):
                    positions, outer_existing_centers = create_polygon(
                        existing_centers=outer_existing_centers,
                        num_vertices_min=8,
                        num_vertices_max=12,
                        sphere_radius=0.1,
                        world_semidim=self.world_semidim,
                        device=self.device
                    )
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


                self.obstacle_manager_list = []
                for d in range(self.world.batch_dim):
                    single_batch_obstacles = [obs.state.pos[d, :].squeeze() for obs in self.obstacles]
                    manager = ObstacleManager(single_batch_obstacles)
                    self.obstacle_manager_list.append(manager)


                    
    def reset_world_at(self, env_index: int = None):
        print("reset_world_at {}".format(env_index))
        self.update_formation_assignment_time[env_index] = time.time()
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim+1, self.world_semidim),
            (-self.world_semidim+4, self.world_semidim),
        )

        ScenarioUtils.spawn_entities_randomly(
            self.world.landmarks,
            self.world,
            env_index,
            0.1,
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
                            -4.3536, 
                            0.3536,
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
                            -4.3536,  
                            -0.3536,
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
            elif i == 4:
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

    def find_clear_direction(self, current_pos, current_direction, obstacle_manager, max_scan_angle, scan_step, dim):
        """
        Find a clear direction to move in, starting from the current_direction.
        Scans left and right up to max_scan_angle, in increments of scan_step.
        """
        # First, check the forward direction
        if self.is_direction_clear(current_pos, current_direction, obstacle_manager, dim):
            return current_direction

        # Initialize variables
        angles_to_check = []
        for delta_angle in np.arange(scan_step, max_scan_angle + scan_step, scan_step):
            angles_to_check.append(current_direction + delta_angle)  # Left side
            angles_to_check.append(current_direction - delta_angle)  # Right side

        # Check each angle
        for angle in angles_to_check:
            if self.is_direction_clear(current_pos, angle, obstacle_manager, dim):
                return angle

        # If no clear direction found, return current_direction
        return current_direction

    def is_direction_clear(self, current_pos, direction, obstacle_manager, dim):
        """
        Checks if the formation can move in the given direction without colliding with obstacles.
        """
        scan_distance = self.scan_distance
        formation_width = self.calculate_formation_width()

        # Number of points across the formation width and along the path
        num_checks_width = 5
        num_checks_distance = 10  # Increase for finer resolution

        half_width = formation_width / 2

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
                obstacles = obstacle_manager.get_near_obstacles(point.cpu().numpy(), self.agent_radius)
                if obstacles:
                    return False  # Path is not clear
                # Also check bounds
                if not self.is_within_bounds(point):
                    return False
        return True

    def avoid_boundaries(self, tentative_next_pos, current_direction, current_pos, max_steering_angle):
        boundary_threshold_distance = 1.1
        repulsion_strength = 0.5

        # Compute distances to boundaries
        dx_right = self.world_semidim - tentative_next_pos[0]
        dx_left = tentative_next_pos[0] + self.world_semidim
        dy_top = self.world_semidim - tentative_next_pos[1]
        dy_bottom = tentative_next_pos[1] + self.world_semidim

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
        return (-self.world_semidim < x < self.world_semidim) and (-self.world_semidim < y < self.world_semidim)    
    
    def process_action(self, agent: Agent):
        self.velocity_limit = 0.03  # Adjusted speed for smoother movement
        is_first = agent == self.world.agents[0]
        if is_first:
            current_positions = torch.stack([agent.state.pos for agent in self.world.agents])  # [num_agents, batch_dim, 2]
            current_positions = current_positions.permute(1, 0, 2)  # [batch_dim, num_agents, 2]
            self.agent_history.update(current_positions)
            formation_movement = "random"
            if formation_movement == "random":
                for dim in range(self.world.batch_dim):
                    # Parameters
                    max_scan_angle = torch.pi / 2  # Maximum scanning angle (90 degrees)
                    scan_step = torch.pi / 18      # Scanning step (10 degrees)
                    scan_distance = 1.5            # Distance to check ahead
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
                    max_steering_angle = 0.06  # Maximum change in direction per step
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
                    self.leader_agent.set_pos(
                        torch.tensor(
                            [
                                (self.t-20)/30*0.5 - 4,
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


                
        
        # angles = [-135/180.0*math.pi, 135/180.0*math.pi, -135/180.0*math.pi,  135/180.0*math.pi]
        # dists = [-0.8, -0.8, -1.6, -1.6]
        angles = [0.0, -45/180.0*math.pi, 45/180.0*math.pi, -45/180.0*math.pi,  45/180.0*math.pi]

        dists = [0.0, -0.5, -0.5, -1.0, -1.0]
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
                    # angles = [-135/180.0*math.pi, 135/180.0*math.pi, -135/180.0*math.pi,  135/180.0*math.pi]
                    angle = angles[i]  # Get the angle for the current agent
                    dist = dists[i]    # Get the distance for the current agent

                    # Calculate the angle for the current formation
                    formation_angle = self.formation_center_pos[:, 2] + angle  # Shape [10, 1] + scalar

                    # Update the goals using torch.cos and torch.sin
                    self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + torch.cos(formation_angle) * dist
                    self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + torch.sin(formation_angle) * dist
                    # self.formation_goals[i][:, 0] = self.formation_center_pos[:, 0] + math.cos(self.formation_center_pos[:, 2] + angles[i]) * dists[i]
                    # self.formation_goals[i][:, 1] = self.formation_center_pos[:, 1] + math.sin(self.formation_center_pos[:, 2] + angles[i]) * dists[i]
                    self.formation_normal_width = math.sin(45/180.0*math.pi)*0.5* 4
                    agent.set_vel(
                            torch.stack([3*(self.formation_goals[i][:, 0] - agent.state.pos[:, 0]), 3*(self.formation_goals[i][:, 1] - agent.state.pos[:, 1])], dim=-1) ,
                        batch_index=None,
                    )
                    # agent.set_vel(
                    #     self.compute_agent_velocity(agent, i),
                    #     batch_index=None,
                    # )
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
        
    
    def compute_agent_velocity(self, agent, agent_index):
        # Get the agent's current position (batch_size x 2)
        current_pos = agent.state.pos  # Shape: [batch_size, 2]

        # Get the agent's formation goal position
        goal_pos = self.formation_goals[agent_index][:, :2]  # Shape: [batch_size, 2]

        # Compute the goal direction vector
        goal_direction = goal_pos - current_pos  # Shape: [batch_size, 2]
        goal_distance = torch.norm(goal_direction, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        goal_direction_normalized = goal_direction / (goal_distance + 1e-6)  # Avoid division by zero

        # Set the goal attraction strength
        k_goal = 1.0

        # Initialize the total force vector
        total_force = k_goal * goal_direction_normalized  # Shape: [batch_size, 2]

        # Initialize repulsive forces
        repulsive_forces = torch.zeros_like(total_force)  # Shape: [batch_size, 2]

        # For each batch dimension
        batch_size = current_pos.shape[0]
        for dim in range(batch_size):
            # Get obstacle manager for this environment
            obstacle_manager = self.obstacle_manager_list[dim]
            # Get obstacles near the agent
            obstacles = obstacle_manager.get_near_obstacles(current_pos[dim].cpu().numpy(), 1.0)  # Adjust search radius as needed

            if obstacles:
                # Convert obstacle positions to torch tensor
                obs_positions = torch.stack(obstacles).to(self.device)  # Shape: [num_obstacles, 2]

                # Compute vectors from agent to obstacles
                obs_vectors = current_pos[dim].unsqueeze(0) - obs_positions  # Shape: [num_obstacles, 2]

                # Compute distances
                obs_distances = torch.norm(obs_vectors, dim=1, keepdim=True)  # Shape: [num_obstacles, 1]

                # Avoid division by zero
                obs_distances = torch.clamp(obs_distances, min=0.1)

                # Compute repulsive forces
                k_rep = 0.5  # Adjust repulsion strength as needed
                influence_range = 1.0  # Obstacles within this range influence the agent
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

        # Add repulsive forces to total force
        total_force += repulsive_forces

        # Normalize the total force to get the velocity direction
        total_force_norm = torch.norm(total_force, dim=1, keepdim=True)
        velocity_direction = total_force / (total_force_norm + 1e-6)

        # Set the agent's speed (you can adjust the speed as needed)
        max_speed = 0.05  # Limit speed to prevent excessive velocities
        agent_speed = torch.clamp(goal_distance, max=max_speed)  # Shape: [batch_size, 1]

        # Compute the velocity
        velocity = velocity_direction * agent_speed

        return velocity  # Shape: [batch_size, 2]

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
            near_obstacles = obstacle_manager.get_near_obstacles(self.leader_robot.state.pos[d, :].cpu(), 1.2)

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
                                                    torch.sin(leader_direction + torch.pi / 2)], device=self.device)

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
                            left_opening = torch.abs(torch.dot(left_obstacle - self.leader_robot.state.pos[d, :], torch.tensor([torch.cos(leader_direction + torch.pi / 2), torch.sin(leader_direction + torch.pi / 2)], device=self.device)))
                            right_obstacle = None
                            right_opening = torch.tensor(2.0)
                        elif valid_obstacles[i+1][1] < 0:  # Both on the right
                            right_obstacle = valid_obstacles[i + 1][0]
                            right_opening = torch.abs(torch.dot(right_obstacle - self.leader_robot.state.pos[d, :], torch.tensor([torch.cos(leader_direction + torch.pi / 2), torch.sin(leader_direction + torch.pi / 2)], device=self.device)))
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
        
        agent.pos_rew = self.single_agent_reward_graph_formation_maintained(agent)
 
            

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
         
            task_mask = available_tasks.unsqueeze(1).expand(-1, num_agents, -1).transpose(0, 2)
          
            inf_mask = torch.where(task_mask, torch.zeros_like(working_distances), torch.inf)
            # Apply the updated inf_mask and compute valid distances
            valid_distances = torch.where(task_mask, working_distances, inf_mask)
        

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

        

    def batch_from_dense_to_ptg(self, x, fov=0.35 * torch.pi, max_distance=10.0):
        batch_size, num_agents, feature_length = x.shape

        # Extract positions and rotations
        positions = x[..., 0:2]  # Shape: (batch_size, num_agents, 2)
        # rotations = x[..., 2]    # Shape: (batch_size, num_agents)

        # Create a mask to avoid self-loops
        mask = ~torch.eye(num_agents, dtype=torch.bool, device=x.device).unsqueeze(0)  # Shape: (1, num_agents, num_agents)

        # Compute pairwise differences
        dx = positions[:, :, 0].unsqueeze(2) - positions[:, :, 0].unsqueeze(1)  # Shape: (batch_size, num_agents, num_agents)
        dy = positions[:, :, 1].unsqueeze(2) - positions[:, :, 1].unsqueeze(1)  # Shape: (batch_size, num_agents, num_agents)
        distances = torch.sqrt(dx**2 + dy**2)  # Shape: (batch_size, num_agents, num_agents)

        # Compute angles
        angles = torch.atan2(dy, dx)   # Shape: (batch_size, num_agents, num_agents)

        # angles = torch.atan2(dy, dx) - rotations.unsqueeze(2)  # Shape: (batch_size, num_agents, num_agents)
        angles = (angles + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize angles to [-pi, pi]

        # Apply field of view and distance constraints
        fov_mask = (angles.abs() <= fov) & (distances <= max_distance)  # Shape: (batch_size, num_agents, num_agents)

        # Combine masks
        final_mask = mask & fov_mask

        # Prepare edge_index and edge_attr
        edge_index_list = []
        edge_attr_list = []

        for b in range(batch_size):
            src, dst = final_mask[b].nonzero(as_tuple=True)
            edge_index = torch.stack([src, dst], dim=0)
            edge_attr = distances[b][final_mask[b]].unsqueeze(1)

            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        data_list = [Data(x=x[b], edge_index=edge_index_list[b], edge_attr=edge_attr_list[b]) for b in range(batch_size)]
        batch = Batch.from_data_list(data_list)
        
        return batch


    # def single_graph_from_data(self, nominal_formation_tensor, near_obstacles_in_leader_frame, previous_positions):
    #     """
    #     Constructs a graph with node features including current and previous positions.

    #     Args:
    #         nominal_formation_tensor (torch.Tensor): Tensor of shape [agent_num, 2] containing [x, y].
    #         near_obstacles_in_leader_frame (torch.Tensor or None): Tensor of shape [num_near_obstacles, 2] or None.
    #         previous_positions (torch.Tensor): Tensor of shape [agent_num, history_length, 2] containing previous positions.

    #     Returns:
    #         Data: A PyTorch Geometric Data object representing the graph.
    #     """

    #     if near_obstacles_in_leader_frame is not None and near_obstacles_in_leader_frame.numel() != 0:
    #         # Assign a category to distinguish between agents and obstacles
    #         agent_categories = torch.zeros((nominal_formation_tensor.size(0), 1), device=self.device)  # [agent_num, 1]
    #         obstacle_categories = torch.ones((near_obstacles_in_leader_frame.size(0), 1), device=self.device)  # [num_obstacles, 1]
            
    #         # Concatenate positions and previous positions for agents
    #         # Flatten previous_positions: [agent_num, history_length * 2]
    #         history_length = previous_positions.size(1)
    #         previous_positions_flat = previous_positions.view(nominal_formation_tensor.size(0), -1)  # [agent_num, history_length * 2]
    #         agent_features = torch.cat([nominal_formation_tensor, previous_positions_flat], dim=1)  # [agent_num, 2 + history_length * 2]
            
    #         # For obstacles, previous positions can be set to zero or handled differently
    #         # Here, we set them to zero since obstacles may be static
    #         obstacle_features = torch.cat([
    #             near_obstacles_in_leader_frame,  # [num_obstacles, 2]
    #             torch.zeros((near_obstacles_in_leader_frame.size(0), history_length * 2), device=self.device)  # [num_obstacles, history_length * 2]
    #         ], dim=1)  # [num_obstacles, 2 + history_length * 2]
            
    #         # Concatenate all node features
    #         x = torch.cat([agent_features, obstacle_features], dim=0)  # [agent_num + num_obstacles, 2 + history_length * 2]
            
    #         # Assign categories as additional node features if needed
    #         # For example, concatenate categories to node features
    #         categories = torch.cat([agent_categories, obstacle_categories], dim=0)  # [agent_num + num_obstacles, 1]
    #         x = torch.cat([x, categories], dim=1)  # [agent_num + num_obstacles, 3 + history_length * 2]
    #     else:
    #         # Only agents, no obstacles
    #         history_length = previous_positions.size(1)
    #         previous_positions_flat = previous_positions.view(nominal_formation_tensor.size(0), -1)  # [agent_num, history_length * 2]
    #         x = torch.cat([nominal_formation_tensor, previous_positions_flat], dim=1)  # [agent_num, 2 + history_length * 2]
    #         categories = torch.zeros((x.size(0), 1), device=self.device)  # [agent_num, 1]
    #         x = torch.cat([x, categories], dim=1)  # [agent_num, 3 + history_length * 2]


    #     # Define edges based on your specific requirements
    #     # For simplicity, let's assume a fully connected graph without self-loops
    #     edge_index = []
    #     edge_attr = []

    #     # Number of agents and obstacles
    #     num_agents = nominal_formation_tensor.size(0)
    #     threshold_distance= 0.8
    #     # Connect each nominal formation agent with near obstacles
    #     if near_obstacles_in_leader_frame != None:
    #         num_obstacles = near_obstacles_in_leader_frame.size(0)
    #         for agent_index in range(num_agents):
    #             agent_pos = nominal_formation_tensor[agent_index, :2]  # Get the position part
    #             for obstacle_index in range(num_obstacles):
    #                 obstacle_pos = near_obstacles_in_leader_frame[obstacle_index, :2]  # Get the position part
    #                 distance = torch.norm(agent_pos - obstacle_pos)
    #                 if distance <= threshold_distance:  # Check if within threshold distance
    #                     # Add edges from agent to obstacle
    #                     edge_index.append([agent_index, num_agents + obstacle_index])  # Agent to obstacle
    #                     edge_index.append([num_agents + obstacle_index, agent_index])  # Obstacle to agent
    #                     edge_attr.append([1])  # Edge type 1 for agent-obstacle
                        
    #     # Connect each pair of nominal formation agents
    #     for i in range(num_agents):
    #         for j in range(i + 1, num_agents):
    #             # Add edges between agents
    #             edge_index.append([i, j])  # Agent to agent
    #             edge_index.append([j, i])  # Agent to agent (reverse direction)
    #             edge_attr.append([0])  # Edge type 0 for agent-agent

    #     # Convert edge index and edge attributes to tensors
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Shape: [num_edges, 1]

    #     # Create the PyTorch Geometric data object
    #     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    #     return data
    def tensors_to_data_list(self, tensor_dict):
        x_tensor = tensor_dict['x']
        edge_index_tensor = tensor_dict['edge_index']
        edge_attr_tensor = tensor_dict['edge_attr']
        num_nodes_tensor = tensor_dict['num_nodes']
        
        data_list = []
        node_start = 0
        edge_start = 0
        
        for num_nodes in num_nodes_tensor:
            # Extract the portion of the tensor corresponding to this graph
            x = x_tensor[node_start:node_start + num_nodes]
            node_end = node_start + num_nodes
            edge_mask = (edge_index_tensor[0] >= node_start) & (edge_index_tensor[0] < node_end)
            edge_index = edge_index_tensor[:, edge_mask] - node_start  # Adjust indices relative to this graph
            edge_attr = edge_attr_tensor[edge_mask]

            # Create the Data object and append it to the list
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)
            
            node_start += num_nodes
        
        return data_list

    def data_list_to_tensors(self, data_list):
        # Initialize lists for storing graph attributes
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        num_nodes_list = []
        
        for data in data_list:
            # Append each attribute to the corresponding list
            x_list.append(data.x)
            edge_index_list.append(data.edge_index)
            edge_attr_list.append(data.edge_attr)
            num_nodes_list.append(data.x.size(0))  # Number of nodes in the graph
        
        # Convert lists to tensors
        x_tensor = torch.cat(x_list, dim=0)  # Stack all node features
        edge_index_tensor = torch.cat(edge_index_list, dim=1)  # Stack edge indices
        edge_attr_tensor = torch.cat(edge_attr_list, dim=0)  # Stack edge attributes
        num_nodes_tensor = torch.tensor(num_nodes_list)  # Save the number of nodes per graph

        # Save everything into a dictionary
        return {'x': x_tensor, 'edge_index': edge_index_tensor, 'edge_attr': edge_attr_tensor, 'num_nodes': num_nodes_tensor}
    def single_graph_from_data(self, nominal_formation_tensor, near_obstacles_in_leader_frame, threshold_distance=1):
        # print("nominal_formation_tensor shape:{}".format(nominal_formation_tensor.shape))
        
        nominal_formation_category = torch.zeros((nominal_formation_tensor.size(0), 1), device=self.device)  # [5, 1] with category 0
        nominal_formation_tensor = torch.cat((nominal_formation_tensor, nominal_formation_category), dim=1)  # Shape: [5, 3]
        if near_obstacles_in_leader_frame != None:
            # print("near_obstacles shape:{}".format(near_obstacles_in_leader_frame.shape))
        
        
            near_obstacles_category = torch.ones((near_obstacles_in_leader_frame.size(0), 1), device=self.device)  # [obstacle_num, 1] with category 1
            near_obstacles_in_leader_frame = torch.cat((near_obstacles_in_leader_frame, near_obstacles_category), dim=1)  # Shape: [obstacle_num, 3]
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
        if near_obstacles_in_leader_frame != None:
            for agent_index in range(num_agents):
                agent_pos = nominal_formation_tensor[agent_index, :2]  # Get the position part
                for obstacle_index in range(num_obstacles):
                    obstacle_pos = near_obstacles_in_leader_frame[obstacle_index, :2]  # Get the position part
                    distance = torch.norm(agent_pos - obstacle_pos)
                    if distance <= threshold_distance:  # Check if within threshold distance
                        # Add edges from agent to obstacle
                        edge_index.append([agent_index, num_agents + obstacle_index])  # Agent to obstacle
                        edge_index.append([num_agents + obstacle_index, agent_index])  # Obstacle to agent
                        edge_attr.append([distance])  # Edge type 1 for agent-obstacle
                        edge_attr.append([distance])
        # Connect each pair of nominal formation agents
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Add edges between agents
                agent_pos_i = nominal_formation_tensor[i, :2]  # Get the position part
                agent_pos_j = nominal_formation_tensor[j, :2]  # Get the position part

                distance = torch.norm(agent_pos_j - agent_pos_i)
                if distance <= 0.8:  # Check if within threshold distance
                    # Add edges from agent to obstacle
                    # print("add edges index")
                    edge_index.append([i,  j])  # Agent to obstacle
                    edge_index.append([j, i])  # Obstacle to agent
                    edge_attr.append([distance])  # Edge type 1 for agent-obstacle
                    edge_attr.append([distance])
        # Convert edge index and edge attributes to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Shape: [num_edges, 1]

        # Create the PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data



    def set_last_policy_output(self, output):
        self.last_policy_output = copy.deepcopy(output)

    def observation(self, agent: Agent):

        goal_poses = []
        goal_rot_poses = []
        current_agent_index = self.world.agents.index(agent)
        agent_num = len(self.world.agents)
        current_agent_index = self.world.agents.index(agent)
        graph_list = []
        
        if current_agent_index == 0:
            agent_num = len(self.world.agents)
            nominal_formation_tensor = torch.zeros((self.world.batch_dim, agent_num, 2), device=self.device)
            
            # Define nominal positions for agents (assuming 5 agents)
            # nominal_positions_x = [0.0, -0.3536, -0.3536, -0.7071, -0.7071]
            # nominal_positions_y = [0.0, 0.35366, -0.3536, 0.7071, -0.7071]
            nominal_positions_x = [0.0, -0.5534992, -0.5534992, -0.95179105, -0.95179105]
            nominal_positions_y = [0.0, 0.35284618, -0.35284618, 0.7946659, -0.7946659]
            for i, nomi_agent in enumerate(self.world.agents):
                nominal_formation_tensor[:, i, 0] = nominal_positions_x[i]
                nominal_formation_tensor[:, i, 1] = nominal_positions_y[i]
            previous_positions = self.agent_history.get_previous_positions()  # [num_agents, history_length, 2]


            for d in range(self.world.batch_dim):
                obstacle_manager = self.obstacle_manager_list[d]
                leader_pos = self.leader_robot.state.pos[d, :]  # Shape: [2]
                leader_rot = self.leader_robot.state.rot[d]      # Scalar (radians)
                
                agent_prev_positions = previous_positions[d, :, :, :] 
                # Transform previous positions relative to leader's frame
                # Translation: agent_prev_positions - leader_pos
                translated_prev_positions = agent_prev_positions - leader_pos.unsqueeze(0).unsqueeze(1)  # [agent_num, history_length, 2]
                
                # Rotation: Apply rotation matrix
                cos_theta = torch.cos(leader_rot)
                sin_theta = torch.sin(leader_rot)
                # print("sin_theta shape:{}".format(sin_theta.shape))
                rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                                [sin_theta, cos_theta]], device=self.device)  # [2, 2]
                
                # Apply rotation to each agent's previous positions
                # Reshape for batch matrix multiplication
                agent_prev_positions_flat = translated_prev_positions.view(-1, 2)  # [agent_num * history_length, 2]
                transformed_prev_positions = torch.matmul(agent_prev_positions_flat, rotation_matrix)  # [agent_num * history_length, 2]
                transformed_prev_positions = transformed_prev_positions.view(agent_num, -1, 2)  # [agent_num, history_length, 2]
                
                # Retrieve transformed previous positions for all agents
                transformed_prev_positions_batch = transformed_prev_positions  # [agent_num, history_length, 2]



                near_obstacles = obstacle_manager.get_near_obstacles(leader_pos.cpu(), 1.5)
                # near_obstacles: list of tensors, each of shape [2]
                
                if len(near_obstacles) != 0:
                    near_obstacles_tensor = torch.stack(near_obstacles)  # Shape: [num_obstacles, 2]
                    
                    # Translate obstacles relative to leader's position
                    translated_obstacles = near_obstacles_tensor - leader_pos  # Shape: [num_obstacles, 2]
                    
                    
                    # Apply rotation: (num_obstacles, 2) x (2, 2) -> (num_obstacles, 2)
                    near_obstacles_in_leader_frame = torch.matmul(translated_obstacles , rotation_matrix)
                    
                    # current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], near_obstacles_in_leader_frame, transformed_prev_positions_batch)
                    current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], near_obstacles_in_leader_frame)

                else:
                    # current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], None, transformed_prev_positions_batch)
                    current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], None)
                
                graph_list.append(current_graph)

            return self.data_list_to_tensors(graph_list)



        
        else:
            relative_to_other_robots_pose = []
            relative_to_other_formation_pose = []

            for i in range(len(self.world.agents)):
                # formation_goals_positions.append(self.formation_goals_landmark[i].state.pos[0])
                relative_to_other_formation_pose.append(self.formation_goals_landmark[i].state.pos - agent.state.pos)
            

            relative_to_leader_pose = [self.leader_robot.state.pos - agent.state.pos]
        
            for a in self.world.agents:
                if a != agent:
                    relative_to_other_robots_pose.append(a.state.pos - agent.state.pos)
        
            observation_tensor = torch.cat(
                relative_to_leader_pose + relative_to_other_robots_pose  ,
                dim=-1)
            # print("agent {} obs:{} shape:{}".format(current_agent_index, observation_tensor, observation_tensor.shape))

            return observation_tensor

    
      
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
        
        # if current_agent_index == 0:
        #     agent_num = len(self.world.agents)
        #     nominal_formation_tensor = torch.zeros((self.world.batch_dim, agent_num, 2), device=self.device)
            
        #     # Define nominal positions for agents (assuming 5 agents)
        #     # nominal_positions_x = [0.0, -0.3536, -0.3536, -0.7071, -0.7071]
        #     # nominal_positions_y = [0.0, 0.35366, -0.3536, 0.7071, -0.7071]
        #     nominal_positions_x = [0.0, -0.5534992, -0.5534992, -0.95179105, -0.95179105]
        #     nominal_positions_y = [0.0, 0.35284618, -0.35284618, 0.7946659, -0.7946659]
        #     for i, nomi_agent in enumerate(self.world.agents):
        #         nominal_formation_tensor[:, i, 0] = nominal_positions_x[i]
        #         nominal_formation_tensor[:, i, 1] = nominal_positions_y[i]
        #     previous_positions = self.agent_history.get_previous_positions()  # [num_agents, history_length, 2]


        #     for d in range(self.world.batch_dim):
        #         obstacle_manager = self.obstacle_manager_list[d]
        #         leader_pos = self.leader_robot.state.pos[d, :]  # Shape: [2]
        #         leader_rot = self.leader_robot.state.rot[d]      # Scalar (radians)
                
        #         agent_prev_positions = previous_positions[d, :, :, :] 
        #         # Transform previous positions relative to leader's frame
        #         # Translation: agent_prev_positions - leader_pos
        #         translated_prev_positions = agent_prev_positions - leader_pos.unsqueeze(0).unsqueeze(1)  # [agent_num, history_length, 2]
                
        #         # Rotation: Apply rotation matrix
        #         cos_theta = torch.cos(leader_rot)
        #         sin_theta = torch.sin(leader_rot)
        #         # print("sin_theta shape:{}".format(sin_theta.shape))
        #         rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
        #                                         [sin_theta, cos_theta]], device=self.device)  # [2, 2]
                
        #         # Apply rotation to each agent's previous positions
        #         # Reshape for batch matrix multiplication
        #         agent_prev_positions_flat = translated_prev_positions.view(-1, 2)  # [agent_num * history_length, 2]
        #         transformed_prev_positions = torch.matmul(agent_prev_positions_flat, rotation_matrix)  # [agent_num * history_length, 2]
        #         transformed_prev_positions = transformed_prev_positions.view(agent_num, -1, 2)  # [agent_num, history_length, 2]
                
        #         # Retrieve transformed previous positions for all agents
        #         transformed_prev_positions_batch = transformed_prev_positions  # [agent_num, history_length, 2]



        #         near_obstacles = obstacle_manager.get_near_obstacles(leader_pos.cpu(), 1.5)
        #         # near_obstacles: list of tensors, each of shape [2]
                
        #         if len(near_obstacles) != 0:
        #             near_obstacles_tensor = torch.stack(near_obstacles)  # Shape: [num_obstacles, 2]
                    
        #             # Translate obstacles relative to leader's position
        #             translated_obstacles = near_obstacles_tensor - leader_pos  # Shape: [num_obstacles, 2]
                    
                    
        #             # Apply rotation: (num_obstacles, 2) x (2, 2) -> (num_obstacles, 2)
        #             near_obstacles_in_leader_frame = torch.matmul(translated_obstacles , rotation_matrix)
                    
        #             # current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], near_obstacles_in_leader_frame, transformed_prev_positions_batch)
        #             current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], near_obstacles_in_leader_frame)

        #         else:
        #             # current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], None, transformed_prev_positions_batch)
        #             current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], None)
                
        #         graph_list.append(current_graph)
            
            # End of batch_dim loop
            # for d in range(self.world.batch_dim):
            #     current_graph = self.single_graph_from_data(nominal_formation_tensor[d, :, :], None)
            #     graph_list.append(current_graph)

        # ============================
        # Transform optimized_target_pos
        # ============================

        # Translate agent positions relative to leader's position
        translated_agent_pos = agent.state.pos - self.leader_robot.state.pos  # Shape: [batch_dim, 2]

        # Rotate agent positions into leader's frame
        # Assume self.leader_robot.state.rot is a tensor of shape [batch_dim]
        cos_theta = torch.cos(self.leader_robot.state.rot).unsqueeze(-1)  # Shape: [batch_dim, 1]
        sin_theta = torch.sin(self.leader_robot.state.rot).unsqueeze(-1)  # Shape: [batch_dim, 1]
        
        # Define rotation matrices for each batch
        rotation_matrices = torch.cat([cos_theta, sin_theta, -sin_theta, cos_theta], dim=1).reshape(-1, 2, 2)  # Shape: [batch_dim, 2, 2]
        
        # Apply rotation: [batch_dim, 2, 2] x [batch_dim, 2, 1] -> [batch_dim, 2, 1]
        optimized_target_pos = torch.bmm(rotation_matrices, translated_agent_pos.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_dim, 2]

        return {
            # "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            # "final_rew": self.final_rew,
            # "agent_collisions": agent.agent_collision_rew,
            # "formation_goal": agent.goal.state.pos,
            # "formation_main_rew":self.formation_maintain_rew,
            "optimized_target_pos": optimized_target_pos,  # Transformed to leader's frame
            # "graph_list": graph_list,
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

if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
