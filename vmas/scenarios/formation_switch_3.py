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
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
import networkx as nx


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

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -0.00005)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 5
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
            self.formation_goals_modified[i] = Landmark(
                name=f"modified_goal_{i}",
                collide=False,
                movable=True,
                rotatable=True,
                color=modified_color,
            )
            world.add_landmark(self.formation_goals_modified[i])
            
            self.formation_goals_landmark[i] = Landmark(
                name=f"formation goal{i}",
                collide=False,
                movable=True,
                rotatable=True,
                color=Color.GRAY,
            )
            world.add_landmark(self.formation_goals_landmark[i])
            
        world.add_landmark(self.formation_center)

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

        self.obstacle_pattern = 1
        # self.create_obstacles(self.obstacle_pattern, world)


        formation_control_policy_checkpoint = "/home/ysc/multi_formation/multiagent/outputs/2024-05-31/11-12-20/test_policy_75.pth"
        
        actor_obs_dim = 28
        actor_output_action_dim = 3
        self.actor_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=actor_obs_dim,
                n_agent_outputs=2 * actor_output_action_dim,     
                n_agents=self.n_agents,
                centralised=False,
                share_params=True,
                device="cpu" if not torch.cuda.device_count() else "cuda:0",
                depth=2,
                num_cells=256,
                activation_class=nn.Tanh,
            ),
            NormalParamExtractor(),
        )
        policy_module = TensorDictModule(
            self.actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        self.lower_policy = ProbabilisticActor(
            module=policy_module,
            # spec=CompositeSpec(agents: CompositeSpec(action: BoundedTensorSpec(shape=torch.Size([self.n_agents, actor_output_action_dim]),space=ContinuousBox(
                # low=Tensor(shape=torch.Size([self.n_agents, actor_output_action_dim]), device=cuda:0, dtype=torch.float32, contiguous=True),
                # high=Tensor(shape=torch.Size([self.n_agents, actor_output_action_dim]), device=cuda:0, dtype=torch.float32, contiguous=True)),
            # device=cuda:0,
            # dtype=torch.float32,
            # domain=continuous), device=cuda:0, shape=torch.Size([self.n_agents])), device=cuda:0, shape=torch.Size([])),
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[('agents', 'action')],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": torch.tensor([[-1., -1.],[-1., -1.],[-1., -1.],[-1., -1.],[-1., -1.]], device='cuda:0'),
                "max": torch.tensor([[1., 1.],[1., 1.],[1., 1.],[1., 1.],[1., 1.]], device='cuda:0'),
            },
            return_log_prob=True,
        )
        # self.lower_policy.load_state_dict(torch.load(formation_control_policy_checkpoint))


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
            agent.agent_collision_rew = agent.pos_rew.clone()
            agent.target_collision_rew = agent.pos_rew.clone()
            agent.target_diff_rew = agent.pos_rew.clone()
            agent.output_action_rew = agent.pos_rew.clone()
            agent.graph_connect_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals

            agent.target_pos = copy.deepcopy(self.formation_goals_landmark[i])
            agent.local_target_pos = copy.deepcopy(self.formation_goals_landmark[i])
            agent.goal = copy.deepcopy(self.formation_goals_landmark[i])
            agent.last_target_pos = copy.deepcopy(self.formation_goals_landmark[i])

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.graph_connect_rew = torch.zeros(batch_dim, device=device) 
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
            self.box_width = 0.4
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
                            3.3
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
                elif is_one.any():
                    pos = torch.tensor(
                    [
                        1,
                        -3.3,
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
                            3,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ).repeat(i.shape[0], 1)
                    return pos
                elif is_one.any():
                    pos = torch.tensor(
                    [
                        0,
                        -3,
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
            (-self.world_semidim, -3),
            (-self.world_semidim+2, self.world_semidim-2),
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
        # self.spawn_obstacles(self.obstacle_pattern, env_index)
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
    
    def env_process_action_collectively(self):

        for a_index, a in enumerate(self.world.agents):
            # print("agent net action:{}".format(a.action.u))
            modified_local_goal = copy.deepcopy(a.action.u) 
            # print("modified_local_goal shape:{}".format(modified_local_goal.shape))
            #     )
            a.local_target_pos.state.pos = modified_local_goal[:,:2]
            a.local_target_pos.state.rot = modified_local_goal[:,2]
            modified_goal = a.state.pos + modified_local_goal[:,:2]
            # print("agent {} modified_goal:{}".format(a_index, modified_goal))

            modified_goal_rot = a.state.rot + modified_local_goal[:, 2].unsqueeze(1)
            self.formation_goals_modified[a_index].set_pos(modified_goal, batch_index = None)
            self.formation_goals_modified[a_index].set_rot(modified_goal_rot, batch_index = None)
            # a.last_target_pos.state.pos = copy.deepcopy(a.target_pos.state.pos)
            a.target_pos.set_pos(modified_goal, batch_index = None)
            # a.last_target_pos.state.rot = copy.deepcopy(a.target_pos.state.rot)
            a.target_pos.set_rot(modified_goal_rot, batch_index=None)

        lower_policy_obs_list = []
        for a in self.world.agents:
            lower_policy_obs_single = self.lower_policy_observation(a)
            # print("lower_policy_obs_single shape:{}".format(lower_policy_obs_single.shape))
            lower_policy_obs_list.append(lower_policy_obs_single)
        # print("lower_policy_obs shape:{}".format(lower_policy_obs.shape))
        stacked_tensor = torch.stack(lower_policy_obs_list, dim=0)
        # print("lower_policy_obs_list:{}".format(lower_policy_obs_list)) 
        combined_tensor = stacked_tensor.permute(1, 0, 2).to("cuda:0")
        # combined_tensor = torch.cat(lower_policy_obs_list, dim=1).to("cuda:0")
        # print("combined_tensor:{}".format(combined_tensor))
        # Use the shape of one of the single observations to determine batch_dim and num_agents
        batch_dim = lower_policy_obs_list[0].shape[0]
        num_agents = len(self.world.agents)

        # Reshape the combined tensor to [batch_dim, num_agents, feature_dim]
        self.lower_policy_obs = combined_tensor.view(batch_dim, num_agents, -1)
        # print("self.lower_policy_obs:{}".format(self.lower_policy_obs))
        # print("self.lower_policy_obs shape:{}".format(self.lower_policy_obs.shape))
        # print("--------------------end---------------------")
        # print("--------------------end---------------------")
        # print("--------------------end---------------------")

        pre_joint_action = self.actor_net(self.lower_policy_obs)
        # print("pre oint_action shape:{}".format(pre_joint_action))

        # joint_action = self.lower_policy(self.lower_policy_obs)
        # print("joint_action :{}".format(pre_joint_action))
        # print("joint_action shape:{}".format(pre_joint_action[0].shape))
        for a_index, a in enumerate(self.world.agents):
            # print("a.action.u device:{}".format(a.action.u.device))
            a.action.u = pre_joint_action[0][:,a_index,:]
            # print("after low a.action.u:{}".format(a.action.u))
            # print("a.action.u shape:{}".format(a.action.u.shape))
    def process_action(self, agent: Agent):
        from vmas.simulator.utils import TorchUtils

        # call the formation_control policy, and outputs robot force control action. 
        # then redefine agent.action.u with the output robot force control action.
        
        # agent.action.u = self.lower_policy(self.lower_policy_obs)
        
        
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
            elif formation_movement == "horizental":
                ###move from left to right, test formation's ability to cross through tunnel
                t = self.t / 30
                if self.t < 10000:
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


                    formation_type = "ren_shape"
                    # formation_type = "line"
                    # formation_type = "rectangle"
                    # formation_type = "vertical_line"
                    # if self.t < 200:
                    #     formation_type = "rectangle"
                    # else:
                    #     formation_type = "ren_shape"
                    
                    if formation_type == "ren_shape":
                        #大雁人字形
                        self.formation_goals[i][0] = self.formation_center_pos[0] + math.cos(self.formation_center_pos[2] + angles[i-1]) * dists[i-1]
                        self.formation_goals[i][1] = self.formation_center_pos[1] + math.sin(self.formation_center_pos[2] + angles[i-1]) * dists[i-1]
                    elif formation_type == "vertical_line":
                        dists = [0.5, 1, -0.5, -1]
                        self.formation_goals[i][0] = self.formation_center_pos[0] 
                        self.formation_goals[i][1] = self.formation_center_pos[1] + dists[i-1]
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
            self.graph_connect_rew[:] = 0
            self.formation_maintain_rew[:] = 0

            self.formation_maintain_rew = self.agent_reward_graph_formation_maintained()
            self.graph_connect_rew = self.agent_reward_graph_connectivity_continuous()
            # print("graph connect rew:{}".format(self.graph_connect_rew))
            for a in self.world.agents:

                self.pos_rew += self.agent_reward(a)
                a.target_collision_rew[:] = 0
                a.target_diff_rew[:] = 0
            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            # for i, a in enumerate(self.world.agents):
            #     for j, b in enumerate(self.world.agents):
            #         if i <= j:
            #             continue
            #         if self.world.collides(a, b):
            #             distance = self.world.get_distance(a, b)
            #             a.agent_collision_rew[
            #                 distance <= self.min_collision_distance
            #             ] += self.agent_collision_penalty
            #             b.agent_collision_rew[
            #                 distance <= self.min_collision_distance
            #             ] += self.agent_collision_penalty

            # for i, a in enumerate(self.world.agents):
            #     for obs in self.obstacles:
            #         if self.world.collides(a, obs):
            #             distance = self.world.get_distance(a, obs)
            #             a.agent_collision_rew[
            #                 distance <= self.min_collision_distance
            #             ] += self.agent_collision_penalty

            # for i, a in enumerate(self.world.agents):
            #     for j, b in enumerate(self.world.agents):
            #         if i <= j:
            #             continue
            #         if self.world.collides(self.formation_goals_modified[i], self.formation_goals_modified[j]):
            #             distance = self.world.get_distance(self.formation_goals_modified[i], self.formation_goals_modified[j])
            #             a.target_collision_rew[
            #                 distance <= self.min_collision_distance
            #             ] += self.agent_collision_penalty
            #             b.target_collision_rew[
            #                 distance <= self.min_collision_distance
            #             ] += self.agent_collision_penalty

            # for i, a in enumerate(self.world.agents):
            #     for obs in self.obstacles:
            #         # if self.world.collides(self.formation_goals_modified[i], obs):
            #         distance = self.world.get_distance(self.formation_goals_modified[i], obs)
            #         # print("distance:{}".format(distance))
            #         # a.target_collision_rew[
            #         #     distance <= self.min_collision_distance
            #         # ] += self.agent_collision_penalty
            #         a.target_collision_rew[
            #             distance <= self.min_collision_distance
            #         ] += distance[distance <= self.min_collision_distance]*0.001
                # a.target_diff_rew = -10*torch.linalg.vector_norm(a.target_pos.state.pos - a.last_target_pos.state.pos, dim=-1)
                # a.output_action_rew = -10*torch.linalg.vector_norm(a.local_target_pos.state.pos, dim=-1)
                # print("a.output_rew shape:{}".format(torch.linalg.vector_norm(a.action.u, dim=-1).shape))
        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        # return 20*pos_reward + self.graph_connect_rew + self.final_rew  + agent.target_collision_rew + agent.target_diff_rew
        # return pos_reward + self.graph_connect_rew + self.final_rew  + agent.target_collision_rew + agent.output_action_rew
        return self.graph_connect_rew + self.formation_maintain_rew


        # return pos_reward + self.final_rew + agent.agent_collision_rew + agent.target_collision_rew
    def agent_reward_graph_connectivity_continuous(self):
        target_positions_dict = {}
        target_rotations_dict = {}
        for a_index, agent in enumerate(self.world.agents):
            target_positions_dict[a_index] = agent.target_pos.state.pos 
            target_rotations_dict[a_index] = agent.target_pos.state.rot
            
        batch_size = next(iter(target_positions_dict.values())).shape[0]
        graph_connect_rew = torch.zeros(batch_size, device=self.device) 
        num_agents = len(target_positions_dict)
        D = 0.6  # Example distance threshold
        FOV_min = -0.35 * np.pi
        FOV_max = 0.35 * np.pi
        alpha = 1.0  # Weight for distance difference
        beta = 1.0   # Weight for angle difference
        
        def calculate_weighted_distance(pos_i, rot_i, pos_j, rot_j):
            rel_pos = pos_j - pos_i
            d_ij = torch.norm(rel_pos).item()
            distance_diff = max(d_ij - D, 0)

            theta_ij = torch.atan2(rel_pos[1], rel_pos[0]).item() - rot_i.item()
            theta_ij = np.arctan2(np.sin(theta_ij), np.cos(theta_ij))
            angle_diff = max(abs(theta_ij) - (FOV_max / 2), 0)

            weighted_sum = alpha * distance_diff + beta * angle_diff
            return weighted_sum

        for batch_idx in range(batch_size):
            G = nx.Graph()
            for i in range(num_agents):
                G.add_node(i)
            
            # Add edges based on FOV and distance criteria
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        pos_i = target_positions_dict[i][batch_idx]
                        pos_j = target_positions_dict[j][batch_idx]
                        rot_i = target_rotations_dict[i][batch_idx]
                        
                        rel_pos = pos_j - pos_i
                        d_ij = torch.norm(rel_pos).item()
                        
                        if d_ij <= D:
                            theta_ij = torch.atan2(rel_pos[1], rel_pos[0]).item() - rot_i.item()
                            theta_ij = np.arctan2(np.sin(theta_ij), np.cos(theta_ij))
                            
                            if FOV_min <= theta_ij <= FOV_max:
                                G.add_edge(i, j)

            # Calculate AgentDist(i) for each agent i
            agent_dists = []
            for i in range(num_agents):
                reconnect_dists = []
                for j in range(num_agents):
                    if i != j and not nx.has_path(G, i, j):
                        # Find the shortest weighted reconnect distance
                        shortest_reconnect_dist = float('inf')
                        for k in range(num_agents):
                            if k != i and nx.has_path(G, i, k):
                                for m in range(num_agents):
                                    if m != i and nx.has_path(G, j, m):
                                        dist_km = calculate_weighted_distance(target_positions_dict[k][batch_idx], target_rotations_dict[k][batch_idx],
                                                                            target_positions_dict[m][batch_idx], target_rotations_dict[m][batch_idx])
                                        if dist_km < shortest_reconnect_dist:
                                            shortest_reconnect_dist = dist_km
                        
                        # If no intermediate connections are found, calculate direct reconnect distance
                        if shortest_reconnect_dist == float('inf'):
                            for m in range(num_agents):
                                if m != i and m != j and nx.has_path(G, j, m):
                                    dist_im = calculate_weighted_distance(target_positions_dict[i][batch_idx], target_rotations_dict[i][batch_idx],
                                                                        target_positions_dict[m][batch_idx], target_rotations_dict[m][batch_idx])
                                    if dist_im < shortest_reconnect_dist:
                                        shortest_reconnect_dist = dist_im
                        if shortest_reconnect_dist == float('inf'):
                            shortest_reconnect_dist = calculate_weighted_distance(target_positions_dict[i][batch_idx], target_rotations_dict[i][batch_idx],
                                                                              target_positions_dict[j][batch_idx], target_rotations_dict[j][batch_idx])
                        reconnect_dists.append(shortest_reconnect_dist)
                if reconnect_dists:
                    agent_dists.append(np.mean(reconnect_dists))
            
            if agent_dists:
                avg_agent_dist = np.mean(agent_dists)
            else:
                avg_agent_dist = 0
            
            # Calculate the reward
            graph_connect_rew[batch_idx] = -avg_agent_dist*0.1
        
        return graph_connect_rew
    
    
    def agent_reward_graph_formation_maintained(self):
        target_positions_dict = {}
        target_rotations_dict = {}
        for a_index, agent in enumerate(self.world.agents):
            target_positions_dict[a_index] = agent.target_pos.state.pos
            target_rotations_dict[a_index] = agent.target_pos.state.rot
            
        batch_size = next(iter(target_positions_dict.values())).shape[0]
        graph_connect_rew = torch.zeros(batch_size, device=self.device) 
        num_agents = len(target_positions_dict)

        formation_goals_positions = []
        for i in range(num_agents):
            formation_goals_positions.append(self.formation_goals_landmark[i].state.pos[0])
        formation_goals_positions = torch.stack(formation_goals_positions)

        def geometric_center(positions):
            return torch.mean(positions, dim=0)

        for batch_idx in range(batch_size):
            target_positions = torch.stack([target_positions_dict[i][batch_idx] for i in range(num_agents)])
            formation_center = geometric_center(formation_goals_positions)
            target_center = geometric_center(target_positions)
            # print("formation_goals_positions shape:{}".format(formation_goals_positions.shape))
            # print("target_positions shape:{}".format(target_positions.shape))
            # print("target_center shape:{}".format(target_center.shape))
            # Center the positions
            centered_formation_positions = formation_goals_positions - formation_center
            centered_target_positions = target_positions - target_center
            # print("centered_target-positions shape:{}".format(centered_target_positions.shape))
            # Calculate the optimal assignment using the Hungarian algorithm (linear_sum_assignment)
            # print("centered_formation_positions shape:{}".format(centered_formation_positions.shape))
            cost_matrix = torch.cdist(centered_target_positions, centered_formation_positions)
            # print("cost_matrix shape:{}".format(cost_matrix.shape))
            # Ensure cost_matrix is 2D (num_agents x num_agents)
            if cost_matrix.dim() != 2:
                raise ValueError("Cost matrix must be 2D")
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            
            optimal_assignment_distances = cost_matrix[row_ind, col_ind]
            
            # Calculate the optimal scale factor
            optimal_scale_factor = (torch.norm(centered_target_positions[row_ind], dim=1) / torch.norm(centered_formation_positions[col_ind], dim=1)).mean()

            # Scale the formation positions
            scaled_formation_positions = centered_formation_positions * optimal_scale_factor

            # Calculate the rotation matrix using Singular Value Decomposition (SVD)
            H = torch.matmul(centered_target_positions[row_ind].T, scaled_formation_positions[col_ind])
            U, S, Vt = torch.linalg.svd(H)
            rotation_matrix = torch.matmul(U, Vt)

            # Rotate the formation positions
            rotated_formation_positions = torch.matmul(scaled_formation_positions, rotation_matrix.T)

            # Calculate the matching error
            matching_error = torch.norm(centered_target_positions[row_ind] - rotated_formation_positions[col_ind], dim=1).mean()


            center_matching_error = torch.norm(target_center - formation_center).item()

            # Calculate the reward
            max_reward = 1.0
            reward = max_reward - (matching_error.item() + center_matching_error)
            graph_connect_rew[batch_idx] = reward
            # Calculate the reward
        print("formation_maintain reward:{}".format(graph_connect_rew))
        return graph_connect_rew
    


    #only return 1 or -1 for connected or disconnected
    def agent_reward_graph_connectivity(self):
        target_positions_dict = {}
        target_rotations_dict = {}
        for a_index, agent in enumerate(self.world.agents):
            target_positions_dict[a_index] = agent.target_pos.state.pos
            target_rotations_dict[a_index] = agent.target_pos.state.rot
            
            # print("target_positions shape:{}".format(target_positions_dict[a_index].shape))
            # print("target_rotations shape:{}".format(target_rotations_dict[a_index].shape))
        
        batch_size = next(iter(target_positions_dict.values())).shape[0]
        graph_connect_rew = torch.zeros(batch_size, device=self.device) 
        num_agents = len(target_positions_dict)
        D = 0.2  # Example distance threshold
        FOV_min = -0.35 * np.pi
        FOV_max = 0.35 * np.pi
        
        for batch_idx in range(batch_size):
            adjacency_matrix = np.zeros((num_agents, num_agents))
            
            # Add edges based on FOV and distance criteria
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        pos_i = target_positions_dict[i][batch_idx]
                        pos_j = target_positions_dict[j][batch_idx]
                        rot_i = target_rotations_dict[i][batch_idx]
                        
                        rel_pos = pos_j - pos_i
                        d_ij = torch.norm(rel_pos).item()
                        
                        if d_ij <= D:
                            theta_ij = torch.atan2(rel_pos[1], rel_pos[0]).item() - rot_i.item()
                            theta_ij = np.arctan2(np.sin(theta_ij), np.cos(theta_ij))
                            
                            if FOV_min <= theta_ij <= FOV_max:
                                adjacency_matrix[i, j] = 1
            
            # Check if the graph is weakly connected using DFS
            def dfs(node, visited):
                stack = [node]
                while stack:
                    current = stack.pop()
                    if not visited[current]:
                        visited[current] = True
                        stack.extend(np.where(adjacency_matrix[current] == 1)[0])
            
            visited = np.zeros(num_agents, dtype=bool)
            dfs(0, visited)
            
            if all(visited):
                graph_connect_rew[batch_idx] = 1.0
            else:
                graph_connect_rew[batch_idx] = -1.0
            
        return graph_connect_rew




    



    # def agent_reward(self, agent: Agent):
    #     # Extract positions
    #     agent_reward_start = time.time()
    #     agent_positions = torch.stack([a.state.pos for a in self.world.agents])  # [num_agents, 1200, 2]
    #     goal_positions = torch.stack([g.state.pos for g in self.formation_goals_landmark.values()])  # [num_goals, 1200, 2]
    #     # print("goal_positions:{}".format(goal_positions))
    #     modified_positions = torch.stack([m.state.pos for m in self.formation_goals_modified.values()])
    #     # print("agent pos:{}".format(agent_positions))
    #     # print("goal_positions:{}".format(goal_positions))
    #     # print("agent_positions shape:{}".format(agent_positions.shape))
    #     # print("goal positions shape:{}".format(goal_positions.shape))
    #     num_agents = agent_positions.shape[0]
    #     num_goals = goal_positions.shape[0]
    #     num_envs = agent_positions.shape[1]  # 1200 environments
    #     distances = torch.zeros((num_agents, num_goals, num_envs), device=self.device)
    #     target_goal_distances = torch.zeros((num_agents, num_goals, num_envs), device=self.device)
    #     # Calculate distances
    #     for i in range(num_agents):
    #         for j in range(num_goals):
    #             # print("i,j:{},{}".format(i, j))
    #             # print("agent shape:{}".format(agent_positions[i, :, :].shape))
    #             # print("goal shape:{}".format(goal_positions[j, :, :].shape))
    #             # temp = modified_positions[i, :, :] - goal_positions[j, :, :]
    #             temp = agent_positions[i, :, :] - goal_positions[j, :, :]
    #             # print("minus shape:{}".format(temp.shape))
    #             distances[i, j, :] = torch.linalg.vector_norm(temp, dim=-1)
    #             temp_target_goal = modified_positions[i, :, :] - goal_positions[j, :, :]
    #             target_goal_distances[i, j, :] = torch.linalg.vector_norm(temp_target_goal, dim=-1)
    #     # print("distances:{}".format(distances))
    #     # Allocate storage for agent-specific rewards
    #     # agent_rewards = torch.zeros(num_envs)

    #     agent.pos_rew = torch.zeros(num_envs, device =self.device )

    #     # Get assignments from the batch_greedy_assignment

    #     assignments = self.batch_greedy_assignment(distances)
    #     # print("assignments:{}".format(assignments))
    #     # print("after assignment")
    #     # Get the assigned goal for the current agent in each environment
    #     current_agent_index = self.world.agents.index(agent)
    #     assigned_goals = assignments[:, current_agent_index]
    #     # print("assigned_goals:{}".format(assigned_goals))
    #     # print("num_envs:{}".format(num_envs))
    #     for e in range(num_envs):
    #         agent.goal.state.pos[e] = copy.deepcopy(self.formation_goals_landmark[assigned_goals[e].item()].state.pos[0])

    #     # print("agent goal shape:{}".format(agent.goal.state.pos))

        
    #     rows = torch.arange(num_envs)

    #     min_distances = target_goal_distances[current_agent_index, assigned_goals, rows]
    #     # print("min_distances:{}".format(min_distances))
    #     # Check which assignments were made (assigned_goal_index != -1)
    #     valid_assignments = assigned_goals != -1
        
    #     # Calculate whether the agent is on its goal for all environments
    #     agent_on_goal = min_distances < 0.05
    #     agent_on_goal = agent_on_goal & valid_assignments  # Only consider valid assignments
        
    #     # Calculate position shaping reward for all environments
    #     pos_shaping = min_distances * self.pos_shaping_factor
    #     pos_rew = agent.pos_shaping - pos_shaping
    #     pos_rew = torch.where(valid_assignments, pos_rew, torch.zeros_like(pos_rew))  # Set rewards to 0 where no assignment
        
    #     # Update agent's rewards and position shaping
    #     agent.pos_rew = pos_rew
    #     agent.pos_shaping = torch.where(valid_assignments, pos_shaping, agent.pos_shaping)  # Only update shaping where valid
        
    #     # Update on_goal status
    #     agent.on_goal = agent_on_goal
    #     agent_reward_time = time.time() - agent_reward_start

    #     # print("Agent reward computation time:{}".format(agent_reward_time))

    #     return agent.pos_rew

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
    def agent_reward(self, agent: Agent):
        current_agent_index = self.world.agents.index(agent)
        agent.goal.state.pos = copy.deepcopy(self.formation_goals_landmark[current_agent_index].state.pos)
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew



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


    def lower_policy_observation(self, agent: Agent):
        
        goal_poses = []
        goal_rot_poses = []
        observations = [agent.state.pos, agent.state.vel, agent.state.rot]

    # Add positions and velocities of all agents (including the current agent)
        # for a in self.world.agents:
        #     observations.append(a.state.pos)
        #     observations.append(a.state.vel)

        # if self.observe_all_goals:
        #     for g in self.formation_goals_landmark.values():
        #         goal_poses.append(agent.state.pos - g.state.pos.clone())
        # else:
        goal_poses.append(agent.state.pos - agent.target_pos.state.pos)
        goal_rot_poses.append(agent.state.rot - agent.target_pos.state.rot)
        observation_tensor = torch.cat(
            observations + goal_poses + goal_rot_poses +  (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1
        )
        return observation_tensor

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
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
            "target_collisions": agent.target_collision_rew,
            "target_diff": agent.target_diff_rew,
            "graph_connect_rew": self.graph_connect_rew,
            "formation_maintain_rew": self.formation_maintain_rew,

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




if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
