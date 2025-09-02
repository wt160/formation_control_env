#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Union

import torch

import vmas.simulator.core
import vmas.simulator.utils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Sensor(ABC):
    def __init__(self, world: vmas.simulator.core.World):
        super().__init__()
        self._world = world
        self._agent: Union[vmas.simulator.core.Agent, None] = None

    @property
    def agent(self) -> Union[vmas.simulator.core.Agent, None]:
        return self._agent

    @agent.setter
    def agent(self, agent: vmas.simulator.core.Agent):
        self._agent = agent

    @abstractmethod
    def measure(self):
        raise NotImplementedError

    @abstractmethod
    def render(self, env_index: int = 0) -> "List[Geom]":
        raise NotImplementedError

    def to(self, device: torch.device):
        raise NotImplementedError


class Lidar(Sensor):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        angle_start: float = 0.0,
        angle_end: float = 2 * torch.pi,
        n_rays: int = 8,
        max_range: float = 1.0,
        entity_filter: Callable[[vmas.simulator.core.Entity], bool] = lambda _: True,
        render_color: vmas.simulator.utils.Color = vmas.simulator.utils.Color.GRAY,
    ):
        super().__init__(world)
        if (angle_start - angle_end) % (torch.pi * 2) < 1e-5:
            angles = torch.linspace(
                angle_start, angle_end, n_rays + 1, device=self._world.device
            )[:n_rays]
        else:
            angles = torch.linspace(
                angle_start, angle_end, n_rays, device=self._world.device
            )

        self._angles = angles.repeat(self._world.batch_dim, 1)
        self._max_range = max_range
        self._last_measurement = None
        self._entity_filter = entity_filter
        self._render_color = render_color
        self._num_rays = n_rays

    def to(self, device: torch.device):
        self._angles = self._angles.to(device)

    @property
    def entity_filter(self):
        return self._entity_filter

    @entity_filter.setter
    def entity_filter(
        self, entity_filter: Callable[[vmas.simulator.core.Entity], bool]
    ):
        self._entity_filter = entity_filter


    def measure(self):
        agent_pos = self.agent.state.pos  # Expected Shape: [batch_dim_env, 2]
        agent_rot = self.agent.state.rot.squeeze(-1) # Expected Shape: [batch_dim_env]
        
        batch_dim_env = agent_pos.shape[0]
        if batch_dim_env == 0: 
            return torch.empty(0, self._num_rays, device=self.device) # Use self._num_rays
            
        # self._angles is 1D [self._num_rays]
        # agent_rot is 1D [batch_dim_env]
        # Broadcasting:
        # self._angles.unsqueeze(0) -> [1, self._num_rays]
        # agent_rot.unsqueeze(1) -> [batch_dim_env, 1]
        # Result: world_angles_batched -> [batch_dim_env, self._num_rays]
        world_angles_batched = self._angles.unsqueeze(0) + agent_rot.unsqueeze(1)
        
        # agent_pos_expanded: [batch_dim_env, self._num_rays, 2]
        agent_pos_expanded = agent_pos.unsqueeze(1).expand(-1, self._num_rays, 2)

        origins_flat = agent_pos_expanded.reshape(-1, 2) # Shape: [batch_dim_env * self._num_rays, 2]
        angles_flat = world_angles_batched.reshape(-1)   # Shape: [batch_dim_env * self._num_rays]
        
        # env_indices_for_rays_flat maps each ray in the flattened batch to its original environment index
        env_indices_for_rays_flat = torch.arange(batch_dim_env).repeat_interleave(self._num_rays)

        all_rays_distances_flat = self._world.cast_ray(
            ray_origins=origins_flat,
            ray_world_angles=angles_flat,
            bitmap_env_indices_for_rays=env_indices_for_rays_flat, 
            max_range=self._max_range,
            casting_entity=self.agent, 
            entity_filter=self.entity_filter
        )
        
        measurement = all_rays_distances_flat.view(batch_dim_env, self._num_rays)
        # print("measure:{}".format(measurement))
        self._last_measurement = measurement
        return measurement



    # def measure(self):
    #     dists = []
    #     import time
    #     print("measure angles:{}".format(self._angles))
    #     measure_start = time.time()
    #     for angle in self._angles.unbind(1):
    #         print("angle:{}".format(angle))
    #         dists.append(
    #             self._world.cast_ray(
    #                 self.agent,
    #                 angle + self.agent.state.rot.squeeze(-1),
    #                 max_range=self._max_range,
    #                 entity_filter=self.entity_filter,
    #             )
    #         )
    #     print("measure time:{}".format(time.time() - measure_start))
    #     measurement = torch.stack(dists, dim=1)
    #     # print("measure:{}".format(measurement))
    #     self._last_measurement = measurement
    #     return measurement

    def render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering
        # return
        # print("render!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("render angles:{}".format(self._angles))
        geoms: List[rendering.Geom] = []
        if self._last_measurement is not None:
            for angle, dist in zip(
                self._angles.unbind(1), self._last_measurement.unbind(1)
            ):
                angle = angle[env_index] + self.agent.state.rot.squeeze(-1)[env_index]
                ray = rendering.Line(
                    (0, 0),
                    (dist[env_index], 0),
                    width=0.005,
                )
                xform = rendering.Transform()
                xform.set_translation(*self.agent.state.pos[env_index])
                xform.set_rotation(angle)
                ray.add_attr(xform)
                # if dist < 2.45:
                ray_circ = rendering.make_circle(0.1)
                # else:
                    # ray_circ = rendering.make_circle(0.02)

                ray_circ.set_color(*vmas.simulator.utils.Color.RED.value)
                xform = rendering.Transform()
                rot = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
                pos_circ = (
                    self.agent.state.pos[env_index] + rot * dist.unsqueeze(1)[env_index]
                )
                xform.set_translation(*pos_circ)
                ray_circ.add_attr(xform)
                # print("1")
                geoms.append(ray)
                geoms.append(ray_circ)
        else:
            print("last_measurement is None")
            input("last_measurement None")
        return geoms