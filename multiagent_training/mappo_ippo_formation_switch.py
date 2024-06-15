# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import hydra
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torch.distributions import Normal
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from utils.logging import init_logging, log_evaluation, log_training, save_video
from utils.utils import DoneTransform


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

@hydra.main(version_base="1.1", config_path=".", config_name="mappo_ippo_formation_switch")
def train(cfg: "DictConfig"):  # noqa: F821
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # Create env and env_test
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    print("env:{}".format(env))

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    print("env action keys:{}".format(env.action_keys))
    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env_test._init_rendering()
    env_test.viewer.set_bounds(
                torch.tensor(-5),
                torch.tensor(5),
                torch.tensor(-5),
                torch.tensor(5),
            )

    hidden_dim = 64
    output_dim = 6  # Local target position (x, y)
    num_layers = 3  # Number of transformer encoder layers
    nhead = 8  # Number of attention heads
    print("env action dim:{}".format(env.action_spec.shape[-1]))
    gnn_actor_net =  torch.nn.Sequential(
        GINPolicyNetwork(hidden_dim, output_dim, num_layers, nhead),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    ).cuda()

    gnn_policy_module = TensorDictModule(
        gnn_actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale") ],
        # out_keys=[env.action_key],
    )
    print(env.unbatched_action_spec[("agents", "action")].space.low / 2)
    print(env.unbatched_action_spec[("agents", "action")].space.high /2 )
    
    gnn_policy = ProbabilisticActor(
        module=gnn_policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        # distribution_class=Normal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[("agents", "action")].space.low / 2,
            "max": env.unbatched_action_spec[("agents", "action")].space.high /2 ,
        },
        return_log_prob=True,
    )
    actor_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            # n_agent_outputs = 2* 2,
            n_agents=env.n_agents,
            centralised=True,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[("agents", "action")].space.low / 2,
            "max": env.unbatched_action_spec[("agents", "action")].space.high / 2,
        },
        return_log_prob=True,
    )

    # Critic
    critic_output_dim = 1
    critic_module = GINCriticNetwork(hidden_dim, critic_output_dim, num_layers, nhead)


    module = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=cfg.model.centralised_critic,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation")],
    )
    value_module = value_module.to(cfg.train.device)
    collector = SyncDataCollector(
        env,
        gnn_policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # Loss
    loss_module = ClipPPOLoss(
        actor_network=gnn_policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        # print("tensordict_data:{}".format(tensordict_data))
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()
                print("subdata:{}".format(subdata.shape))
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=gnn_policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )
                evaluation_time = time.time() - evaluation_start
                save_checkpoint(gnn_policy, "test_policy_{}.pth".format(i))
                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()




import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor

# def batch_from_dense_to_ptg(x, fov=0.35 * torch.pi, max_distance=10.0):
#     batch_size, num_agents, feature_length = x.shape
#     data_list = []

#     for b in range(batch_size):
#         node_features = x[b]  # Shape: (num_agents, feature_length)
#         positions = node_features[:, 0:2]  # Extracting positions
#         rotations = node_features[:, 2]    # Extracting rotations
#         edge_index = []
#         edge_attr = []

#         for i in range(num_agents):
#             for j in range(num_agents):
#                 if i != j:
#                     dx = positions[j, 0] - positions[i, 0]
#                     dy = positions[j, 1] - positions[i, 1]
#                     distance = torch.sqrt(dx**2 + dy**2)
#                     angle = torch.atan2(dy, dx) - rotations[i]
#                     angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize angle to [-pi, pi]

#                     if -fov <= angle <= fov and distance <= max_distance:
#                         edge_index.append([i, j])
#                         edge_attr.append([distance])  # Use the distance directly

#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.empty((0, 1), dtype=torch.float32)
#         # print("edge_index:{}".format(edge_index))
#         data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
#         data_list.append(data)

#     batch = Batch.from_data_list(data_list)
#     return batch

def batch_from_dense_to_ptg(x, fov=0.35 * torch.pi, max_distance=10.0):
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

class GINPolicyNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, nhead, lidar_dim=20, processed_lidar_dim=20):
        super(GINPolicyNetwork, self).__init__()

        # LiDAR preprocessing layers
        self.lidar_fc = torch.nn.Sequential(
            torch.nn.Linear(lidar_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, processed_lidar_dim),  # Ensure the final output dimension matches processed_lidar_dim
            torch.nn.ReLU()
        )

        # Define GINConv layers
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(processed_lidar_dim + 6, hidden_dim),  # Adjust input_dim to include preprocessed LiDAR features
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2)

        # Transformer encoder layers
        # print("hidden_dim:{}".format(hidden_dim))
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Fully connected layer to output local target positions for each node
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        # Gather input

        batch_size, num_agents, feature_length = data.shape

        # Separate original features and LiDAR data
        original_features, lidar_data = data[:, :, :6], data[:, :, 6:]

        # Preprocess LiDAR data
        lidar_features = self.lidar_fc(lidar_data)

        # Concatenate original features with preprocessed LiDAR features
        node_features = torch.cat([original_features, lidar_features], dim=-1)
        # print(f"Node features shape: {node_features.shape}")  # Debug print

        # Convert dense tensor data to graph batch for the entire batch
        # node_features = node_features.view(-1, node_features.shape[-1])  # Flatten the batch dimension
        # batch_graph = batch_from_dense_to_ptg(node_features.view(batch_size, num_agents, -1))
        batch_graph = batch_from_dense_to_ptg(node_features)
        batch_graph = batch_graph.to("cuda:0")
        x, edge_index, edge_attr = batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr


        # print("x shape:{}".format(x.shape))
        # print("before conv1 edge_index:{}".format(edge_index))

        # Apply GINConv layers
        if edge_index.size(0) > 0:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        else:
            x = self.conv1(x, torch.zeros((2, 0), dtype=torch.long, device=x.device))
            x = F.relu(x)
            x = self.conv2(x, torch.zeros((2, 0), dtype=torch.long, device=x.device))
            x = F.relu(x)

        # Apply transformer encoder layers
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        # print("x shape after transformer:{}".format(x.shape))
        # Reshape x to (batch_size, num_agents, hidden_dim)
        x = x.view(batch_size, num_agents, -1)

        # Output local target positions for each node
        out = self.fc(x)  # Each node's hidden representation is passed through the fully connected layer
        # print("policy out:{}".format(out))
        # print("policy out shape:{}".format(out.shape))
        return out

def batch_from_dense_to_ptg_critic(x, fov=0.35 * torch.pi, max_distance=10.0):
    critic_start = time.time()
    batch_size, num_agents, feature_length = x.shape

    # Extract positions and rotations
    positions = x[..., 0:2]  # Shape: (batch_size, num_agents, 2)
    rotations = x[..., 2]    # Shape: (batch_size, num_agents)

    # Create a mask to avoid self-loops
    mask = ~torch.eye(num_agents, dtype=torch.bool, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_agents, num_agents)

    # Compute pairwise differences
    dx = positions[:, :, 0].unsqueeze(2) - positions[:, :, 0].unsqueeze(1)  # Shape: (batch_size, num_agents, num_agents)
    dy = positions[:, :, 1].unsqueeze(2) - positions[:, :, 1].unsqueeze(1)  # Shape: (batch_size, num_agents, num_agents)
    distances = torch.sqrt(dx**2 + dy**2)  # Shape: (batch_size, num_agents, num_agents)

    # Apply the mask to avoid self-loops
    valid_edges_mask = mask  # Shape: (batch_size, num_agents, num_agents)

    # Generate all possible edge combinations
    all_indices = torch.arange(num_agents, device=x.device)
    src_indices, dst_indices = torch.meshgrid(all_indices, all_indices)
    src_indices = src_indices.flatten()
    dst_indices = dst_indices.flatten()

    # Repeat for all batches
    batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(num_agents * num_agents)
    src_indices = src_indices.repeat(batch_size)
    dst_indices = dst_indices.repeat(batch_size)

    # Flatten the valid_edges_mask and distances for filtering
    flat_valid_edges_mask = valid_edges_mask.flatten()
    flat_distances = distances.flatten()

    # Filter based on the valid_edges_mask
    valid_src_indices = src_indices[flat_valid_edges_mask]
    valid_dst_indices = dst_indices[flat_valid_edges_mask]
    valid_batch_indices = batch_indices[flat_valid_edges_mask]
    valid_distances = flat_distances[flat_valid_edges_mask]

    # Create a list of Data objects
    data_list = []
    for b in range(batch_size):
        batch_mask = valid_batch_indices == b
        batch_src = valid_src_indices[batch_mask]
        batch_dst = valid_dst_indices[batch_mask]
        batch_edge_index = torch.stack([batch_src, batch_dst], dim=0)
        batch_edge_attr = valid_distances[batch_mask].unsqueeze(1)
        data = Data(x=x[b], edge_index=batch_edge_index, edge_attr=batch_edge_attr)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    print("critic time:{}".format(time.time() - critic_start))
    return batch


# def batch_from_dense_to_ptg_critic(x, fov=0.35 * torch.pi, max_distance=10.0):
#     critic_start = time.time()
#     batch_size, num_agents, feature_length = x.shape
#     data_list = []
#     for b in range(batch_size):
#         node_features = x[b].clone()
#         positions = node_features[:, 0:2] # Ensure it's a regular tensor
#         rotations = node_features[:, 2]  # Ensure Clone to make it a regular tensor
#         edge_index = []
#         edge_attr = []

#         for i in range(num_agents):
#             for j in range(num_agents):
#                 if i != j:
#                     dx = positions[j, 0] - positions[i, 0]
#                     dy = positions[j, 1] - positions[i, 1]
#                     distance = torch.sqrt(dx**2 + dy**2)
#                     # angle = torch.atan2(dy, dx) - rotations[i]
#                     # angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize angle to [-pi, pi]
#                     # print("angle:{}".format(angle))


#                     # if (-fov <= angle <= fov):
#                     # if (distance <= max_distance):
#                     edge_index.append([i, j])
#                     edge_attr.append([distance])  # Use the distance directly

#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         # edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
#         # print("edge_index:{}".format(edge_index))
#         # edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.empty((0, 1), dtype=torch.float32)

#         data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
#         data_list.append(data)

#     batch = Batch.from_data_list(data_list)
#     print("critic time:{}".format(time.time() - critic_start))
#     return batch

class GINCriticNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, nhead, lidar_dim=20, processed_lidar_dim=20):
        super(GINCriticNetwork, self).__init__()

        # LiDAR preprocessing layers
        self.lidar_fc = torch.nn.Sequential(
            torch.nn.Linear(lidar_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, processed_lidar_dim),  # Ensure the final output dimension matches processed_lidar_dim
            torch.nn.ReLU()
        )

        # Define GINConv layers
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(processed_lidar_dim + 7, hidden_dim),  # Adjust input_dim to include preprocessed LiDAR features
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2)

        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Fully connected layer to output values for each node
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        if data.dim() == 4:  # Scenario 1: (batch_size, episode_steps, num_agents, feature_length)
            batch_size, episode_steps, num_agents, feature_length = data.shape
            # Separate original features and LiDAR data
            original_features, lidar_data = data[:, :, :, :7], data[:, :, :, 7:]
        elif data.dim() == 3:  # Scenario 2: (batch_size, num_agents, feature_length)
            batch_size, num_agents, feature_length = data.shape
            episode_steps = 1  # Set episode_steps to 1 for this case
            # Separate original features and LiDAR data
            original_features, lidar_data = data[:, :, :7], data[:, :, 7:]
        else:
            raise ValueError("Unexpected input shape: {}".format(data.shape))

        # Preprocess LiDAR data
        lidar_features = self.lidar_fc(lidar_data)

        # Concatenate original features with preprocessed LiDAR features
        node_features = torch.cat([original_features, lidar_features], dim=-1)

        # Flatten the batch and episode dimensions for graph processing
        if episode_steps > 1:
            node_features = node_features.view(-1, num_agents, node_features.shape[-1])
        else:
            node_features = node_features.view(batch_size, num_agents, -1)

        # Convert dense tensor data to graph batch
        batch_graph = batch_from_dense_to_ptg_critic(node_features) if episode_steps > 1 else batch_from_dense_to_ptg(node_features.view(batch_size, num_agents, -1))
        batch_graph = batch_graph.to("cuda:0")
        x, edge_index, edge_attr = batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr

        # Apply GINConv layers
        if edge_index.size(1) > 0:  # Check if there are any edges
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        else:
            x = self.conv1(x, torch.zeros((2, 0), dtype=torch.long, device=x.device))
            x = F.relu(x)
            x = self.conv2(x, torch.zeros((2, 0), dtype=torch.long, device=x.device))
            x = F.relu(x)

        # Apply transformer encoder layers
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)

        # Reshape x back to (batch_size * episode_steps, num_agents, hidden_dim) for episode-based input
        if episode_steps > 1:
            x = x.view(batch_size * episode_steps, num_agents, -1)
        else:
            x = x.view(batch_size, num_agents, -1)

        # Output values for each node
        out = self.fc(x)  # Each node's hidden representation is passed through the fully connected layer

        # Reshape out to (batch_size, episode_steps, num_agents, output_dim) for episode-based input
        if episode_steps > 1:
            out = out.view(batch_size, episode_steps, num_agents, -1)
        else:
            out = out.view(batch_size, num_agents, -1)

        return out





@hydra.main(version_base="1.1", config_path=".", config_name="mappo_ippo_formation_switch")
def evaluation(cfg: "DictConfig"):  # noqa: F821
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device
    checkpoint_path = cfg.eval.checkpoint_path
    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=1,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env_test._init_rendering()
    env_test.viewer.set_bounds(
                torch.tensor(-5),
                torch.tensor(5),
                torch.tensor(-5),
                torch.tensor(5),
            )
    # print("env_test:{}".format(env_test))
    # print("env obs spec:{}".format(env_test.observation_spec))

    # print("env action spec:{}".format(env_test.action_spec))

    edge_input_dim = 1  # For example: distance between agents
    hidden_dim = 64
    output_dim = 6  # Local target position (x, y)
    num_layers = 3  # Number of transformer encoder layers
    nhead = 8  # Number of attention heads
    print("env action dim:{}".format(env_test.action_spec.shape[-1]))
    gnn_actor_net =  torch.nn.Sequential(
        GINPolicyNetwork(hidden_dim, output_dim, num_layers, nhead),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    ).cuda()

    gnn_policy_module = TensorDictModule(
        gnn_actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale") ],
        # out_keys=[env.action_key],
    )
    gnn_policy = ProbabilisticActor(
        module=gnn_policy_module,
        spec=env_test.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env_test.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env_test.unbatched_action_spec[("agents", "action")].space.low,
            "max": env_test.unbatched_action_spec[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

    # gnn_policy_module()
    # gnn_policy = ProbabilisticActor(
    #     module=gnn_policy_module,
    #     spec=env_test.unbatched_action_spec,
    #     in_keys=[("agents", "loc"), ("agents", "scale")],
    #     out_keys=[env_test.action_key],
    #     distribution_class=TanhNormal,
    #     distribution_kwargs={
    #         "min": env_test.unbatched_action_spec[("agents", "action")].space.low,
    #         "max": env_test.unbatched_action_spec[("agents", "action")].space.high,
    #     },
    #     return_log_prob=True,
    # )

    gnn_policy.load_state_dict(torch.load(checkpoint_path))





    actor_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env_test.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env_test.action_spec.shape[-1],
            n_agents=env_test.n_agents,
            centralised=False,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env_test.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env_test.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env_test.unbatched_action_spec[("agents", "action")].space.low,
            "max": env_test.unbatched_action_spec[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
        )
    # evaluation_start = time.time()
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        env_test.frames = []
        rollouts = env_test.rollout(
            max_steps=500,
            policy=gnn_policy,
            callback=rendering_callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
            # We are running vectorized evaluation we do not want it to stop when just one env is done
        )
        # evaluation_time = time.time() - evaluation_start
        save_video(rollouts, env_test)

        # log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)
if __name__ == "__main__":
    train()
    # checkpoint_path = "test.pth"
    # evaluation()
