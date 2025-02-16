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
from torch_geometric.nn import GATConv
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
    device = cfg.env.device
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
    print(env.unbatched_action_spec[("agents", "action")].space.low / 2)
    print(env.unbatched_action_spec[("agents", "action")].space.high /2 )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    print("unbatchspec:{}".format(env.unbatched_action_spec))
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

    num_agents=5
    # sample_graph = dataset[0]
    in_channels = 3
    hidden_dim = 64
    output_dim = 6  # Assuming 2D positions

    # Initialize the model, loss function, and optimizer


    gnn_actor_net =  torch.nn.Sequential(
            GATModel(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=output_dim, num_agents=num_agents, max_obstacles= 100, num_node_features=3, num_edge_features=1),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        ).to(device)

    gnn_policy_module = TensorDictModule(
        gnn_actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
        # out_keys=["loc", "scale" ],
        # out_keys=[env.action_key],
    )
    # print(env.unbatched_action_spec[("agents", "action")].space.low / 2)
    # print(env.unbatched_action_spec[("agents", "action")].space.high /2 )

    gnn_policy = ProbabilisticActor(
        module=gnn_policy_module,
        # in_keys=["loc", "scale"],
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        # in_keys=[env.action_key],
        out_keys=[env.action_key],
        # out_keys=[("agents", 'action')],
        distribution_class=TanhNormal,
        # distribution_class=Normal,
        return_log_prob=True,
    ).to(device)







    
    # Critic
    critic_output_dim = 1
    critic_module = GATCriticModel(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=critic_output_dim, num_agents=num_agents, max_obstacles= 100, num_node_features=3, num_edge_features=1)
    critic_module = critic_module.to(device)

    value_module = ValueOperator(
        module=critic_module,
        in_keys=[("agents", "observation")],
    )
    # module = MultiAgentMLP(
    #     # n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    #     n_agent_inputs=10,
    #     n_agent_outputs=1,
    #     n_agents=env.n_agents,
    #     centralised=cfg.model.centralised_critic,
    #     share_params=cfg.model.shared_parameters,
    #     device=cfg.train.device,
    #     depth=2,
    #     num_cells=256,
    #     activation_class=nn.Tanh,
    # )
    # value_module = ValueOperator(
    #     module=module,
    #     in_keys=[("agents", "observation")],
    # )
    # value_module = value_module.to(cfg.train.device)
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
                # print("subdata:{}".format(subdata.shape))
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



def batch_from_dense_to_ptg(x, fov=0.35 * torch.pi, max_distance=10.0):
    batch_size, num_agents, feature_length = x.shape

    # Extract positions and rotations
    positions = x[..., 0:2]  # Shape: (batch_size, num_agents, 2)
    # rotations = x[..., 2]    # Shape: (batch_size, num_agents)

    # Create a mask to avoid self-loopstrain
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


# def single_tensor_to_data_list(tensor, max_obstacles, num_node_features, num_edge_features, num_agents):
#     """
#     Reconstructs a list of torch_geometric.Data objects from the serialized tensor.
#     Assumes that all graph data is serialized into the 0th agent's feature vector.

#     Args:
#         tensor (torch.Tensor): Serialized tensor of shape (batch_size, feature_length). Should be on the correct device.
#         max_obstacles (int): Maximum number of obstacles per graph.
#         num_node_features (int): Number of features per node (excluding category).
#         num_edge_features (int): Number of features per edge.
#         num_agents (int): Number of agents per graph.

#     Returns:
#         List[Data]: Reconstructed list of Data objects.
#     """
#     batch_size, agent_num, feature_length = tensor.shape
#     device = tensor.device
#     data_list = []
#     print("critic input tensor shape:{}".format(tensor.shape))
#     # Calculate feature_length per graph
#     expected_feature_length = (num_agents + max_obstacles) * num_node_features + max_obstacles * 2 + max_obstacles * num_edge_features
#     if feature_length < expected_feature_length:
#         raise ValueError(f"Serialized tensor feature_length ({feature_length}) is less than expected ({expected_feature_length}).")

#     for batch_idx in range(batch_size):
#         # Extract 0th agent's feature vector
#         graph_feature = tensor[batch_idx, 0]  # [feature_length]

#         # Extract node features
#         node_features_length = (num_agents + max_obstacles) * num_node_features
#         flattened_node_features = graph_feature[:node_features_length]
#         node_features = flattened_node_features.view(num_agents + max_obstacles, num_node_features)

#         # Extract edge indices
#         edge_index_length = max_obstacles * 2  # Each edge has two indices
#         flattened_edge_index = graph_feature[node_features_length:node_features_length + edge_index_length]
#         edge_index = flattened_edge_index.view(max_obstacles, 2).long()

#         # Extract edge attributes
#         edge_attr_start = node_features_length + edge_index_length
#         flattened_edge_attr = graph_feature[edge_attr_start:edge_attr_start + max_obstacles * num_edge_features]
#         edge_attr = flattened_edge_attr.view(max_obstacles, num_edge_features)

#         # Reconstruct edge_index and edge_attr by removing padding
#         # Create a mask to identify valid edges
#         max_nodes = num_agents + max_obstacles
#         valid_mask = (edge_index[:, 0] < max_nodes) & (edge_index[:, 1] < max_nodes)
#         print("edge_index shape:{}".format(edge_index.shape))
#         print("valid mask shape:{}".format(valid_mask.shape))
#         # Apply mask to edge_index and edge_attr
#         edge_index = edge_index[valid_mask].t().contiguous()
#         edge_attr = edge_attr[valid_mask]

#         # Create Data object
#         data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
#         data_list.append(data)

#     return data_list

def single_tensor_to_data_list(tensor, max_obstacles, num_node_features, num_edge_features, num_agents):
    """
    Reconstructs a list of torch_geometric.Data objects from the serialized tensor.
    Assumes that all graph data is serialized into the 0th agent's feature vector.

    Args:
        tensor (torch.Tensor): Serialized tensor of shape (batch_size, agent_num, feature_length).
                               Should be on the correct device.
        max_obstacles (int): Maximum number of obstacles per graph.
        num_node_features (int): Number of features per node (excluding category).
        num_edge_features (int): Number of features per edge.
        num_agents (int): Number of agents per graph.

    Returns:
        List[Data]: Reconstructed list of Data objects.
    """
    batch_size, agent_num, feature_length = tensor.shape
    device = tensor.device
    data_list = []

    # Calculate feature_length per graph
    expected_feature_length = (num_agents + max_obstacles) * num_node_features + \
        max_obstacles * 2 + max_obstacles * num_edge_features
    if feature_length < expected_feature_length:
        raise ValueError(
            f"Serialized tensor feature_length ({feature_length}) is less than expected ({expected_feature_length})."
        )

    for batch_idx in range(batch_size):
        # Extract 0th agent's feature vector
        graph_feature = tensor[batch_idx, 0]  # Shape: [feature_length]

        # Extract node features
        node_features_length = (num_agents + max_obstacles) * num_node_features
        flattened_node_features = graph_feature[:node_features_length]
        node_features = flattened_node_features.view(num_agents + max_obstacles, num_node_features)

        # Extract edge indices
        edge_index_length = max_obstacles * 2  # Each edge has two indices
        edge_index_start = node_features_length
        edge_index_end = edge_index_start + edge_index_length
        flattened_edge_index = graph_feature[edge_index_start:edge_index_end]
        edge_index = flattened_edge_index.view(2, max_obstacles).long()

        # Extract edge attributes
        edge_attr_start = edge_index_end
        edge_attr_end = edge_attr_start + max_obstacles * num_edge_features
        flattened_edge_attr = graph_feature[edge_attr_start:edge_attr_end]
        edge_attr = flattened_edge_attr.view(max_obstacles, num_edge_features)

        # Create a mask to identify valid edges
        max_nodes = num_agents + max_obstacles
        valid_mask = (edge_index[0, :] < max_nodes) & (edge_index[1, :] < max_nodes)  # Shape: [max_obstacles]

        # Use torch.where to set invalid edges to default values without control flow
        adjusted_edge_index = torch.where(
            valid_mask.unsqueeze(0),
            edge_index,
            torch.zeros_like(edge_index)
        )

        adjusted_edge_attr = torch.where(
            valid_mask.unsqueeze(-1),
            edge_attr,
            torch.zeros_like(edge_attr)
        )

        # Create Data object
        data = Data(
            x=node_features.to(device),
            edge_index=adjusted_edge_index.to(device),
            edge_attr=adjusted_edge_attr.to(device)
        )
        data_list.append(data)

    return data_list


# def single_tensor_to_data_list(tensor, max_obstacles, num_node_features, num_edge_features, num_agents):
#     """
#     Reconstructs a list of torch_geometric.Data objects from the serialized tensor.
#     Assumes that all graph data is serialized into the 0th agent's feature vector.
    
#     Args:
#         tensor (torch.Tensor): Serialized tensor of shape (batch_size, num_agents, feature_length).
#         max_obstacles (int): Maximum number of obstacles per graph.
#         num_node_features (int): Number of features per node (excluding category).
#         num_edge_features (int): Number of features per edge.
#         num_agents (int): Number of agents per graph.
    
#     Returns:
#         List[Data]: Reconstructed list of Data objects.
#     """
#     print("tensor device:{}".format(tensor.device))
#     batch_size, num_agents_tensor, feature_length = tensor.shape
#     data_list = []
    
#     # Calculate feature_length per graph
#     expected_feature_length = (num_agents + max_obstacles) * num_node_features + max_obstacles * 2 + max_obstacles * num_edge_features
#     if feature_length < expected_feature_length:
#         raise ValueError(f"Serialized tensor feature_length ({feature_length}) is less than expected ({expected_feature_length}).")
    
#     for batch_idx in range(batch_size):
#         # Extract 0th agent's feature vector
#         graph_feature = tensor[batch_idx, 0]  # [feature_length]
        
#         # Extract node features
#         node_features_length = (num_agents + max_obstacles) * num_node_features
#         flattened_node_features = graph_feature[:node_features_length]  # [num_agents + max_obstacles, num_node_features]
#         node_features = flattened_node_features.view(num_agents + max_obstacles, num_node_features)
        
#         # Extract edge indices
#         edge_index_length = max_obstacles * 2  # Each edge has two indices
#         flattened_edge_index = graph_feature[node_features_length:node_features_length + edge_index_length]
#         edge_index = flattened_edge_index.view(max_obstacles, 2).long()  # [max_obstacles, 2]
        
#         # Extract edge attributes
#         edge_attr_start = node_features_length + edge_index_length
#         flattened_edge_attr = graph_feature[edge_attr_start:edge_attr_start + max_obstacles * num_edge_features]
#         edge_attr = flattened_edge_attr.view(max_obstacles, num_edge_features)  # [max_obstacles, num_edge_features]
        
#         # Reconstruct edge_index and edge_attr by removing padding
#         valid_edges = []
#         valid_edge_attrs = []
#         for edge_idx in range(edge_index.size(0)):
#             src, dst = edge_index[edge_idx]
#             if src < (num_agents + max_obstacles) and dst < (num_agents + max_obstacles):
#                 # Assuming padding was done with zeros beyond valid nodes
#                 # If an edge connects to node 0 (agent), it might be valid, so you might need another padding value
#                 # Here, we consider all edges as valid except those connecting to padded nodes
#                 # Modify this condition based on your padding strategy
#                 if src >= num_agents and src >= num_agents + max_obstacles - (num_agents + max_obstacles):
#                     continue
#                 if dst >= num_agents and dst >= num_agents + max_obstacles - (num_agents + max_obstacles):
#                     continue
#                 valid_edges.append([src, dst])
#                 valid_edge_attrs.append(edge_attr[edge_idx])
        
#         if valid_edges:
#             edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()  # [2, num_valid_edges]
#             edge_attr = torch.stack(valid_edge_attrs)  # [num_valid_edges, num_edge_features]
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = torch.empty((0, num_edge_features), dtype=torch.float)
        
#         # Create Data object
#         data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
#         print("data device:{}".format(node_features.device))
#         data_list.append(data)
    
#     return data_list

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents, max_obstacles, num_node_features, num_edge_features):
        super(GATModel, self).__init__()
        self.num_agents = num_agents
        self.max_obstacles = max_obstacles
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

        # Global pooling layer
        self.pool = global_mean_pool

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16, hidden_channels*4),
            nn.ReLU()
            # nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(4*hidden_channels, out_channels)


    def forward(self, tensor):
        """
        Args:
            data (Batch): Batched PyG Data object containing multiple graphs.

        Returns:
            predicted_positions (torch.Tensor): Predicted positions for all agents in all graphs.
                                               Shape: [batch_size, num_agents, out_channels]
        """
        # print("tensor shape:{}".format(tensor.shape))
        batch_size, agent_num,  feature_length = tensor.shape
        device = tensor.device
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # torch.set_printoptions(threshold=torch.inf)
        data_list = single_tensor_to_data_list(
            tensor, 
            max_obstacles=self.max_obstacles, 
            num_node_features=self.num_node_features, 
            num_edge_features=self.num_edge_features,
            num_agents=self.num_agents
        )
        batch = Batch.from_data_list(data_list)
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        # print("x after conv1:{}".format(x))
        # print("after4 conv1 x shape:{}".format(x.shape))
        # input("0, 1")
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)
        # print("x:{}".format(x))
        # Global graph embedding
        graph_embedding = self.pool(x, batch.batch)  # Shape: [batch_size, hidden_channels]

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, batch.batch, batch.num_graphs)
        # print("agent_embedding:{}".format(agent_embeddings))
        # input("1")
        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)  # Shape: [batch_size*num_agents, hidden_channels]

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)  # Shape: [batch_size*num_agents, 2*hidden_channels]
        # print("combined shape:{}".format(combined.shape))
        # Fully connected layers
        combined = self.fc1(combined)
        combined = torch.relu(combined)
        predicted_positions = self.fc2(combined)  # Shape: [batch_size*num_agents, out_channels]
        # print("202 predicted positions:{}".format(predicted_positions))
        # Reshape to [batch_size, num_agents, out_channels]
        predicted_positions = predicted_positions.view(batch.num_graphs, self.num_agents, -1)
        return predicted_positions

    def extract_agent_embeddings(self, x, batch, batch_size):
        """
        Extracts agent node embeddings from the batched node features.

        Args:
            x (torch.Tensor): Node features after GIN layers. Shape: [total_nodes, hidden_channels]
            batch (torch.Tensor): Batch vector, which assigns each node to a specific graph. Shape: [total_nodes]
            batch_size (int): Number of graphs in the batch.

        Returns:
            agent_embeddings (torch.Tensor): Agent embeddings for all graphs. Shape: [batch_size * num_agents, hidden_channels]
        """
        agent_node_indices = []
        for graph_idx in range(batch_size):
            # Find node indices for the current graph
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            # Take the first `num_agents` nodes as agent nodes
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)
        
        # Concatenate all agent node indices
        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]  # Shape: [batch_size * num_agents, hidden_channels]
        return agent_embeddings

class GATCriticModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents, max_obstacles, num_node_features, num_edge_features):
        super(GATCriticModel, self).__init__()
        self.num_agents = num_agents
        self.max_obstacles = max_obstacles
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        # GAT layers
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean', add_self_loops=False
        )
        self.conv2 = GATConv(
            hidden_channels * 8, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean'
        )
        self.conv3 = GATConv(
            hidden_channels * 8, hidden_channels, heads=8, concat=True,
            edge_dim=1, fill_value='mean'
        )

        # Global pooling layer
        self.pool = global_mean_pool

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels * 16, hidden_channels * 4),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(4 * hidden_channels, out_channels)

    def forward(self, tensor):
        """
        Args:
            tensor: Input tensor of shape:
                - (batch_size, agent_num, feature_length)
                - (batch_size, collect_step_num, agent_num, feature_length)

        Returns:
            predicted_positions: Output tensor with appropriate shape.
        """
        device = tensor.device

        if tensor.ndim == 3:
            # Shape: (batch_size, agent_num, feature_length)
            batch_size, agent_num, feature_length = tensor.shape
            tensor_reshaped = tensor
            reshape_back = False
        elif tensor.ndim == 4:
            # Shape: (batch_size, collect_step_num, agent_num, feature_length)
            batch_size, collect_step_num, agent_num, feature_length = tensor.shape
            # Combine batch_size and collect_step_num into a single dimension
            tensor_reshaped = tensor.view(batch_size * collect_step_num, agent_num, feature_length)
            reshape_back = True
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        # Now, tensor_reshaped has shape (new_batch_size, agent_num, feature_length)
        new_batch_size = tensor_reshaped.size(0)

        # Process the tensor as before
        data_list = single_tensor_to_data_list(
            tensor_reshaped,
            max_obstacles=self.max_obstacles,
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
            num_agents=self.num_agents
        )

        batch = Batch.from_data_list(data_list).to(device)
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Global graph embedding
        graph_embedding = self.pool(x, batch.batch)  # Shape: [new_batch_size, hidden_channels]

        # Extract agent node embeddings
        agent_embeddings = self.extract_agent_embeddings(x, batch.batch, new_batch_size)

        # Repeat graph embedding for each agent
        graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)

        # Concatenate agent embeddings with graph embeddings
        combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)

        # Fully connected layers
        combined = self.fc1(combined)
        combined = torch.relu(combined)
        predicted_positions = self.fc2(combined)  # Shape: [new_batch_size * num_agents, out_channels]

        # Reshape back to original dimensions
        if reshape_back:
            # Reshape to [batch_size, collect_step_num, num_agents, out_channels]
            predicted_positions = predicted_positions.view(batch_size, collect_step_num, self.num_agents, -1)
        else:
            # Reshape to [batch_size, num_agents, out_channels]
            predicted_positions = predicted_positions.view(batch_size, self.num_agents, -1)

        return predicted_positions

    def extract_agent_embeddings(self, x, batch, batch_size):
        """
        Extracts agent node embeddings from the batched node features.

        Args:
            x (torch.Tensor): Node features after GAT layers. Shape: [total_nodes, hidden_channels]
            batch (torch.Tensor): Batch vector, assigning each node to a graph. Shape: [total_nodes]
            batch_size (int): Number of graphs in the batch.

        Returns:
            agent_embeddings (torch.Tensor): Shape: [batch_size * num_agents, hidden_channels]
        """
        agent_node_indices = []
        for graph_idx in range(batch_size):
            # Find node indices for the current graph
            node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
            # Take the first `num_agents` nodes as agent nodes
            agent_nodes = node_indices[:self.num_agents]
            agent_node_indices.append(agent_nodes)

        # Concatenate all agent node indices
        agent_node_indices = torch.cat(agent_node_indices, dim=0)
        agent_embeddings = x[agent_node_indices]
        return agent_embeddings
    

# class GATCriticModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_agents, max_obstacles, num_node_features, num_edge_features):
#         super(GATCriticModel, self).__init__()
#         self.num_agents = num_agents
#         self.max_obstacles = max_obstacles
#         self.num_node_features = num_node_features
#         self.num_edge_features = num_edge_features

#         # GAT layers
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean', add_self_loops=False)
#         self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')
#         self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, edge_dim=1, fill_value='mean')

#         # Global pooling layer
#         self.pool = global_mean_pool

#         # Fully connected layers
#         self.fc1 = nn.Sequential(
#             nn.Linear(hidden_channels * 16, hidden_channels*4),
#             nn.ReLU()
#             # nn.Dropout(p=0.5)
#         )
#         self.fc2 = nn.Linear(4*hidden_channels, out_channels)


#     def forward(self, tensor):
#         """
#         Args:
#             data (Batch): Batched PyG Data object containing multiple graphs.

#         Returns:
#             predicted_positions (torch.Tensor): Predicted positions for all agents in all graphs.
#                                                Shape: [batch_size, num_agents, out_channels]
#         """
#         print("critic input shape{}".format(tensor.shape))
#         batch_size, agent_num, feature_length = tensor.shape
#         device = tensor.device
#         # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         # torch.set_printoptions(threshold=torch.inf)
#         data_list = single_tensor_to_data_list(
#             tensor, 
#             max_obstacles=self.max_obstacles, 
#             num_node_features=self.num_node_features, 
#             num_edge_features=self.num_edge_features,
#             num_agents=self.num_agents
#         )
#         batch = Batch.from_data_list(data_list)
#         x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

#         x = self.conv1(x, edge_index, edge_attr.squeeze(dim=1))
#         x = torch.relu(x)
#         # print("x after conv1:{}".format(x))
#         # print("after4 conv1 x shape:{}".format(x.shape))
#         # input("0, 1")
#         x = self.conv2(x, edge_index, edge_attr)
#         x = torch.relu(x)
#         x = self.conv3(x, edge_index, edge_attr)
#         x = torch.relu(x)
#         # print("x:{}".format(x))
#         # Global graph embedding
#         graph_embedding = self.pool(x, batch.batch)  # Shape: [batch_size, hidden_channels]

#         # Extract agent node embeddings
#         agent_embeddings = self.extract_agent_embeddings(x, batch.batch, batch.num_graphs)
#         # print("agent_embedding:{}".format(agent_embeddings))
#         # input("1")
#         # Repeat graph embedding for each agent
#         graph_embedding_repeated = graph_embedding.repeat_interleave(self.num_agents, dim=0)  # Shape: [batch_size*num_agents, hidden_channels]

#         # Concatenate agent embeddings with graph embeddings
#         combined = torch.cat([agent_embeddings, graph_embedding_repeated], dim=1)  # Shape: [batch_size*num_agents, 2*hidden_channels]
#         # print("combined shape:{}".format(combined.shape))
#         # Fully connected layers
#         combined = self.fc1(combined)
#         combined = torch.relu(combined)
#         predicted_positions = self.fc2(combined)  # Shape: [batch_size*num_agents, out_channels]
#         # print("202 predicted positions:{}".format(predicted_positions))
#         # Reshape to [batch_size, num_agents, out_channels]
#         predicted_positions = predicted_positions.view(batch.num_graphs, self.num_agents, -1)
#         return predicted_positions

#     def extract_agent_embeddings(self, x, batch, batch_size):
#         """
#         Extracts agent node embeddings from the batched node features.

#         Args:
#             x (torch.Tensor): Node features after GIN layers. Shape: [total_nodes, hidden_channels]
#             batch (torch.Tensor): Batch vector, which assigns each node to a specific graph. Shape: [total_nodes]
#             batch_size (int): Number of graphs in the batch.

#         Returns:
#             agent_embeddings (torch.Tensor): Agent embeddings for all graphs. Shape: [batch_size * num_agents, hidden_channels]
#         """
#         agent_node_indices = []
#         for graph_idx in range(batch_size):
#             # Find node indices for the current graph
#             node_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
#             # Take the first `num_agents` nodes as agent nodes
#             agent_nodes = node_indices[:self.num_agents]
#             agent_node_indices.append(agent_nodes)
        
#         # Concatenate all agent node indices
#         agent_node_indices = torch.cat(agent_node_indices, dim=0)
#         agent_embeddings = x[agent_node_indices]  # Shape: [batch_size * num_agents, hidden_channels]
#         return agent_embeddings




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
            # n_agent_inputs=env_test.observation_spec["agents", "observation"].shape[-1],
            n_agent_inputs=10,
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
