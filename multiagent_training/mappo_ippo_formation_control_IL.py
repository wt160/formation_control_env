# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
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
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from scipy.optimize import linear_sum_assignment

from utils.logging import init_logging, log_evaluation, log_training, save_video, save_video_from_frames
from utils.utils import DoneTransform

def collect_from_expert_policy(env, expert_policy_fn, device, max_steps=50, frames_per_batch=1000, max_speed=1, max_rotation=0.1 ):
    all_observations = []
    all_expert_actions = []
    print("env:{}".format(env))
    print("max_steps:{}".format(max_steps))
    print("frames_per_batch:{}".format(frames_per_batch))
    tensordict = env.reset()
    collected_frames = 0
    print("frames_per_batch:{}".format(frames_per_batch))
    while collected_frames < frames_per_batch:
        print("collected frames:{}".format(collected_frames))
        tensordict = env.reset()
        for i in range(max_steps):

            if i == 0:
                observations = tensordict["agents", "observation"].to(device)
                formation_goals_positions = tensordict["agents", "info", "formation_goal"].to(device)
            else:
                observations = tensordict["next", "agents", "observation"].to(device)
                formation_goals_positions = tensordict["next", "agents", "info", "formation_goal"].to(device)
                rewards = tensordict["next", "agents","reward"]
                mean_reward = rewards.mean()
                print("mean reward:{}".format(mean_reward))
            print("obs shape:{}".format(observations.shape))
            actions = expert_policy_fn(observations, formation_goals_positions, device, max_speed, max_rotation)
            tensordict["agents", "action"] = actions

            new_tensordict = env.step(tensordict)

            all_observations.append(observations.cpu())
            all_expert_actions.append(actions.cpu())
            collected_frames += observations.shape[0]
            print("collected_frams:{}".format(collected_frames))
            if new_tensordict["done"].any():
                print("done break")
                break
            tensordict = new_tensordict.clone()

        if collected_frames >= frames_per_batch:
            break

    all_observations = torch.cat(all_observations, dim=0).to(device)
    all_expert_actions = torch.cat(all_expert_actions, dim=0).to(device)

    return all_observations, all_expert_actions

def train_initial_policy(policy, train_obs, train_exp_actions, val_obs, val_exp_actions, optimizer, scheduler, batch_size, num_epochs, patience, min_delta):
    train_dataset = TensorDataset(train_obs, train_exp_actions)
    val_dataset = TensorDataset(val_obs, val_exp_actions)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        policy.train()
        for obs_batch, expert_action_batch in train_dataloader:
            optimizer.zero_grad()
            predicted_actions = policy(obs_batch)
            loss = nn.MSELoss()(predicted_actions[0], expert_action_batch)
            print(f"Init Imitation Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, expert_action_batch in val_dataloader:
                predicted_actions = policy(obs_batch)
                batch_loss = nn.MSELoss()(predicted_actions[0], expert_action_batch)
                val_loss += batch_loss.item()
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(policy, "best_policy.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break



def evaluate_expert(env, device, max_steps=100, max_speed=1, max_rotation=0.1):
    tensordict = env.reset()
    frames = []
    total_rewards = []

    for i in range(max_steps):
        print("step {}".format(i))
        if i == 0:
            observations = tensordict["agents", "observation"].to(device)
            formation_goals_positions = tensordict["agents", "info", "formation_goal"].to(device)
        else:
            observations = tensordict["next", "agents", "observation"].to(device)
            formation_goals_positions = tensordict["next", "agents", "info", "formation_goal"].to(device)

        actions = expert_policy(observations, formation_goals_positions, device, max_speed, max_rotation)
        tensordict["agents", "action"] = actions
        # print("actions:{}".format(actions))
        new_tensordict = env.step(tensordict)
        # print("new_tensordict:{}".format(new_tensordict))
        # print("agent pos:{}".format(new_tensordict["next", "agents", "info", "agent_pos"]))
        frames.append(env.render(mode="rgb_array"))  # Assuming the env has a render method

        rewards = new_tensordict["next", "agents", "info", "formation_main_rew"].cpu().numpy()
        print("reward:{}".format(rewards))
        total_rewards.append(rewards)
        if new_tensordict["done"].any():
            break
        tensordict = new_tensordict.clone()

    # Save video
    save_video_from_frames(frames, env)

    # Save rewards
    np.save("expert_policy_rewards.npy", np.array(total_rewards))

    return np.sum(total_rewards)

def expert_policy(observations, formation_goals_positions, device, max_speed=1.0, max_rotation=0.1):
    # print("obs shape:{}".format(observations.shape))
    print("obs:{}".format(observations))
    batch_size, num_agents, obs_size = observations.shape
    actions = torch.zeros((batch_size, num_agents, 3), device=device)
    print()
    for b in range(batch_size):
        batch_observations = observations[b]
        batch_goals = formation_goals_positions[b]

        # Calculate the cost matrix based on the distance between observations and formation goals
        cost_matrix = torch.cdist(batch_observations[:, :2], batch_goals)
        
        # Solve the assignment problem to minimize the travel distance
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

        # Calculate the actions to move towards the assigned goals
        for i in range(num_agents):
            # print("col_ind:{}".format(col_ind))
            goal = batch_goals[col_ind[i]]
            # print("goal:{}".format(goal))
            current_pos = batch_observations[i, :2]
            # print("current pos:{}".format(current_pos))
            current_rot = batch_observations[i, 2]
            action = goal - current_pos

            # Normalize the action to ensure it's within the max speed limit
            norm = torch.norm(action)
            if norm > max_speed:
                action = action / norm * max_speed
            print("actions:{}".format(action))
            # Calculate the required rotation to face the target
            target_angle = torch.atan2(action[1], action[0])
            rotation_force = target_angle - current_rot
            rotation_force = (rotation_force + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize to [-pi, pi]

            # Clip the rotation force to the maximum rotation
            if rotation_force > max_rotation:
                rotation_force = max_rotation
            elif rotation_force < -max_rotation:
                rotation_force = -max_rotation

            actions[b, i, :2] = action
            actions[b, i, 2] = rotation_force

    return actions

def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
def save_tensors(file_path, *tensors):
    torch.save(tensors, file_path)

def load_tensors(file_path):
    return torch.load(file_path)

def imitation_learning(cfg, env, policy, optimizer):
    num_agents = cfg.env.scenario.n_agents
    batch_size = cfg.train.minibatch_size
    num_epochs = cfg.imitation.num_epochs
    iterations_per_batch = cfg.imitation.iterations_per_batch
    patience = cfg.imitation.patience
    min_delta = cfg.imitation.min_delta

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = np.inf
    patience_counter = 0
    expert_data_file = "../expert_data.pt"

    if os.path.exists(expert_data_file):
        all_observations, all_expert_actions = load_tensors(expert_data_file)
        print("Loaded expert data from file")
    else:
        all_observations, all_expert_actions = collect_from_expert_policy(env, expert_policy, cfg.train.device, 45, frames_per_batch=20*cfg.collector.frames_per_batch)
        print("Collected data from expert policy")
        save_tensors(expert_data_file, all_observations, all_expert_actions)
        print(f"Saved expert data to {expert_data_file}")
    
    #cfg.collector.frames_per_batch
    print("after collect from expert policy")
     # Split data into training and validation sets
    train_obs, val_obs, train_exp_actions, val_exp_actions = train_test_split(
        all_observations.view(-1, all_observations.shape[-1]),
        all_expert_actions.view(-1, all_expert_actions.shape[-1]),
        test_size=0.2,
        random_state=cfg.seed
    )
    print("policy device:{}".format(policy.device))
    print("train_obs device:{}".format(train_obs.device))
    train_obs = train_obs.view(-1, num_agents, 26)
    val_obs = val_obs.view(-1, num_agents, 26)
    train_exp_actions = train_exp_actions.view(-1, num_agents, 3)
    val_exp_actions = val_exp_actions.view(-1, num_agents, 3)

    if os.path.exists("../best_policy.pth"):
        policy.load_state_dict(torch.load("../best_policy.pth"))
    else:
    # Train the initial policy with the collected expert data
        train_initial_policy(policy, train_obs, train_exp_actions, val_obs, val_exp_actions, optimizer, scheduler, batch_size, num_epochs, patience, min_delta)
        policy.load_state_dict(torch.load("best_policy.pth"))

    # Load the best policy  

    # Use the new collector from the beginning
    

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        policy.train()

        current_observations = []
        current_expert_actions = []

        imitate_collector = SyncDataCollector(
            env,
            policy,  # Use the current policy for data collection
            device=cfg.env.device,
            storing_device=cfg.train.device,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
        )

        for i, tensordict_data in enumerate(imitate_collector):
            observations = tensordict_data["agents", "observation"]
            formation_goals_positions = tensordict_data["agents", "info", "formation_goal"]
            rewards = tensordict_data["next", "agents","reward"]
            reward_slice = rewards[:, 50:, :, :]
            mean_reward = reward_slice.mean()
            batch_size_1, batch_size_2, num_agents, obs_size = observations.shape
            combined_batch_size = batch_size_1 * batch_size_2
            
            combined_observations = observations.view(combined_batch_size, num_agents, obs_size)
            combined_goals = formation_goals_positions.view(combined_batch_size, num_agents, 2)
            
            print("Combined obs shape:", combined_observations.shape)  # Should print [combined_batch_size, num_agents, obs_size]
            print("Combined formation goal shape:", combined_goals.shape)  # Should print [combined_batch_size, num_agents, 2]
            
            print("reward mean:{}".format(mean_reward))

            expert_actions = expert_policy(combined_observations, combined_goals, cfg.train.device)
            current_observations.append(combined_observations.cpu())
            current_expert_actions.append(expert_actions.cpu())
            if i >= 2:  # Limit the number of iterations per epoch
                break

        current_observations = torch.cat(current_observations, dim=0).to(policy.device)
        current_expert_actions = torch.cat(current_expert_actions, dim=0).to(policy.device)

        train_obs = torch.cat([train_obs, current_observations], dim=0)
        train_exp_actions = torch.cat([train_exp_actions, current_expert_actions], dim=0)
        print("train_obs size:{}".format(train_obs.shape))
        train_dataset = TensorDataset(train_obs, train_exp_actions)
        val_dataset = TensorDataset(val_obs, val_exp_actions)

        train_dataloader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8192, shuffle=False)

        for obs_batch, expert_action_batch in train_dataloader:
            print("obs_batch")
            for _ in range(iterations_per_batch):
                optimizer.zero_grad()
                predicted_actions = policy(obs_batch)
                # print("predicted_actions[0] shape:{}".format(predicted_actions[0].shape))
                # print("predicted_actions[1] shape:{}".format(predicted_actions[1].shape))
                # print("predicted_actions[2] shape:{}".format(predicted_actions[2].shape))
                # print("predicted_actions[3] shape:{}".format(predicted_actions[3].shape))

                # print("predicted_actions:{}".format(predicted_actions))
                # print("expert actions:{}".format(expert_action_batch))
                loss = nn.MSELoss()(predicted_actions[0], expert_action_batch)
                print(f"Imitation Loss: {loss.item()}")
                loss.backward()
                optimizer.step()

        scheduler.step()

        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, expert_action_batch in val_dataloader:
                predicted_actions = policy(obs_batch)
                batch_loss = nn.MSELoss()(predicted_actions[0], expert_action_batch)
                val_loss += batch_loss.item()
        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(policy, "best_policy.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        



@hydra.main(version_base="1.1", config_path=".", config_name="mappo_ippo_formation_control")
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
        # num_envs = 1,
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
        # num_envs=cfg.eval.evaluation_episodes,
        num_envs = 1,
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
    # Policy
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
            "max": env.unbatched_action_spec[("agents", "action")].space.high /2,
        },
        return_log_prob=True,
    )



    actor_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
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
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[("agents", "action")].space.low,
            "max": env.unbatched_action_spec[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

    # Critic
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




    # Check if the IL policy exists
    il_policy_module = TensorDictModule(
        gnn_actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    il_policy = ProbabilisticActor(
        module=il_policy_module,
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
    il_policy_path = "../il_policy.pth"
    if os.path.exists(il_policy_path):
        print("Loading IL policy from checkpoint...")
        il_policy.load_state_dict(torch.load(il_policy_path))
    else:
        # Imitation Learning
        
        il_optim = torch.optim.Adam(il_policy.parameters(), lr=cfg.train.lr)

        imitation_learning(cfg, env ,il_policy, il_optim)
        print("Imitation learning done")
        # Transfer learned weights from IL to PPO policy
        save_checkpoint(il_policy, il_policy_path)
    gnn_policy.load_state_dict(il_policy.state_dict())


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
        print("data_view:{}".format(data_view))
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
            and i != 0
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
                print("rollouts:{}".format(rollouts))
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
        batch_graph = batch_from_dense_to_ptg(node_features, fov=torch.pi, max_distance=10.0)
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


@hydra.main(version_base="1.1", config_path=".", config_name="mappo_ippo_formation_control")
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
    # env_test = TransformedEnv(
    #     env_test,
    #     RewardSum(in_keys=[env_test.reward_key], out_keys=[("agents", "episode_reward")]),
    # )
    print("env obs spec:{}".format(env_test.observation_spec))

    print("env action spec:{}".format(env_test.unbatched_action_spec))
    print("n_agent_inputs:{}".format(env_test.observation_spec["agents", "observation"].shape[-1]))
    print("n_agents_outputs:{}".format(2 * env_test.action_spec.shape[-1]))
    print("n_agents:{}".format(env_test.n_agents))
    print("spec:{}".format(env_test.unbatched_action_spec))
    print("out_keys:{}".format(env_test.action_key))
    print("min:{}".format(env_test.unbatched_action_spec[("agents", "action")].space.low))
    print("max:{}".format(env_test.unbatched_action_spec[("agents", "action")].space.high))
    evaluate_expert(env_test, cfg.train.device)
    input("evaluate expert")
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

    policy.load_state_dict(torch.load(checkpoint_path))
    print("policy:{}".format(policy))
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
            policy=policy,
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
