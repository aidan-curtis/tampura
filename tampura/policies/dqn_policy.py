from __future__ import annotations

import itertools
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from tampura.policies.policy import Policy
from tampura.spec import ProblemSpec
from tampura.structs import Action, AliasStore, Belief, NoOp


class GymEnv:
    def __init__(self, belief: Belief, problem_spec: ProblemSpec, store: AliasStore):
        self.initial_belief = belief
        self.current_belief = belief
        self.problem_spec = problem_spec
        self.store = store  # Assumed static
        belief_vec = self.current_belief.vectorize()
        self.observation_space = spaces.Box(0, 1, belief_vec.shape)
        self.num_schemas = len(self.problem_spec.action_schemas)
        self.discrete_actions = []
        for a in self.problem_spec.action_schemas:
            for object_combo in itertools.product(*[store.type_dict[t] for t in a.input_types]):
                self.discrete_actions.append(Action(a.name, list(object_combo)))

        # Precompute a mapping from discrete action choice to state
        self.action_space = spaces.Discrete(len(self.discrete_actions))

    def reset(self, seed):
        self.current_belief = self.initial_belief
        return self.current_belief.vectorize(), {}

    def step(self, actions):
        """
        Args:
            actions: element of :attr:`action_space` Batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)
        """
        action: Action = self.discrete_actions[actions[0]]

        ab = self.current_belief.abstract(self.store)
        r = 0
        if action in self.problem_spec.applicable_actions(ab, self.store):
            abs = self.problem_spec.get_action_schema(action.name).effects_fn(
                action, self.current_belief, self.store
            )
            counts = np.array(list(abs.ab_counts.values()))

            sampled_ab = np.random.choice(list(abs.ab_counts.keys()), p=counts / sum(counts))
            r = self.problem_spec.get_reward(sampled_ab, self.store)
            sampled_b = np.random.choice(abs.belief_map[sampled_ab])
            self.current_belief = sampled_b

        return (np.array([self.current_belief.vectorize()]), [r], [r > 0], [False], [{}])


@dataclass
class DQNArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment."""
    seed: int = 1
    """Seed of the experiment."""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """If toggled, cuda will be enabled by default."""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases."""
    wandb_project_name: str = "cleanRL"
    """The wandb's project name."""
    wandb_entity: str = None
    """The entity (team) of wandb's project."""
    capture_video: bool = False
    """Whether to capture videos of the agent performances (check out `videos`
    folder)"""
    save_model: bool = False
    """Whether to save model into the `runs/{run_name}` folder."""
    upload_model: bool = False
    """Whether to upload the saved model to huggingface."""
    hf_entity: str = ""
    """The user or org name of the model repository from the Hugging Face
    Hub."""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """The id of the environment."""
    total_timesteps: int = 500000
    """Total timesteps of the experiments."""
    learning_rate: float = 2.5e-4
    """The learning rate of the optimizer."""
    num_envs: int = 1
    """The number of parallel game environments."""
    buffer_size: int = 10000
    """The replay memory buffer size."""
    gamma: float = 0.99
    """The discount factor gamma."""
    tau: float = 1.0
    """The target network update rate."""
    target_network_frequency: int = 500
    """The timesteps it takes to update the target network."""
    batch_size: int = 128
    """The batch size of sample from the reply memory."""
    start_e: float = 1
    """The starting epsilon for exploration."""
    end_e: float = 0.05
    """The ending epsilon for exploration."""
    exploration_fraction: float = 0.5
    """The fraction of `total-timesteps` it takes from start-e to go end-e."""
    learning_starts: int = 10
    """Timestep to start learning."""
    train_frequency: int = 1
    """The frequency of training."""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNPolicy(Policy):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.initialized = False
        super(DQNPolicy, self).__init__(config, *args, **kwargs)

    def get_action(self, belief: Belief, store: AliasStore) -> Tuple[Action, AliasStore]:
        ab = belief.abstract(store)
        if self.problem_spec.get_reward(ab, store) > 0:
            # Already in goal state, execute noop
            return Action("no-op"), store

        if not self.initialized:
            store = self.problem_spec.flat_stream_sample(
                ab, times=self.config["flat_width"], store=store
            )
            cuda = False
            self.num_envs = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
            self.args = DQNArgs()
            self.writer = SummaryWriter(f"runs/{str(time.time())}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s"
                % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
            )

            self.gym_env = GymEnv(belief, self.problem_spec, store)
            self.q_network = QNetwork(self.gym_env).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
            self.target_network = QNetwork(self.gym_env).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            self.rb = ReplayBuffer(
                self.args.buffer_size,
                self.gym_env.observation_space,
                self.gym_env.action_space,
                self.device,
                handle_timeout_termination=False,
            )
            self.initialized = True

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.gym_env.reset(seed=self.args.seed)
        for global_step in range(self.config["num_samples"]):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(
                self.args.start_e,
                self.args.end_e,
                self.args.exploration_fraction * self.args.total_timesteps,
                global_step,
            )
            if random.random() < epsilon:
                actions = np.array(
                    [self.gym_env.action_space.sample() for _ in range(self.num_envs)]
                )
            else:
                q_values = self.q_network(torch.Tensor(obs).to(self.device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.gym_env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        self.writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.args.learning_starts:
                if global_step % self.args.train_frequency == 0:
                    data = self.rb.sample(self.args.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + self.args.gamma * target_max * (
                            1 - data.dones.flatten()
                        )
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = torch.nn.functional.mse_loss(td_target, old_val)
                    print("DQN Loss: {}".format(loss.item()))
                    if global_step % 100 == 0:
                        self.writer.add_scalar("losses/td_loss", loss, global_step)
                        self.writer.add_scalar(
                            "losses/q_values", old_val.mean().item(), global_step
                        )

                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # update target network
                if global_step % self.args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(
                        self.target_network.parameters(), self.q_network.parameters()
                    ):
                        target_network_param.data.copy_(
                            self.args.tau * q_network_param.data
                            + (1.0 - self.args.tau) * target_network_param.data
                        )

        q_values = self.q_network(torch.Tensor([belief.vectorize()]).to(self.device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        action = self.gym_env.discrete_actions[actions[0]]

        if action in self.problem_spec.applicable_actions(ab, store):
            return action, {}, store
        else:
            return NoOp(), {}, store
