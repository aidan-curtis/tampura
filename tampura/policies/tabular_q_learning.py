from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np

from tampura.policies.policy import Policy
from tampura.spec import ProblemSpec
from tampura.structs import (
    AbstractBelief,
    AbstractRewardModel,
    AbstractTransitionModel,
    Action,
    AliasStore,
    Belief,
)


def get_q_value(q_table, state, action):
    """Retrieves the Q-value for a given state-action pair from the sparse
    Q-table."""
    if (tuple(state), action) in q_table:
        return q_table[(tuple(state), action)]
    else:
        return 0  # Default Q-value if not yet visited


def set_q_value(q_table, state, action, value):
    """Sets the Q-value for a given state-action pair in the sparse Q-table."""
    q_table[(tuple(state), action)] = value


def random_argmax(value_list):
    """A random tie-breaking argmax."""
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


def choose_action(state, q_table, epsilon, actions) -> Action:
    """Epsilon-greedy action selection."""
    if random.uniform(0, 1) < epsilon:
        return actions[random.randint(0, len(actions) - 1)]  # Choose a random action
    else:
        # Use Q-values to choose the best action if known, else choose randomly
        action_values = [get_q_value(q_table, state, actions[i]) for i in range(len(actions))]
        return actions[int(random_argmax(action_values))]


def update_q_table(q_table, state, action, reward, next_state, environment, alpha, gamma, store):
    """Q-learning update rule using the sparse representation with dynamic
    action space."""
    current = get_q_value(q_table, state, action)
    valid_actions = environment.get_valid_actions(next_state, store)
    if valid_actions:
        future_values = [get_q_value(q_table, next_state, a) for a in valid_actions]
        max_future = max(future_values)
    else:
        max_future = 0  # If no valid actions, assume future value of zero

    new_value = current + alpha * (reward + gamma * max_future - current)
    set_q_value(q_table, state, action, new_value)


class QEnv:
    def __init__(self, initial_b: Belief, initial_ab: AbstractBelief, spec: ProblemSpec):
        self.initial_b_ab = (initial_b, initial_ab)
        self.current_b_ab = (initial_b, initial_ab)
        self.spec = spec

    def reset(self):
        self.current_b_ab = self.initial_b_ab
        return self.current_b_ab

    def step(self, action: Action, store: AliasStore):
        current_b, current_ab = self.current_b_ab
        new_abstract_belief_set = self.spec.get_action_schema(action.name).effects_fn(
            action, current_b, store
        )

        new_ab = new_abstract_belief_set.sample()
        new_b = random.choice(new_abstract_belief_set.belief_map[new_ab])

        self.current_b_ab = (new_b, new_ab)

        reward = self.spec.get_reward(new_ab, store=store)
        return self.current_b_ab, reward, reward > 0, store

    def get_valid_actions(self, state, store):
        b, ab = state
        return self.spec.applicable_actions(ab, store)


class TabularQLearning(Policy):
    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        super(TabularQLearning, self).__init__(config, *args, **kwargs)
        self.t = 0
        self.sampled = False

    def get_action(self, init_b: Belief, store: AliasStore) -> Tuple[Action, AliasStore]:
        init_ab = init_b.abstract(store)

        if not self.sampled and len(self.problem_spec.stream_schemas) > 0:
            store = self.problem_spec.flat_stream_sample(
                init_ab, times=self.config["flat_width"], store=store
            )
            self.sampled = True

        qenv = QEnv(init_b, init_ab, self.problem_spec)

        self.F = AbstractTransitionModel()
        self.R = AbstractRewardModel()

        q_table = {}
        exploration_eps = 0.2
        alpha = 0.2
        current_ab = init_ab
        current_b = init_b
        total_sims = 0
        while total_sims <= self.config["num_samples"]:
            print("Q Learning Episode " + str(total_sims))
            state = qenv.reset()
            done = False
            t = 0
            while not done:
                applicable = self.problem_spec.applicable_actions(current_ab, store)
                action = choose_action(state, q_table, exploration_eps, applicable)
                print("Testing action : " + str(action))
                (current_b, current_ab), reward, done, store = qenv.step(action, store)
                print("Reward: " + str(reward))
                update_q_table(
                    q_table,
                    state,
                    action,
                    reward,
                    (current_b, current_ab),
                    qenv,
                    alpha,
                    self.config["gamma"],
                    store,
                )
                t += 1
                total_sims += 1
                if t >= self.config["max_steps"]:
                    done = True

        return (
            choose_action(
                (init_b, init_ab), q_table, 0, self.problem_spec.applicable_actions(init_ab, store)
            ),
            {},
            store,
        )
