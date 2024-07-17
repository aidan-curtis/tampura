from __future__ import annotations

import math
import warnings

import mdptoolbox
import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_matrix

from tampura.solvers.lao_star import LAOStar
from tampura.structs import (
    AbstractRewardModel,
    AbstractSolution,
    AbstractTransitionModel,
)

warnings.simplefilter("ignore", SparseEfficiencyWarning)


def to_csr(F: AbstractTransitionModel, R: AbstractRewardModel):
    # Extract the number of states and actions from the AbstractTransitionModel and AbstractRewardModel
    S = len(R.reward) + 1  # +1 for the terminal state
    A = len(F.effects)  # Number of unique actions

    # Cache the list of keys
    reward_keys = list(R.reward.keys())

    # Convert the AbstractTransitionModel into transition matrices
    transitions = [csr_matrix((S, S)) for _ in range(A)]  # initialize empty matrices

    for action_idx, (action, beliefs_dict) in enumerate(F.effects.items()):
        for src_belief, dest_belief_set in beliefs_dict.items():
            if src_belief in reward_keys:
                src_state = reward_keys.index(src_belief)
            else:
                continue  # Skip this belief if it's not in our reward model

            for dest_belief, prob in dest_belief_set.ab_probs.items():
                if dest_belief in reward_keys:
                    dest_state = reward_keys.index(dest_belief)
                else:
                    continue  # Skip this belief if it's not in our reward model

                transitions[action_idx][src_state, dest_state] += prob

        # Ensure each state has outgoing transitions that sum to 1.
        for s in range(S - 1):  # Excluding terminal state
            if transitions[action_idx][s].sum() == 0:
                transitions[action_idx][s, S - 1] += 1.0  # Transition to terminal state

    # Terminal state always transitions to itself
    for action_idx in range(A):
        transitions[action_idx][S - 1, S - 1] += 1.0

    # Convert the AbstractRewardModel into reward vectors
    rewards = np.ones((S, A)) * 1e-5

    for s, ab in enumerate(reward_keys):
        for a in range(A):
            reward_val = R.reward.get(ab, 0)  # default reward is set to 0 if not defined
            if isinstance(reward_val, (list, tuple, np.ndarray)):
                raise ValueError(f"Reward for AbstractBelief {ab} is not a scalar: {reward_val}")
            rewards[s, a] += reward_val

    # Reward for the terminal state is always 0
    rewards[S - 1, :] = 0.0

    return transitions, rewards


def to_ml_outcome_csr(F: AbstractTransitionModel, R: AbstractRewardModel):
    S = len(R.reward) + 1  # +1 for the terminal state
    A = len(F.effects)  # Number of unique actions
    reward_keys = list(R.reward.keys())

    # Create a transition matrix for each action
    transitions = [csr_matrix((S, S)) for _ in range(A)]
    rewards = []

    for action_idx, (action, beliefs_dict) in enumerate(F.effects.items()):
        reward_vector = np.zeros((S, S))
        for src_belief, dest_belief_set in beliefs_dict.items():
            if src_belief not in reward_keys:
                continue

            src_state = reward_keys.index(src_belief)

            # Find the most likely outcome for this action
            most_likely_outcome = max(dest_belief_set.ab_probs.items(), key=lambda x: x[1])[0]

            if most_likely_outcome not in reward_keys:
                continue

            dest_state = reward_keys.index(most_likely_outcome)

            # Set the transition probability to 1 for the most likely outcome
            transitions[action_idx][src_state, dest_state] = 1.0

            reward_vector[src_state, dest_state] = R.reward.get(most_likely_outcome, 0)
            reward_vector[:, S - 1] = -1e-3

        rewards.append(reward_vector)

        # Transition to terminal state if no other transitions are defined for a state
        for s in range(S - 1):  # Excluding terminal state
            if transitions[action_idx][s].sum() == 0:
                transitions[action_idx][s, S - 1] += 1.0

        # Terminal state always transitions to itself
        transitions[action_idx][S - 1, S - 1] += 1.0

    rewards = np.array(rewards)
    return transitions, rewards


def to_all_outcome_csr(F: AbstractTransitionModel, R: AbstractRewardModel, cost_weighted=True):
    """To be used over all_outcomes for sparse representations."""
    S = len(R.reward) + 1  # +1 for the terminal state
    reward_keys = list(R.reward.keys())

    transitions = []
    rewards = []

    # This mapping will hold the relation between new MDP actions and original MDP actions.
    new_action_to_original = {}
    new_action_idx = 0  # Counter for the new MDP actions.

    for action, beliefs_dict in F.effects.items():
        reward_vector = np.zeros((S, S))
        for src_belief, dest_belief_set in beliefs_dict.items():
            if src_belief not in reward_keys:
                continue

            src_state = reward_keys.index(src_belief)
            for dest_belief, prob in dest_belief_set.ab_probs.items():
                if dest_belief not in reward_keys:
                    continue

                dest_state = reward_keys.index(dest_belief)

                # Initially set all transitions to the terminal state
                row_ind = list(range(S))  # all rows have a transition
                col_ind = [S - 1] * S  # all transitions go to the terminal state
                data = [1.0] * S  # transition probabilities are 1.0

                # Update the specific transition for src_state to dest_state
                row_ind[src_state] = src_state
                col_ind[src_state] = dest_state
                data[src_state] = 1.0

                # Create a transition matrix for this specific action-outcome pair
                trans_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(S, S))

                # Create a reward vector for this specific action-outcome pair
                if not cost_weighted:
                    penalty = 0
                else:
                    penalty = -math.log(prob) if prob > 0 else float("inf")

                reward_vector[src_state, dest_state] = 10 * R.reward.get(dest_belief, 0) - penalty

                transitions.append(trans_matrix)

                new_action_to_original[new_action_idx] = action
                new_action_idx += 1

                # Small negative penalty for terminal state otherwise VI throws errors
                reward_vector[:, S - 1] = -1e-3
                rewards.append(reward_vector)

    rewards = np.array(rewards)  # Transpose to match required shape
    return transitions, rewards, new_action_to_original


def solve_mdp(
    F: AbstractTransitionModel,
    R: AbstractRewardModel,
    gamma: float,
    decision_strategy: str = "prob",
    planner: str = "lao-star",
    b0=None,
) -> AbstractSolution:
    assert len(R.reward.keys()) > 0 and len(F.effects.keys()) > 0
    all_outcome = False
    if decision_strategy == "mlo":
        transitions, rewards = to_ml_outcome_csr(F, R)
    elif decision_strategy == "wao":
        all_outcome = True
        transitions, rewards, action_mapping = to_all_outcome_csr(F, R)
    elif decision_strategy == "ao":
        all_outcome = True
        transitions, rewards, action_mapping = to_all_outcome_csr(F, R, cost_weighted=False)
    elif decision_strategy == "prob":
        transitions, rewards = to_csr(F, R)
    else:
        raise NotImplementedError

    if planner == "vi":
        # Initialize and solve the MDP
        mdp_result = mdptoolbox.mdp.ValueIteration(
            transitions=transitions,
            reward=rewards,
            discount=gamma,
            epsilon=0.01,
            max_iter=1000,
        )

        mdp_result.run()
        policy = mdp_result.policy
        V = mdp_result.V

    elif planner == "lao-star":
        dense_transitions = np.concatenate(
            [np.expand_dims(t.todense(), axis=0) for t in transitions], axis=0
        )

        lao_star = LAOStar(
            transitions=dense_transitions,
            reward=rewards,
            discount=gamma,
            epsilon=0.01,
            max_iter=1000,
        )

        V, policy = lao_star.run(initial_state=list(R.reward.keys()).index(b0))
    else:
        raise NotImplementedError

    solution = AbstractSolution()
    actions = list(F.effects.keys())
    states = list(R.reward.keys())

    for ab, pi_b, v in zip(states, policy, V):
        if all_outcome:
            solution.policy[ab] = action_mapping[pi_b]
        else:
            solution.policy[ab] = actions[pi_b]

        solution.value[ab] = v

    return solution
