from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tampura.policies.policy import Policy
from tampura.solvers.mdp_solver import solve_mdp
from tampura.solvers.policy_search import normalize, policy_search
from tampura.spec import ProblemSpec
from tampura.structs import (
    AbstractBelief,
    AbstractBeliefSet,
    AbstractRewardModel,
    AbstractTransitionModel,
    Action,
    AliasStore,
    Belief,
)


def generate_symbolic_model(
    ab: AbstractBelief, problem_spec: ProblemSpec, store: AliasStore
) -> Tuple[AbstractTransitionModel, AbstractRewardModel]:
    """Symbolic model grounding -- very slow for large object sets and many
    actions/outcomes."""
    open_set = [ab]
    closed_set = []
    F = AbstractTransitionModel()
    R = AbstractRewardModel()
    r, _ = problem_spec.get_reward(ab, store)
    R.reward[ab] = r
    while len(open_set) > 0:
        ab = open_set.pop(0)
        if ab not in closed_set:
            r, _ = problem_spec.get_reward(ab, store)
            R.reward[ab] = r

            closed_set.append(ab)
            for a_p in problem_spec.applicable_actions(ab, store):
                if a_p not in F.effects:
                    F.effects[a_p] = {}

                if ab not in F.effects[a_p]:
                    F.effects[a_p][ab] = AbstractBeliefSet()

                for ab_p in problem_spec.get_successors(ab, a_p):
                    F.effects[a_p][ab].add(ab_p, None, 0)
                    if ab_p not in closed_set and ab_p not in closed_set:
                        open_set.append(ab_p)

    return F, R


class TampuraPolicy(Policy):
    def __init__(self, config: Dict[str, Any], *args, **kwargs) -> None:
        super(TampuraPolicy, self).__init__(config, *args, **kwargs)
        self.sampled = False
        self.envelope: List[Tuple[AbstractBelief, Action]] = []
        self.F = AbstractTransitionModel(root=None)
        self.R = AbstractRewardModel()
        self.belief_map: Dict[AbstractBelief, List[Belief]] = defaultdict(list)
        self.t = 0

    def get_action(
        self, belief: Belief, store: AliasStore
    ) -> Tuple[Action, Dict[str, Any], AliasStore]:
        ab = belief.abstract(store)

        self.F.root = ab
        self.R.reward[ab] = self.problem_spec.get_reward(ab, store)

        if self.R.reward[ab] > 0:
            # Already in goal state, execute noop
            self.t += 1
            return Action("no-op"), {"F": self.F}, store

        # This envelope mechanism speeds up planning by continuing a previous plan execution
        if len(self.envelope) > 0:
            envelope_prob = 1.0
            for eidx in range(len(self.envelope)):
                (previous_ab, a) = self.envelope[eidx]
                effect_ab = (self.envelope + [(ab, None)])[eidx + 1][0]
                if (
                    a in self.F.effects
                    and previous_ab in self.F.effects[a]
                    and effect_ab in self.F.effects[a][previous_ab].ab_probs
                ):
                    envelope_prob *= self.F.effects[a][previous_ab].ab_probs[effect_ab]
                else:
                    envelope_prob = 0.0
        else:
            envelope_prob = 0.0

        if envelope_prob < self.config["envelope_threshold"]:
            if self.config["from_scratch"]:
                self.F = AbstractTransitionModel(root=None)
                self.R = AbstractRewardModel()
                self.F.root = ab
                self.R.reward[ab] = self.problem_spec.get_reward(ab, store)
                self.belief_map = defaultdict(list)

            self.envelope = []

            if not self.sampled and len(self.problem_spec.stream_schemas) > 0:
                sample_width = 1
                if self.config["flat_sample"]:
                    sample_width = self.config["flat_width"]
                store = self.problem_spec.flat_stream_sample(ab, store=store, times=sample_width)
                self.sampled = True

            self.belief_map[ab].append(belief)
            n = 0

            while n < self.config["num_samples"]:
                # Set up dynamics and reward based on symbolic effects
                self.F, self.R, self.belief_map, plan_success = policy_search(
                    belief,
                    self.problem_spec,
                    self.F,
                    self.R,
                    self.belief_map,
                    store,
                    self.config,
                    save_dir=os.path.join(self.config["save_dir"], f"pddl_t={self.t}_s={n}"),
                )
                if not plan_success:
                    break

                if self.config["vis_graph"]:
                    save_file = os.path.join(
                        self.config["save_dir"], f"logs/transition_function_t={self.t}_s={n}.png"
                    )
                    self.F.visualize(self.R, save_file=save_file)

                n += self.config["batch_size"]

            logging.debug("Current abstract belief: " + str(ab))
            self.problem_spec.print_F(self.F)

        if len(self.F.effects) == 0:
            selected_action = Action("no-op")
        else:
            norm_F = normalize(self.F)
            sol = solve_mdp(
                norm_F,
                self.R,
                self.config["gamma"],
                decision_strategy=self.config["decision_strategy"],
                b0=ab,
            )
            applicable_actions = self.problem_spec.applicable_actions(ab, store)
            if sol.policy[ab] in applicable_actions:
                selected_action = sol.policy[ab]
            else:
                logging.warning(
                    f"WARNING: Taking a random action because {str(sol.policy[ab])} is not feasible."
                )
                selected_action = random.choice(applicable_actions)

        self.envelope.append((ab, selected_action))

        self.t += 1
        return selected_action, {"F": self.F}, store
