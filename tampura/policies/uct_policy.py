from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from tampura.policies.policy import Policy
from tampura.solvers.mdp_solver import solve_mdp
from tampura.solvers.sample_uct import uct_search
from tampura.structs import (
    AbstractRewardModel,
    AbstractTransitionModel,
    Action,
    AliasStore,
    Belief,
)


class UCTPolicy(Policy):
    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        super(UCTPolicy, self).__init__(config, *args, **kwargs)
        self.t = 0
        self.sampled = False

    def get_action(self, belief: Belief, store: AliasStore) -> Tuple[Action, AliasStore]:
        ab = belief.abstract(store)

        self.F = AbstractTransitionModel()
        self.R = AbstractRewardModel()

        a_b = belief.abstract(store)
        if not self.sampled and len(self.problem_spec.stream_schemas) > 0:
            store = self.problem_spec.flat_stream_sample(
                a_b, times=self.config["flat_width"], store=store
            )
            self.sampled = True

        self.F, self.R = uct_search(
            self.F, self.R, belief, self.problem_spec, store, self.config, self.t
        )
        solution = solve_mdp(self.F, self.R, self.config["gamma"])

        if self.config["vis_graph"]:
            save_file = os.path.join(
                self.config["save_dir"], f"logs/transition_function_t={self.t}.png"
            )
            self.F.visualize(self.R, save_file=save_file)

        self.t += 1
        return solution.policy[ab], {}, store
