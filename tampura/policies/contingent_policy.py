from __future__ import annotations

import os
import random
from typing import Tuple

from tampura.policies.policy import Policy
from tampura.solvers.symk import symk_search, symk_translate
from tampura.structs import Action, AliasStore, Belief
from tampura.symbolic import ACTION_EXT


class ContingentPolicy(Policy):
    def __init__(self, config, *args, **kwargs):
        super(ContingentPolicy, self).__init__(config, *args, **kwargs)
        self.sampled = False
        self.t = 0

    def get_action(self, belief: Belief, store: AliasStore) -> Tuple[Action, AliasStore]:
        a_b = belief.abstract(store)

        if not self.sampled:
            store = self.problem_spec.flat_stream_sample(
                a_b, times=self.config["flat_width"], store=store
            )
            self.sampled = True

        a_b0 = belief.abstract(store)

        default_cost = 0
        save_dir = os.path.join(self.config["save_dir"], f"pddl_t={self.t}")

        (domain_file, problem_file) = self.problem_spec.save_pddl(
            a_b0, default_cost=default_cost, folder=save_dir, store=store
        )
        output_sas_file = symk_translate(domain_file, problem_file)
        plans = symk_search(output_sas_file, self.config)
        self.t += 1

        if len(plans) == 0 or plans is None or len(plans[0]) == 0:
            all_actions = self.problem_spec.applicable_actions(a_b, store)
            print("All applicable actions: " + str(all_actions))
            return random.choice(all_actions), {}, store
        else:
            plan_action = plans[0][0]
            action_name, _ = plan_action.name.split(ACTION_EXT)
            action = Action(action_name, plan_action.args)
            return action, {}, store
