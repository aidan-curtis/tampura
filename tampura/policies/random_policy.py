from __future__ import annotations

import random
from typing import Tuple

from tampura.policies.policy import Policy
from tampura.structs import Action, AliasStore, Belief


class RandomPolicy(Policy):
    def __init__(self, config, *args, **kwargs):
        super(RandomPolicy, self).__init__(config, *args, **kwargs)
        self.sampled = False

    def get_action(self, belief: Belief, store: AliasStore) -> Tuple[Action, AliasStore]:
        a_b = belief.abstract(store)

        if not self.sampled:
            store = self.problem_spec.flat_stream_sample(
                a_b, times=self.config["flat_width"], store=store
            )
            self.sampled = True

        all_actions = self.problem_spec.applicable_actions(a_b, store)

        if len(all_actions) > 0:
            return random.choice(all_actions), {}, store
        else:
            return Action("no-op"), {}, store
