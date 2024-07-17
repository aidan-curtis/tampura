from __future__ import annotations

from typing import Tuple

from tampura.policies.policy import Policy
from tampura.structs import Action, AliasStore, Belief


class ReplayPolicy(Policy):
    def __init__(self, *args, execution_data=None, **kwargs):
        super(ReplayPolicy, self).__init__(*args, **kwargs)
        self.execution_data = execution_data
        self.t = 0

    def get_action(self, belief: Belief, store: AliasStore) -> Tuple[Action, AliasStore]:
        action = self.execution_data.actions[self.t]
        self.t += 1
        return action, {}, store
