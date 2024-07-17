from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from tampura.spec import ProblemSpec
from tampura.structs import Action, AliasStore, Belief, Observation


class TampuraEnv:
    def __init__(self, config):
        self.config = config
        self.vis = config["vis"]
        self.save_dir = config["save_dir"]
        self.state = None

    @property
    @lru_cache
    def problem_spec(self):
        return self.get_problem_spec()

    def get_problem_spec(self) -> ProblemSpec:
        raise NotImplementedError

    def vis_updated_belief(self, belief: Belief, store: AliasStore):
        """Used to visualize or log the belief after being updated with the
        observation obtained from an environment step.

        This is implemented outside the belief update itself, as the
        recorded information may depend on aspects of the environment
        state as well.
        """
        return

    def initialize(self) -> Tuple[Belief, AliasStore]:
        """This function is called once at the beginning of the entire planning
        loop.

        It takes in an AliasStore and returns an observation with an
        updated alias store.
        """
        raise NotImplementedError

    def wrapup(self):
        """This function is called at the end of an execution to wrap up and
        export videos or gifs and close any open simulators."""
        pass

    def step(self, action: Action, belief: Belief, store: AliasStore) -> Observation:
        """This function is called after a planning step to execute a robot
        action either in simulation or on the robot.

        In addition to execution, this function is also responsible for
        performing a symbolic update on the current environment state to
        update the environment state with a new symbolic state.

        A reward is calculated based on this new symbolic state, and the
        action effect is returned along with the reward.
        """

        # Get the symbolic preconditions and effects of the action
        schema = self.problem_spec.get_action_schema(action.name)

        # Update the symbolic state
        self.state, observation = schema.execute_fn(action, belief, self.state, store)

        return observation
