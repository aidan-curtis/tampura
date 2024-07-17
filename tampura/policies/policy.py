from __future__ import annotations

import copy
import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import yaml

from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (
    AbstractBelief,
    Action,
    AliasStore,
    Belief,
    Observation,
    State,
)
from tampura.symbolic import Action

ALL_PRINT_OPS = "s,b,ab,r,a,o,sp,bp,abp,rp"


@dataclass
class RolloutHistory:
    def __init__(self, config):
        self.config = config
        self.states = []
        self.beliefs = []
        self.abstract_beliefs = []

        self.actions = []
        self.observations = []
        self.rewards = []
        self.time_deltas = []
        self.stores = []
        self.infos = []

    def __len__(self):
        return len(self.states)

    def add(
        self,
        s: State,
        b: Belief,
        a_b: AbstractBelief,
        a: Action,
        o: Observation,
        r: float,
        info: Dict,
        store: AliasStore,
        time_delta: float,
    ):
        self.states.append(s)
        self.beliefs.append(b)
        self.abstract_beliefs.append(a_b)
        self.actions.append(a)
        self.observations.append(o)
        self.rewards.append(r)
        self.time_deltas.append(time_delta)
        self.stores.append(copy.deepcopy(store))
        self.infos.append(info)

    def __str__(self):
        return str([str(a) for a in self.actions])


def save_run_data(history: RolloutHistory, save_dir: str):
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"{save_dir}/{formatted_datetime}.pkl", "wb") as f:
        pickle.dump(history, f)


def save_config(config: Dict[str, Any], save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "config.yml"), "w") as file:
        yaml.dump(config, file, default_flow_style=False)


class Policy:
    def __init__(self, config: Dict[str, Any], problem_spec: ProblemSpec, **kwargs):
        self.config = config
        self.problem_spec = problem_spec
        self.print_options = config["print_options"].split(",")

    def get_action(self, belief: Belief, store: AliasStore) -> Tuple[Action, Dict, AliasStore]:
        raise NotImplementedError

    def rollout(
        self, env: TampuraEnv, b: Belief, store: AliasStore
    ) -> Tuple[RolloutHistory, AliasStore]:
        assert env.problem_spec.verify(store)

        save_config(self.config, self.config["save_dir"])

        history = RolloutHistory(self.config)
        st = time.time()
        for step in range(self.config["max_steps"]):
            s = copy.deepcopy(env.state)
            a_b = b.abstract(store)
            reward = env.problem_spec.get_reward(a_b, store)

            logging.info("\n" + ("=" * 10) + "t=" + str(step) + ("=" * 10))
            if "s" in self.print_options:
                logging.info("State: " + str(s))
            if "b" in self.print_options:
                logging.info("Belief: " + str(b))
            if "ab" in self.print_options:
                logging.info("Abstract Belief: " + str(a_b))
            if "r" in self.print_options:
                logging.info("Reward: " + str(reward))

            action, info, store = self.get_action(b, store)

            if "a" in self.print_options:
                logging.info("Action: " + str(action))

            if action.name == "no-op":
                bp = copy.deepcopy(b)
                observation = None
            else:
                observation = env.step(action, b, store)
                bp = b.update(action, observation, store)

                if self.config["vis"]:
                    env.vis_updated_belief(bp, store)

            a_bp = bp.abstract(store)
            history.add(s, b, a_b, action, observation, reward, info, store, time.time() - st)

            reward = env.problem_spec.get_reward(a_bp, store)
            if "o" in self.print_options:
                logging.info("Observation: " + str(observation))
            if "sp" in self.print_options:
                logging.info("Next State: " + str(env.state))
            if "bp" in self.print_options:
                logging.info("Next Belief: " + str(bp))
            if "abp" in self.print_options:
                logging.info("Next Abstract Belief: " + str(a_bp))
            if "rp" in self.print_options:
                logging.info("Next Reward: " + str(reward))

            # update the belief
            b = bp

        history.add(env.state, bp, a_bp, None, None, reward, info, store, time.time() - st)

        logging.info("=" * 20)

        env.wrapup()

        if not self.config["real_execute"]:
            save_run_data(history, self.config["save_dir"])

        return history, store
