from __future__ import annotations

import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict

from tampura.spec import ProblemSpec
from tampura.structs import (
    AbstractRewardModel,
    AbstractTransitionModel,
    AliasStore,
    Belief,
)


class ActionNode:
    def __init__(self, action, parent=None):
        self.action = action
        self.visits = 0
        self.value = 0
        self.ab_children = []  # Child belief nodes for different outcomes of the action
        self.parent = parent  # The parent will be a BeliefNode

    def update_value(self):
        # Update the value as a weighted sum of child nodes
        self.value = sum(child.value * (child.visits / self.visits) for child in self.ab_children)


class BeliefNode:
    def __init__(self, abstract_belief, reward, parent=None):
        self.abstract_belief = abstract_belief
        self.reward = reward
        self.visits = 0
        self.value = 0
        self.action_children = []
        self.untried_actions = None
        self.parent = parent  # The parent will be an ActionNode

    def select_child(self):
        s = sum(child.visits for child in self.action_children)
        if s <= 0:
            return random.choice(self.action_children)

        def ucb_value(child, c=1.0):
            exploration = c * math.sqrt(2 * math.log(s) / (1 + child.visits))
            exploitation = child.value / (1 + child.visits)
            return exploitation + exploration

        return max(self.action_children, key=ucb_value)


def visualize_tree(node, graph=None):
    import pygraphviz as pgv

    if graph is None:
        graph = pgv.AGraph(directed=True, strict=True, rankdir="LR")

    if isinstance(node, ActionNode):
        graph.add_node(
            node,
            label=f"Value: {node.value:.2f}\nVisits: {node.visits}",
            shape="ellipse",
        )
        for child in node.ab_children:
            visualize_tree(child, graph)
            graph.add_edge(node, child, label=f"R: {child.reward:.2f}")  # or any suitable label
    else:  # BeliefNode
        graph.add_node(
            node,
            label=f"id: {hash(node)}\nValue: {node.value:.2f}\nVisits: {node.visits}",
            shape="box",
        )
        logging.info(f"{hash(node)} node info: " + str(node.abstract_belief))
        for child in node.action_children:
            visualize_tree(child, graph)
            graph.add_edge(node, child, label=child.action)

    return graph


def rollout(node):
    return node.reward


def uct_search(
    F: AbstractTransitionModel,
    R: AbstractRewardModel,
    b0: Belief,
    spec: ProblemSpec,
    store: AliasStore,
    config: Dict[str, float],
    t,
):
    a_b0 = b0.abstract(store)

    F.root = a_b0
    root = BeliefNode(a_b0, 0)
    root.untried_actions = spec.applicable_actions(root.abstract_belief, store)
    r = spec.get_reward(a_b0, store=store)

    a_b_map = defaultdict(list)
    a_b_map[a_b0].append(b0)

    R.reward[a_b0] = r
    num_samples = config["num_samples"]
    n = 0
    while n < num_samples:
        node = root
        expanded = False
        while True:
            if isinstance(node, BeliefNode):
                # For BeliefNode: either expand by choosing an untried action
                # or select one of the existing action children
                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    node.untried_actions.remove(action)
                    action_node = ActionNode(action, parent=node)
                    node.action_children.append(action_node)
                    node = action_node
                    expanded = True
                elif node.action_children:
                    node = node.select_child()
                else:
                    break
            else:  # For ActionNode
                # Sample an outcome based on action
                b = random.choice(a_b_map[node.parent.abstract_belief])
                print(f"Sampling: {n}/{num_samples}")

                new_abstract_belief_set = spec.get_action_schema(node.action.name).effects_fn(
                    node.action, b, store
                )
                F.update(node.action, node.parent.abstract_belief, new_abstract_belief_set)

                ab_ps, probs = zip(*new_abstract_belief_set.ab_probs.items())
                all_children = []
                for ab_p in ab_ps:
                    a_b_map[ab_p] += new_abstract_belief_set.belief_map[ab_p]
                    r = spec.get_reward(ab_p, store=store)
                    R.reward[ab_p] = r

                    matching_children = [
                        child for child in node.ab_children if child.abstract_belief == ab_p
                    ]
                    if matching_children:
                        child_node = matching_children[0]
                    else:
                        child_node = BeliefNode(ab_p, r, parent=node)
                        child_node.untried_actions = spec.applicable_actions(ab_p, store)
                        node.ab_children.append(child_node)

                    all_children.append(child_node)

                node = random.choices(all_children, probs, k=1)[0]
                n += 1
                if n >= num_samples or expanded:
                    break

        # Simulation
        delta = rollout(node)

        # Backpropagation
        while node:
            node.visits += 1
            node.value += delta
            if isinstance(node, ActionNode):
                node.update_value()
            node = node.parent
            delta *= config["gamma"]

    if config["vis"]:
        graph = visualize_tree(root)
        graph.layout(prog="dot")
        graph.draw(os.path.join(config["save_dir"], f"logs/uct_tree_t={t}.png"))

    return F, R
