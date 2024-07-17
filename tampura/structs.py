from __future__ import annotations

import copy
import itertools
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from tampura.store import AliasStore
from tampura.symbolic import (
    Action,
    And,
    Atom,
    Exists,
    Expr,
    ForAll,
    Not,
    OneOf,
    Predicate,
    When,
    eval_expr,
    get_args,
    get_pred_names,
    negate,
    substitute,
)


@dataclass
class AbstractSolution:
    policy: Dict[AbstractBelief, Action] = field(default_factory=lambda: {})
    value: Dict[AbstractBelief, float] = field(default_factory=lambda: {})
    Q: Dict[AbstractBelief, Dict[Action, float]] = field(default_factory=lambda: {})


@dataclass
class AbstractTransitionModel:
    root: AbstractBelief = None
    effects: Dict[Action, Dict[AbstractBelief, AbstractBeliefSet]] = field(
        default_factory=lambda: {}
    )

    def total_count(self):
        return sum(
            itertools.chain(
                *[[ab_set.total_count() for ab_set in e.values()] for e in self.effects.values()]
            )
        )

    def update(self, a: Action, ab: AbstractBelief, ab_p: AbstractBeliefSet):
        if a not in self.effects:
            self.effects[a] = {}

        if ab not in self.effects[a]:
            self.effects[a][ab] = AbstractBeliefSet()

        self.effects[a][ab].merge(ab_p)

    def get_transition(self, a: Action, ab: AbstractBelief) -> AbstractBeliefSet:
        if a not in self.effects.keys() or ab not in self.effects[a]:
            return AbstractBeliefSet()
        else:
            return self.effects[a][ab]

    def visualize(self, R: AbstractRewardModel, save_file="transition_function.png"):
        import pygraphviz as pgv

        graph = pgv.AGraph(directed=True, strict=True, rankdir="LR")

        def belief_str(ab):
            return "ABelief {}".format(str(hash(ab))[:5])

        graph.add_node(hash(self.root), label=belief_str(self.root), shape="box")
        belief_str_map = {}
        for action, belief_dict in self.effects.items():
            for belief, belief_set in belief_dict.items():
                graph.add_node(hash(action) + hash(belief), label=str(action), shape="ellipse")

            for belief, belief_set in belief_dict.items():
                belief_str_map[belief_str(belief)] = str(belief)
                graph.add_node(hash(belief), label=belief_str(belief), shape="box")
                graph.add_edge(hash(belief), hash(action) + hash(belief), color="blue")

                for next_belief, count in belief_set.ab_counts.items():
                    belief_str_map[belief_str(next_belief)] = str(next_belief)
                    graph.add_node(hash(next_belief), label=belief_str(next_belief), shape="box")
                    graph.add_edge(
                        hash(action) + hash(belief), hash(next_belief), label=str(count)
                    )

        for k, v in belief_str_map.items():
            logging.debug("{} : {}".format(k, v))

        for ab, r in R.reward.items():
            if r > 0:
                node = graph.get_node(hash(ab))
                node.attr["style"] = "filled"
                node.attr["color"] = "black"
                node.attr["fillcolor"] = "#98FB98"  # pastel green

        if self.root is not None:
            node = graph.get_node(hash(self.root))
            node.attr["style"] = "filled"
            node.attr["color"] = "black"
            node.attr["fillcolor"] = "#AEC6CF"  # pastel blue

        graph.layout(prog="dot")
        graph.draw(save_file)

        return graph


@dataclass
class AbstractRewardModel:
    reward: Dict[AbstractBelief, float] = field(default_factory=lambda: {})


@dataclass
class Observation:
    pass


def default_effect_fn(
    action: Action, belief: Belief, store: AliasStore
) -> Tuple[AbstractBeliefSet, AliasStore]:
    new_belief = belief.update(action, None, store)
    return AbstractBeliefSet.from_beliefs([new_belief], store)


def effect_from_execute_fn(
    execute_fn: Callable[[Action, Belief, State, AliasStore], Tuple[State, Observation]]
):
    """For convenience, we can simulate the effects of a controller by calling
    that controller's execute function with no state.

    The effects function will need to be able to handle None state as
    input
    """

    def effect_fn(
        action: Action, belief: Belief, store: AliasStore
    ) -> Tuple[AbstractBeliefSet, AliasStore]:
        _, obs = execute_fn(action, belief, None, store)
        new_belief = belief.update(action, obs, store)
        return AbstractBeliefSet.from_beliefs([new_belief], store)

    return effect_fn


@dataclass
class ActionSchema:
    name: str = "default"
    inputs: List[str] = field(default_factory=lambda: [])
    input_types: List[str] = field(default_factory=lambda: [])
    preconditions: List[Any] = field(default_factory=lambda: [])
    effects: List[Any] = field(default_factory=lambda: [])
    verify_effects: List[Expr] = field(default_factory=lambda: [])
    depends: List[Expr] = field(default_factory=lambda: [])
    effects_fn: Callable[[Action, Belief, AliasStore], AbstractBeliefSet] = default_effect_fn
    execute_fn: Callable[[Action, Belief, State, AliasStore], Tuple[State, Observation]] = None


@dataclass
class StreamSchema:
    name: str = "default"
    inputs: List[str] = field(default_factory=lambda: [])
    input_types: List[str] = field(default_factory=lambda: [])
    output: List[str] = field(default_factory=lambda: [])
    output_type: List[str] = field(default_factory=lambda: [])
    preconditions: List[Any] = field(default_factory=lambda: [])
    certified: List[Any] = field(default_factory=lambda: [])
    sample_fn: Callable[[List[str], AliasStore], Tuple[List[str], AliasStore]] = None


@dataclass
class State:
    pass


@dataclass
class AbstractBeliefSet:
    ab_counts: Dict[AbstractBelief, int] = field(default_factory=lambda: {})

    # belief_map for bookkeeping the set of beliefs that correspond to each abstract belief
    belief_map: Dict[AbstractBelief, List[Belief]] = field(default_factory=lambda: {})

    outcome_successes: Dict[AbstractBelief, int] = field(default_factory=lambda: {})
    outcome_attempts: Dict[AbstractBelief, int] = field(default_factory=lambda: {})

    def get_all_counts(self, ab: AbstractBelief):
        return (
            self.get_count(self.ab_counts, ab),
            self.get_count(self.outcome_successes, ab),
            self.get_count(self.outcome_attempts, ab),
        )

    def all_ab(self):
        return set(
            list(self.ab_counts.keys())
            + list(self.outcome_successes.keys())
            + list(self.outcome_attempts.keys())
        )

    def total_count(self):
        return sum(list(self.ab_counts.values()))

    def get_count(self, count_dict: Dict[AbstractBelief, int], ab: AbstractBelief):
        if ab not in count_dict:
            return 0
        else:
            return count_dict[ab]

    def merge(self, belief_set: AbstractBeliefSet):
        for ab, count in belief_set.ab_counts.items():
            if not (ab in self.ab_counts):
                self.ab_counts[ab] = 0
                self.belief_map[ab] = []
            self.belief_map[ab] += belief_set.belief_map[ab]
            self.ab_counts[ab] += count

        for ab, count in belief_set.outcome_attempts.items():
            if ab not in self.outcome_attempts:
                self.outcome_attempts[ab] = 0
            self.outcome_attempts[ab] += belief_set.outcome_attempts[ab]

        for ab, count in belief_set.outcome_successes.items():
            if ab not in self.outcome_successes:
                self.outcome_successes[ab] = 0
            self.outcome_successes[ab] += belief_set.outcome_successes[ab]

    def add(self, ab: AbstractBelief, b: Belief, count: int):
        if not (ab in self.ab_counts):
            self.ab_counts[ab] = 0
            self.belief_map[ab] = []
        self.ab_counts[ab] += count
        self.belief_map[ab].append(b)

    def sample(self) -> AbstractBelief:
        beliefs, probs = zip(*self.ab_probs.items())
        return random.choices(beliefs, weights=probs)[0]

    @staticmethod
    def from_beliefs(beliefs: List[Belief], store) -> AbstractBeliefSet:
        ab_counts = defaultdict(lambda: 0)
        belief_map = defaultdict(lambda: [])
        for belief in beliefs:
            a_belief = belief.abstract(store)
            ab_counts[a_belief] += 1
            belief_map[a_belief].append(belief)
        return AbstractBeliefSet(ab_counts=ab_counts, belief_map=belief_map)

    @property
    def ab_probs(self):
        total = sum(self.ab_counts.values())
        return {k: v / float(total) for k, v in self.ab_counts.items()}


def all_outcome_combos(schema: ActionSchema):
    choices = []
    for ve in schema.verify_effects:
        if isinstance(ve, Atom) or isinstance(ve, Not):
            choices.append([0, 1])
        elif isinstance(ve, OneOf):
            choices.append(list(range(len(ve.components))))
        else:
            logging.error(f"Verified effect type {type(ve)} not supported")
            raise NotImplementedError
    return choices


def outcome_combo_to_expr(schema: ActionSchema, outcome_combo: List[int]) -> List[Expr]:
    veff = []
    for i, ve in enumerate(schema.verify_effects):
        if isinstance(ve, Atom) or isinstance(ve, Not):
            if outcome_combo[i]:
                veff.append(ve)
            else:
                veff.append(negate(ve))
        elif isinstance(ve, OneOf):
            all_negated = [negate(e) for e in ve.components]
            veff.append(
                And(
                    all_negated[: outcome_combo[i]]
                    + [ve.components[outcome_combo[i]]]
                    + all_negated[outcome_combo[i] + 1 :]
                )
            )
    return veff


def get_object_combos(
    preconditions: List[Expr],
    args: List[str],
    arg_types: List[str],
    facts: List[Expr],
    store: AliasStore,
    arg_map: Dict[str, str] = {},
) -> List[Dict[str, str]]:
    Q = [arg_map]
    arg_to_type = {k: v for k, v in zip(args, arg_types)}

    sorted_preconditions = sorted(preconditions, key=lambda x: len(get_args(x)))

    for precond in sorted_preconditions:
        precond_args = get_args(precond)
        new_Q = []
        for q in Q:
            local_type_dict = {}
            for arg in precond_args:
                if arg in q:
                    local_type_dict[arg] = [q[arg]]
                else:
                    local_type_dict[arg] = store.type_dict[arg_to_type[arg]]
            for object_combo in itertools.product(*[local_type_dict[arg] for arg in precond_args]):
                arg_map = {k: v for k, v in zip(precond_args, object_combo)} | q
                if eval_expr(precond, arg_map, facts, store.type_dict):
                    new_Q.append(arg_map)
        Q = copy.deepcopy(new_Q)

    if len(Q) > 0:
        # Add in args that are in no preconditions
        for arg in args:
            if arg not in Q[0]:
                new_Q = []
                for q in Q:
                    for obj in store.type_dict[arg_to_type[arg]]:
                        new_Q.append(q | {arg: obj})
                Q = copy.deepcopy(new_Q)
    return Q


def noop_effect_fn(
    action: Action, belief: Belief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:
    return AbstractBeliefSet.from_beliefs([belief], store)


def noop_execute_fn(
    action: Action,
    belief: Belief,
    state: State,
    store: AliasStore,
    **kwargs,
) -> Tuple[State, Observation]:
    return state, None


# TODO: This could be implemented with generics
@dataclass
class NoOp(ActionSchema):
    name: str = "no-op"
    inputs: List[str] = field(default_factory=lambda: [])
    input_types: List[str] = field(default_factory=lambda: [])
    preconditions: List[Any] = field(default_factory=lambda: [])
    effects: List[Any] = field(default_factory=lambda: [])
    verify_effects: List[Expr] = field(default_factory=lambda: [])

    effects_fn: Callable[
        [List[str], Belief, AliasStore], Tuple[AbstractBeliefSet, AliasStore]
    ] = noop_effect_fn
    execute_fn: Callable[
        [List[str], Belief, State, AliasStore], Tuple[Observation, AliasStore]
    ] = noop_execute_fn


@dataclass
class Stream:
    """A ground stream schema.

    Similar to an Action as an instantiation of a ActionSchema
    """

    name: str
    inputs: List[str]
    output: str


def symbolic_eff(b: AbstractBelief, effect: AbstractEffect, store: AliasStore) -> AbstractBelief:
    new_atoms = copy.deepcopy(b.items)
    for ground_effect in effect.items:
        if isinstance(ground_effect, Not):
            if isinstance(ground_effect.component, Atom):
                if ground_effect.component in b.items:
                    new_atoms = [atom for atom in new_atoms if atom != ground_effect.component]
            else:
                raise NotImplementedError
        elif isinstance(ground_effect, Atom):
            if ground_effect not in new_atoms:
                new_atoms.append(ground_effect)
        elif isinstance(ground_effect, ForAll):
            object_sets = list(
                itertools.product(
                    *[store.type_dict[input_type] for input_type in ground_effect.types]
                )
            )
            for object_set in object_sets:
                input_dict = {k: v for k, v in zip(ground_effect.args, object_set)}
                component = ground_effect.component
                if isinstance(component, When):
                    if eval_expr(
                        component.a, arg_map=input_dict, facts=b.items, type_dict=store.type_dict
                    ):
                        new_atoms = symbolic_eff(
                            AbstractBelief(new_atoms),
                            effect=AbstractEffect([substitute(component.b, input_dict)]),
                            store=store,
                        ).items
                else:
                    new_atoms = symbolic_eff(
                        AbstractBelief(new_atoms),
                        effect=AbstractEffect([substitute(component, input_dict)]),
                        store=store,
                    ).items

        else:
            raise NotImplementedError

    return AbstractBelief(new_atoms)


def symbolic_update(
    b: AbstractBelief, a: Action, schema: ActionSchema, effect: AbstractEffect, store: AliasStore
) -> AbstractBelief:
    """This function is used to update the state atoms after an action
    effect."""
    input_dict = {k: v for k, v in zip(schema.inputs, a.args)}
    all_effects = [substitute(unit, input_dict) for unit in effect.items + schema.effects]
    return symbolic_eff(b, AbstractEffect(all_effects), store)


class Belief:
    def update(
        self,
        action: Action,
        observation: Observation,
        store: AliasStore,
    ) -> Belief:
        raise NotImplementedError

    def abstract(self, store: AliasStore) -> AbstractBelief:
        raise NotImplementedError


@dataclass
class AbstractBelief:
    items: List[Expr] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.items = sorted(list(set(self.items)))

    def __hash__(self):
        return hash(tuple([hash(item) for item in sorted(self.items)]))

    def __eq__(self, ab):
        return hash(self) == hash(ab)

    def fluents(self, fluents: List[Predicate]):
        fluent_names = [f.name for f in fluents]
        fluent_items = []
        for item in self.items:
            if isinstance(item, Atom):
                if item.pred_name in fluent_names:
                    fluent_items.append(item)
            elif isinstance(item, Not):
                if item.component.pred_name in fluent_names:
                    fluent_items.append(item)
        return fluent_items

    def closed_fluents(
        self, fluents: List[Predicate], depends: List[Expr], input_map: Action, store: AliasStore
    ):
        all_negated = []
        for fluent in fluents:
            if any([fluent.name in get_pred_names(depend) for depend in depends]):
                object_sets = list(
                    itertools.product(
                        *[store.type_dict[input_type] for input_type in fluent.arg_types]
                    )
                )

                for object_set in object_sets:
                    a = Atom(fluent.name, list(object_set))
                    does_depend = False
                    for depend in depends:
                        if isinstance(depend, Exists) or isinstance(depend, ForAll):
                            assert len(depend.args) == len(fluent.arg_types)
                            arg_map = {k: v for k, v in zip(depend.args, object_set)}
                            does_depend |= eval_expr(
                                depend.component,
                                arg_map | input_map,
                                [a] + store.certified,
                                store.type_dict,
                            )
                        else:
                            does_depend |= eval_expr(
                                depend, input_map, [a] + store.certified, store.type_dict
                            )
                    if does_depend:
                        if a not in self.items:
                            all_negated.append(Not(a))
                        else:
                            all_negated.append(a)

        return AbstractBelief(items=sorted(all_negated))


@dataclass
class AbstractRollout:
    transitions: List[Tuple[Action, AbstractBelief, AbstractBelief]]


@dataclass
class AbstractEffect(AbstractBelief):
    pass
