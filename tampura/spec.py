from __future__ import annotations

import copy
import itertools
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import beta

from tampura.store import AliasStore
from tampura.structs import (
    AbstractBelief,
    AbstractBeliefSet,
    AbstractEffect,
    AbstractTransitionModel,
    ActionSchema,
    Belief,
    NoOp,
    Observation,
    State,
    StreamSchema,
    all_outcome_combos,
    get_object_combos,
    outcome_combo_to_expr,
    symbolic_eff,
    symbolic_update,
)
from tampura.symbolic import (
    ACTION_EXT,
    VEFFECT_SEPARATOR,
    Action,
    And,
    Atom,
    Eq,
    Exists,
    Expr,
    ForAll,
    Imply,
    IncreaseCost,
    Not,
    OneOf,
    Or,
    Predicate,
    When,
    eval_expr,
    get_pred_names,
    negate,
    substitute,
)

MAX_COST = 1000
BETA_PRIOR = (1, 1)


def beta_quantile(A, B, N):
    # N+2 to give reasonable results when N=0
    # return beta.ppf(1 - (1 / (N + 2)), BETA_PRIOR[0] + A, BETA_PRIOR[1] + B)
    return beta.ppf(0.95, BETA_PRIOR[0] + A, BETA_PRIOR[1] + B)


@dataclass
class ProblemSpec:
    predicates: List[Predicate] = field(default_factory=lambda: {})
    action_schemas: List[ActionSchema] = field(default_factory=lambda: [])
    stream_schemas: List[StreamSchema] = field(default_factory=lambda: [])
    reward: Any = None

    @property
    def types(self):
        return list(
            set(
                list(itertools.chain(*[p.arg_types for p in self.predicates]))
                + list(itertools.chain(*[p.input_types for p in self.action_schemas]))
                + list(
                    itertools.chain(
                        *[p.input_types + [p.output_type] for p in self.stream_schemas]
                    )
                )
            )
        )

    @property
    def continuous_types(self):
        return [s.output_type for s in self.stream_schemas]

    def print_F(self, F: AbstractTransitionModel):
        out = ""
        for action in F.effects.keys():
            out += "-----------\n"
            out += str(action) + "\n"
            for b, bs in F.effects[action].items():
                out += "\t" + str(b.fluents(self.fluent_predicates())) + "\n"
                for bp, prob in bs.ab_probs.items():
                    out += "\t\t{}: {}\n".format(prob, bp.fluents(self.fluent_predicates()))
        logging.debug(out)

    def flat_stream_sample(
        self, ab: AbstractBelief, store: AliasStore, times: int = 1
    ) -> AliasStore:
        logging.info("Flat Stream Sampling")
        if len(self.stream_schemas) == 0:
            return store

        max_depth = max([self.get_stream_depth(schema) for schema in self.stream_schemas])

        for _ in range(times):
            stream_sampled_inputs = defaultdict(list)
            for _ in range(max_depth):
                for sampler in sorted(self.stream_schemas, key=lambda s: self.get_stream_depth(s)):
                    Q = get_object_combos(
                        sampler.preconditions,
                        sampler.inputs,
                        sampler.input_types,
                        ab.items + store.certified,
                        store,
                    )
                    for q in Q:
                        input_sym = [q[arg] for arg in sampler.inputs]
                        # Verify that input symbols are all unique
                        if len(set(input_sym)) != len(input_sym):
                            continue
                        if input_sym not in stream_sampled_inputs[sampler.name]:
                            stream_sampled_inputs[sampler.name].append(input_sym)
                            if None not in store.get_all(input_sym):
                                logging.info(f"Sampling {sampler}({input_sym})")
                                output = sampler.sample_fn(input_sym, store=store)
                                output_object = store.add_typed(output, sampler.output_type)
                                merged_dict = {k: v for k, v in zip(sampler.inputs, input_sym)} | {
                                    sampler.output: output_object
                                }
                                for fact in sampler.certified:
                                    args = [merged_dict[q] for q in fact.args]
                                    store.certified.append(Atom(fact.pred_name, args))
        return store

    def get_reward(self, abstract_belief: AbstractBelief, store: AliasStore) -> float:
        facts = abstract_belief.items + store.certified
        reward = eval_expr(self.reward, {}, facts, store.type_dict)
        return float(reward)

    def get_successors(
        self, abstract_belief: AbstractBelief, action: Action, store: AliasStore
    ) -> List[AbstractBelief]:
        schema = self.get_action_schema(action.name)
        successors = []
        for delta in itertools.product(*[[e, negate(e)] for e in schema.verify_effects]):
            successors.append(
                symbolic_update(
                    abstract_belief,
                    action,
                    schema,
                    AbstractEffect(list(delta) + schema.effects),
                    store,
                )
            )
        return successors

    def get_stream_depth(self, schema: StreamSchema):
        """Returns the maximum number of streams that must be sampled to
        satisfy the certified predicates of stream_name."""
        deps = []
        for pre in schema.preconditions:
            for possible_dep in self.stream_schemas:
                if pre.pred_name in [d.pred_name for d in possible_dep.certified]:
                    deps.append(possible_dep)

        if len(deps) == 0:
            return 1
        else:
            return 1 + max([self.get_stream_depth(dep) for dep in deps])

    def applicable_actions(
        self, abstract_belief: AbstractBelief, store: AliasStore
    ) -> List[Action]:
        combined_items = abstract_belief.items + store.certified

        # First, index all the objects by their type.
        actions = []
        for a in self.action_schemas:
            Q = get_object_combos(a.preconditions, a.inputs, a.input_types, combined_items, store)
            for arg_map in Q:
                object_combo = [arg_map[arg] for arg in a.inputs]
                assert self.check_preconditions(a, object_combo, store=store, facts=combined_items)
                if all([o is not None for o in store.get_all(object_combo)]):
                    actions.append(Action(a.name, object_combo))

        return actions

    def verify_atom(self, atom, current_scope) -> bool:
        # This function is essentially the same as your original verify_atom
        # but with the slight modification to use current_scope instead of combined_map
        predicate = self.get_predicate(atom.pred_name)
        if predicate is None:
            logging.info(f"Used predicate {atom.pred_name} not in predicate set")
            return False
        if len(predicate.arg_types) != len(atom.args):
            logging.info(f"Predicate {predicate} does not match atom {atom}")
            return False
        for i, arg in enumerate(atom.args):
            if predicate.arg_types[i] != current_scope.get(arg):
                logging.info(f"Predicate {predicate} does not match atom {atom}.")
                return False
        return True

    def verify_expr(self, expr, current_scope={}) -> bool:
        if isinstance(expr, Atom):
            if not isinstance(expr.args, list):
                return False
            return self.verify_atom(expr, current_scope)
        elif isinstance(expr, Not):
            return self.verify_expr(expr.component, current_scope)
        elif isinstance(expr, And) or isinstance(expr, Or) or isinstance(expr, OneOf):
            if not isinstance(expr.components, list):
                return False
            return all(self.verify_expr(c, current_scope) for c in expr.components)
        elif isinstance(expr, Exists) or isinstance(expr, ForAll):
            # Extend the current scope with the new arguments introduced by the quantifier
            if not isinstance(expr.args, list) or not isinstance(expr.types, list):
                return False
            extended_scope = {**current_scope, **dict(zip(expr.args, expr.types))}
            return self.verify_expr(expr.component, extended_scope)
        elif isinstance(expr, Imply) or isinstance(expr, When) or isinstance(expr, Eq):
            return self.verify_expr(expr.a, current_scope) and self.verify_expr(
                expr.b, current_scope
            )
        elif isinstance(expr, str):
            return True
        else:
            raise NotImplementedError

    def verify(self, store: AliasStore) -> bool:
        for predicate in self.predicates:
            if not isinstance(predicate.arg_types, list):
                return False

        for schema in self.action_schemas + self.stream_schemas:
            if len(schema.inputs) != len(schema.input_types):
                logging.info(
                    "Input length doesn't match input type length in schema {}".format(schema.name)
                )
                return False

            input_map = {
                arg: t for arg, t in zip(schema.inputs, schema.input_types)
            } | store.als_type
            if isinstance(schema, ActionSchema):
                for depends in schema.depends:
                    if not self.verify_expr(depends, input_map):
                        logging.info(f"Failed to verify depends expression {str(depends)}")
                        return False

                conditions = schema.preconditions + schema.verify_effects + schema.effects
            else:
                input_map = input_map | {schema.output: schema.output_type}
                conditions = schema.preconditions + schema.certified

            for expr in conditions:
                if not self.verify_expr(expr, input_map):
                    logging.info(f"Failed to verify expression {expr} in schema {schema.name}.")
                    return False

        if not self.verify_expr(self.reward, store.als_type):
            logging.info(f"Failed to verify goal.")
            return False

        return True

    def project(self, belief, store, num_init=1) -> Tuple[AbstractBeliefSet, AliasStore]:
        projection = AbstractBeliefSet()
        for _ in range(num_init):
            sampled_atoms = belief.sample_known(store)
            projection.add(sampled_atoms)

        projection.compress()
        return projection, store

    def get_value(self, expr, pred_name) -> str:
        """
        This function is used to extract the new value of a predicate from an effect expression
        Input: expr = Eq(5, Atom("height", ["?o"])), pred_name = "height"
        Output: 5
        """
        if isinstance(expr, Atom) and pred_name == expr.pred_name:
            return "true"
        elif (
            isinstance(expr, Not)
            and isinstance(expr.component, Atom)
            and pred_name == expr.component.pred_name
        ):
            return "false"
        else:
            print(f"{type(expr)} not supported in get_value for predicate name {pred_name}")
            raise NotImplementedError

    def check_preconditions(
        self,
        sampler: Any,
        input_tuple: List[str],
        store: AliasStore,
        facts: List[Expr],
    ) -> bool:
        """This function is used to check if the preconditions of a sampler are
        satisfied."""
        arg_map = {param: val for param, val in zip(sampler.inputs, input_tuple)}
        for precondition in sampler.preconditions:
            if not eval_expr(precondition, arg_map, facts, store.type_dict):
                return False
        return True

    def ground_state(self, state, store) -> State:
        """Given a set of predicates and a state object, ground each predicate
        with different object combinations from the store within the state
        object."""
        atoms = []

        # Keep atoms with no predicate grounding function
        for a in state.atoms:
            if self.get_predicate(self.get_expr_atoms(a)[0].pred_name).grounding is None:
                atoms.append(a)

        # Check object combinations for predicates with grounding functions
        for predicate in self.predicates:
            if predicate.grounding is not None:
                objects = [store.type_dict[t] for t in predicate.arg_types]
                for sym_args in itertools.product(*objects):
                    if predicate.grounding(*sym_args, store=store, state=state):
                        atoms.append(Atom(predicate.name, sym_args))

        state.atoms = atoms
        return state

    def get_action_schema(self, schema_name: str) -> ActionSchema:
        """Given an action name, return the action object."""
        return {a.name: a for a in self.action_schemas}.get(schema_name, None)

    def get_stream_schema(self, schema_name: str) -> StreamSchema:
        """Given an action name, return the action object."""
        return {s.name: s for s in self.stream_schemas}.get(schema_name, None)

    def get_predicate(self, pred_name: str) -> Predicate:
        """Given a predicate name, return the predicate object."""
        return {a.name: a for a in self.predicates}.get(pred_name, None)

    def fluent_predicates(self):
        all_predicates = []
        for schema in self.action_schemas:
            for expr in schema.effects + schema.verify_effects:
                all_predicates += [self.get_predicate(pn) for pn in get_pred_names(expr)]
        return sorted(list(set(all_predicates)))

    def goal_pddl(self):
        goal_lines = []
        goal_pddl = Atom("goal").pddl()
        goal_lines.append(f"  (:derived {goal_pddl}")
        conditions_str = " (and " + str(self.reward.pddl()) + ")"
        goal_lines.append(f"    {conditions_str}")
        goal_lines.append("  )")  # Close Axiom
        return "\n".join(goal_lines)

    def header_pddl(self):
        header_lines = []
        # Domain Header
        header_lines.append("  (:types {} - object)".format(" ".join(self.types)))
        # Predicates (including unary types)
        header_lines.append("  (:predicates")
        for predicate in self.predicates:
            header_lines.append(f"    {predicate.pddl()}")
        header_lines.append(f"    (eq ?o1 - object ?o2 - object)")
        header_lines.append(f"    (goal )")
        header_lines.append("  )")  # Close Predicates
        return "\n".join(header_lines)

    def action_ppddl(self, cost_modifiers: List[CostModifier], store: AliasStore):
        def get_prob_action(name, inputs, input_types, preconditions, effect_probs):
            action_lines = []
            action_lines.append(f"  (:action {name}")
            # Parameters
            params = " ".join([f"{arg} - {t}" for arg, t in zip(inputs, input_types)])
            action_lines.append(f"    :parameters ({params})")

            preconditions = copy.deepcopy(preconditions[:])

            preconditions_str = (
                " (and " + " ".join([precon.pddl() for precon in preconditions]) + ")"
            )
            action_lines.append(f"    :precondition{preconditions_str}")

            action_lines.append(f"    :effect ")
            action_lines.append(f"    (probabilistic ")

            for veff, prob in effect_probs:
                effects_str = "(and " + " ".join([effect.pddl() for effect in veff]) + ")"
                action_lines.append(f"      {prob} {effects_str}")

            action_lines.append("    )")

            action_lines.append("  )")  # Close Action
            return action_lines

        action_lines = []
        objects_str = ""
        for t, sublist in store.type_dict.items():
            for obj_name in sublist:
                objects_str += f"{obj_name} - {t} "

        action_lines.append(f"  (:constants {objects_str})")

        for asc in self.action_schemas:
            if isinstance(asc, NoOp):  # Skip NoOp
                continue

            default_negations = []
            for cost_modifier in cost_modifiers:
                if cost_modifier.action.name == asc.name:
                    choices = all_outcome_combos(asc)
                    obj_eq = [
                        Eq(arg, obj) for arg, obj in zip(asc.inputs, cost_modifier.action.args)
                    ]
                    default_negations.append(Not(And(obj_eq)))
                    choice_combos = list(itertools.product(*choices))
                    veff_probs = [
                        (outcome_combo_to_expr(asc, choice_combo), 1.0 / len(choice_combos))
                        for choice_combo in choice_combos
                    ]
                    action_lines += get_prob_action(
                        cost_modifier.action.name + ACTION_EXT.join(cost_modifier.action.args),
                        asc.inputs,
                        asc.input_types,
                        asc.preconditions + obj_eq + cost_modifier.pre_facts,
                        veff_probs,
                    )

            choices = all_outcome_combos(asc)
            choice_combos = list(itertools.product(*choices))
            veff_probs = [
                (outcome_combo_to_expr(asc, choice_combo), 1.0 / len(choice_combos))
                for choice_combo in choice_combos
            ]
            action_lines += get_prob_action(
                asc.name,
                asc.inputs,
                asc.input_types,
                asc.preconditions + default_negations,
                veff_probs,
            )

        return "\n".join(action_lines)

    def action_pddl(self, default_cost):
        action_lines = []
        for action_schema in self.action_schemas:
            if isinstance(action_schema, NoOp):  # Skip NoOp
                continue

            choices = all_outcome_combos(action_schema)
            for outcome_combo in itertools.product(*choices):
                veff_idx_str = VEFFECT_SEPARATOR.join([str(i) for i in list(outcome_combo)])
                veff_action_name = f"{action_schema.name}{ACTION_EXT}{veff_idx_str}"
                action_lines.append(f"  (:action {veff_action_name}")

                # Parameters
                params = " ".join(
                    [
                        f"{arg} - {t}"
                        for arg, t in zip(action_schema.inputs, action_schema.input_types)
                    ]
                )
                action_lines.append(f"    :parameters ({params})")

                preconditions = copy.deepcopy(action_schema.preconditions[:])
                preconditions.append(Not(Atom("goal")))

                preconditions_str = (
                    " (and " + " ".join([precon.pddl() for precon in preconditions]) + ")"
                )
                action_lines.append(f"    :precondition{preconditions_str}")

                # Effects
                veff = outcome_combo_to_expr(action_schema, outcome_combo)
                all_eff = list(veff) + action_schema.effects + [IncreaseCost(default_cost)]

                if all_eff:  # Using verify_effects for effects
                    effects_str = " (and " + " ".join([effect.pddl() for effect in all_eff]) + ")"
                    action_lines.append(f"    :effect{effects_str}")

                action_lines.append("  )")  # Close Action

        return "\n".join(action_lines)

    def to_pddl_domain(self, default_cost: float = 1.0) -> str:
        domain_lines = []
        domain_lines.append("(define (domain generated)")
        domain_lines.append("  (:requirements :strips :typing)")
        return "\n".join(
            domain_lines
            + [self.header_pddl(), self.action_pddl(default_cost), self.goal_pddl(), ")"]
        )

    def to_ppddl_domain(self, cost_modifiers: List[CostModifier], store: AliasStore) -> str:
        domain_lines = []
        domain_lines.append("(define (domain generated)")
        domain_lines.append(
            "  (:requirements :strips :typing :conditional-effects :probabilistic-effects :equality :rewards)"
        )
        return "\n".join(
            domain_lines
            + [self.header_pddl(), self.action_ppddl(cost_modifiers, store=store), ")"]
        )

    def to_ppddl_problem(self, abstract_belief: AbstractBelief, store: AliasStore) -> str:
        pddl_str = []

        # Problem Header
        pddl_str.append(f"(define (problem generated_0)")
        pddl_str.append(
            f"  (:domain generated)"
        )  # You might want to make this a parameter or derive it from somewhere

        # Init (including unary object type predicates)
        pddl_str.append("  (:init")
        for obj_type, objs in store.type_dict.items():
            for obj in objs:
                pddl_str.append(f"    ({obj_type} {obj})")
                pddl_str.append(f"    (eq {obj} {obj})")

        for expr in abstract_belief.items + store.certified:
            pddl_str.append(f"    {expr.pddl()}")
        pddl_str.append("  )")  # Close Init

        # Goal
        goal_expr = self.reward  # Assuming reward expression is the goal
        pddl_str.append(f"  (:goal {goal_expr.pddl()})")
        pddl_str.append(f"  (:metric maximize (reward))")
        pddl_str.append(")")  # Close Problem

        return "\n".join(pddl_str)

    def to_pddl_problem(self, abstract_belief: AbstractBelief, store: AliasStore) -> str:
        pddl_str = []

        # Problem Header
        pddl_str.append(f"(define (problem generated_0)")
        pddl_str.append(
            f"  (:domain generated)"
        )  # You might want to make this a parameter or derive it from somewhere

        # Objects
        objects_str = ""
        for t, sublist in store.type_dict.items():
            for obj_name in sublist:
                objects_str += f"{obj_name} - {t} "

        pddl_str.append(f"  (:objects {objects_str})")

        # Init (including unary object type predicates)
        pddl_str.append("  (:init")
        for obj_type, objs in store.type_dict.items():
            for obj in objs:
                pddl_str.append(f"    ({obj_type} {obj})")
                pddl_str.append(f"    (eq {obj} {obj})")

        for expr in abstract_belief.items + store.certified:
            pddl_str.append(f"    {expr.pddl()}")
        pddl_str.append("  )")  # Close Init

        # Goal
        goal_expr = self.reward  # Assuming reward expression is the goal
        pddl_str.append(f"  (:goal {goal_expr.pddl()})")
        pddl_str.append(f"  (:metric minimize (total-cost))")
        pddl_str.append(")")  # Close Problem

        return "\n".join(pddl_str)

    def save_files(
        self, domain_str: str, problem_str: str, folder: str = "temp", prefix: str = ""
    ):
        if not os.path.exists(folder):
            os.makedirs(folder)

        domain_file = os.path.join(folder, "{}domain.pddl".format(prefix))
        problem_file = os.path.join(folder, "{}problem.pddl".format(prefix))

        with open(domain_file, "w") as f:
            f.write(domain_str)

        with open(problem_file, "w") as f:
            f.write(problem_str)

        return domain_file, problem_file

    def save_pddl(
        self, abstract_belief: AbstractBelief, default_cost: int, folder: str, store: AliasStore
    ) -> str:
        domain_str = self.to_pddl_domain(default_cost=default_cost)
        problem_str = self.to_pddl_problem(abstract_belief, store)
        return self.save_files(domain_str, problem_str, folder=folder)

    def save_ppddl(
        self,
        abstract_belief: AbstractBelief,
        cost_modifiers: List[CostModifier],
        folder: str,
        store: AliasStore,
    ) -> str:
        domain_str = self.to_ppddl_domain(cost_modifiers, store)
        problem_str = self.to_ppddl_problem(abstract_belief, store)

        if not os.path.exists(folder):
            os.makedirs(folder)

        problem_file = os.path.join(folder, "ppddl_problem.pddl")

        with open(problem_file, "w") as f:
            f.write(domain_str + "\n" + problem_str)

        return problem_file


@dataclass
class CostModifier:
    action: Action
    pre_facts: List[Expr]
    effects_encoding: List[bool]  # An encoding of whether each verified effect was true or false

    def __hash__(self):
        return hash(tuple([self.action] + self.pre_facts + self.effects_encoding))

    def action_str(self):
        post_encoding = VEFFECT_SEPARATOR.join([str(int(e)) for e in self.effects_encoding])
        arg_encoding = " ".join([arg for arg in self.action.args])
        return f"{self.action.name}{ACTION_EXT}{post_encoding} {arg_encoding}"


def compute_cost_modifiers(
    spec: ProblemSpec,
    F: AbstractTransitionModel,
    learning_strategy: str,
    gamma: float,
    store: AliasStore,
):
    global_counts = F.total_count()
    # Collect a list of exceptions from the abstract transition model
    exceptions = defaultdict(lambda: defaultdict(list))
    for a, b_dict in F.effects.items():
        schema = spec.get_action_schema(a.name)
        subs = {k: v for k, v in zip(schema.inputs, a.args)}
        subbed_effects = [substitute(e, subs) for e in schema.effects]
        for b_pre, b_eff in b_dict.items():
            closed_b_pre = b_pre.closed_fluents(
                spec.fluent_predicates(), schema.depends, subs, store=store
            )
            for outcomes in itertools.product(*all_outcome_combos(schema)):
                exceptions[(a, closed_b_pre)][tuple(outcomes)].append((0, 0, 0))

            assert len(b_eff.ab_counts) > 0
            for b_tar in b_eff.all_ab():
                oneof_mismatch = False
                v_eff_outcome = []
                v_eff_encoding = []
                for v_eff in schema.verify_effects:
                    if isinstance(v_eff, Atom) or isinstance(v_eff, Not):
                        ev = eval_expr(v_eff, subs, b_tar.items, store.type_dict)
                        v_eff_encoding.append(int(ev))
                        if ev:
                            v_eff_outcome.append(substitute(v_eff, subs))
                        else:
                            v_eff_outcome.append(negate(substitute(v_eff, subs)))
                    elif isinstance(v_eff, OneOf):
                        encoding = []
                        for v_eff_oo in v_eff.components:
                            ev = eval_expr(v_eff_oo, subs, b_tar.items, store.type_dict)
                            encoding.append(int(ev))
                            if ev:
                                v_eff_outcome.append(substitute(v_eff_oo, subs))
                            else:
                                v_eff_outcome.append(negate(substitute(v_eff_oo, subs)))

                        if sum(encoding) == 1:
                            v_eff_encoding.append(encoding.index(1))
                        else:
                            v_eff_encoding.append(None)
                            oneof_mismatch = True

                sym_eff = symbolic_eff(
                    b=b_pre, effect=AbstractEffect(v_eff_outcome + subbed_effects), store=store
                )

                if sym_eff == b_tar and not oneof_mismatch:
                    exceptions[(a, closed_b_pre)][tuple(v_eff_encoding)].append(
                        b_eff.get_all_counts(b_tar)
                    )
                else:
                    difference1 = set(b_tar.items) - set(sym_eff.items)
                    difference2 = set(sym_eff.items) - set(b_tar.items)
                    logging.debug(
                        "WARNING: unexpected atoms {} observed in the effects of {}".format(
                            difference1, a
                        )
                    )
                    logging.debug(
                        "WARNING: expected atoms {} not observed in the effects of {}".format(
                            difference2, a
                        )
                    )
                    if oneof_mismatch:
                        logging.debug(
                            "WARNING: oneof encoding mismatch. Encoding: {}".format(
                                str(v_eff_encoding)
                            )
                        )
                    exceptions[(a, closed_b_pre)][None].append(b_eff.get_all_counts(b_tar))

    # Need to renormalize the exception list
    cost_modifiers = {}
    for (a, b_pre), av_eff in exceptions.items():
        total_counts = sum([item[0] for sublist in av_eff.values() for item in sublist])
        for eff, count_lists in av_eff.items():
            (outcome_count, success_count, attempt_count) = zip(*count_lists)
            if eff is not None:
                if learning_strategy == "bayes_optimistic":
                    cost = int(
                        -np.log(
                            beta_quantile(
                                sum(success_count),
                                sum(attempt_count) - sum(success_count),
                                global_counts,
                            )
                            * gamma
                        )
                        * (1 / (1 - gamma))
                    )
                elif learning_strategy == "mdp_guided":
                    if sum(outcome_count) == 0:
                        cost = MAX_COST
                    else:
                        cost = int(
                            -np.log(sum(outcome_count) / float(total_counts)) * (1 / (1 - gamma))
                        )
                cost_modifiers[CostModifier(a, b_pre.items, list(eff))] = cost

    return cost_modifiers


def inject_action_costs(
    output_sas_file: str,
    initial_ab: AbstractBelief,
    action_costs: Dict[CostModifier, float],
    store: AliasStore,
):
    # Quick and dirty way to inject action costs into the SAS+ file
    with open(output_sas_file, "r") as file:
        content = file.read()

    variable_strs = content.split("begin_variable")
    var_dict = {}

    for vs in variable_strs[1:]:
        atom_strs = [a for a in vs.split("\n")[4:-1]]

        for aidx, atom_str in enumerate(atom_strs):
            if atom_str.startswith("Atom"):
                paren_open = atom_str.index("(")
                paren_close = atom_str.index(")")
                atom_name = atom_str[:paren_open].replace("Atom ", "")
                atom_args = atom_str[paren_open + 1 : paren_close].split(", ")
                var_dict[Atom(atom_name, [a for a in atom_args if a != ""])] = (
                    vs.split("\n")[1],
                    aidx,
                )
            elif atom_str.startswith("NegatedAtom"):
                paren_open = atom_str.index("(")
                paren_close = atom_str.index(")")
                atom_name = atom_str[:paren_open].replace("NegatedAtom ", "")
                atom_args = atom_str[paren_open + 1 : paren_close].split(", ")
                var_dict[Not(Atom(atom_name, [a for a in atom_args if a != ""]))] = (
                    vs.split("\n")[1],
                    aidx,
                )

    all_lines = content.split("\n")
    name_to_costs = defaultdict(list)
    for action_cost in action_costs.items():
        name_to_costs[action_cost[0].action_str()].append(action_cost)

    for name in name_to_costs:
        cond_sdacs = []
        for cost_modifier, cost in name_to_costs[name]:
            multi = []
            for fact in cost_modifier.pre_facts:
                if fact in var_dict:
                    multi.append(
                        "(1-(abs({} - {}) > 0))".format(var_dict[fact][0], var_dict[fact][1])
                    )
                elif negate(fact) in var_dict:
                    multi.append(
                        "(abs({} - {}) > 0)".format(
                            var_dict[negate(fact)][0], var_dict[negate(fact)][1]
                        )
                    )
                else:
                    if isinstance(fact, Atom):
                        # Special case when a value is constant so it isn't declared a variable in the generated sas+.
                        # We have to find it's constant value in the initial state.
                        if not eval_expr(
                            fact, {}, initial_ab.items + store.certified, store.type_dict
                        ):
                            # This exception expects it to be true, but it is not in the initial state and is constant
                            # Otherwise we don't need to multiply by 1
                            multi.append("0")
                    elif isinstance(fact, Not):
                        if eval_expr(
                            fact.component, {}, initial_ab.items + store.certified, store.type_dict
                        ):
                            multi.append("0")

            if len(multi) > 0:
                cond_sdacs.append("({})".format(" * ".join([str(cost)] + multi)))
            else:
                cond_sdacs = [str(cost)]

        if name not in all_lines:
            # Again handling sas+ efficiency pruning.
            # This action will never be executable, so its cost isn't relevant
            continue

        action_cost_index = all_lines.index(name)

        end_index = all_lines[action_cost_index:].index("end_operator")

        if len(cond_sdacs) == 0:
            sdac = "0"
        elif len(cond_sdacs) == 1:
            sdac = cond_sdacs[0]
        else:
            sdac = "({})".format(" + ".join(cond_sdacs))

        sdac = "(1 + {})".format(sdac)
        all_lines[action_cost_index + end_index - 1] = sdac

    new_content = "\n".join(all_lines)
    with open(output_sas_file, "w") as file:
        file.write(new_content)

    return output_sas_file


class SymbolicBelief(Belief):
    """A particular type of belief that is represented entirely with symbols
    and exactly mimics the dynamics of the ProblemSpec.

    Primarily used for debugging purposes
    """

    def __init__(self, spec: ProblemSpec, initial: List[Expr] = []):
        self.items = initial
        self.schemas = {s.name: s for s in spec.action_schemas}

    def update(self, action: Action, obs: Observation, store: AliasStore) -> Belief:
        new_belief = copy.deepcopy(self)
        schema = self.schemas[action.name]
        ab = new_belief.abstract(store)
        abstract_effect = AbstractEffect(
            [random.choice([a, negate(a)]) for a in schema.verify_effects]
        )
        new_belief.items = symbolic_update(ab, action, schema, abstract_effect, store).items
        return new_belief

    def abstract(self, store: AliasStore) -> AbstractBelief:
        return AbstractBelief(self.items)
