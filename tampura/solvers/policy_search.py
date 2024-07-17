from __future__ import annotations

import copy
import logging
import random
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np

from tampura.solvers.symk import symk_search, symk_translate
from tampura.spec import (
    ProblemSpec,
    beta_quantile,
    compute_cost_modifiers,
    inject_action_costs,
)
from tampura.structs import (
    AbstractBelief,
    AbstractBeliefSet,
    AbstractEffect,
    AbstractRewardModel,
    AbstractRollout,
    AbstractSolution,
    AbstractTransitionModel,
    Action,
    AliasStore,
    Atom,
    Belief,
    Stream,
    eval_expr,
    substitute,
    symbolic_eff,
    symbolic_update,
)
from tampura.symbolic import (
    ACTION_EXT,
    OPT_OBJ,
    VARIABLE_PREFIX,
    VEFFECT_SEPARATOR,
    And,
    Not,
    OneOf,
    negate,
    replace_arg,
)

MAX_COST = 1000


def normalize(F: AbstractTransitionModel) -> AbstractTransitionModel:
    """Normalizes an abstract transition model such that all transition
    probabilities sum to 1.

    If the counts on a transition are zero, the probability mass is
    split equally between all possible outcomes
    """
    norm_F = copy.deepcopy(F)
    for _, transitions in norm_F.effects.items():
        for _, abs_belief_set in transitions.items():
            total_counts = sum(abs_belief_set.ab_counts.values())

            if total_counts == 0:
                assert False
            else:
                for ab, count in abs_belief_set.ab_counts.items():
                    abs_belief_set.ab_counts[ab] = count / total_counts

    return norm_F


def generate_rollouts(
    a_b0: AbstractBelief,
    F: AbstractTransitionModel,
    sol: AbstractSolution,
    num_rollouts=1,
    max_steps=10,
) -> List[AbstractRollout]:
    rollouts = []

    for _ in range(num_rollouts):
        current_belief = a_b0
        rollout_transitions = []
        for _ in range(max_steps):
            if current_belief not in sol.policy:
                break  # We have reached an unhandled belief, so we stop.

            action = sol.policy[current_belief]
            if current_belief not in F.effects[action]:
                break
            next_belief_set = F.effects[action][current_belief]
            next_belief = next_belief_set.sample()

            rollout_transitions.append((action, current_belief, next_belief))
            current_belief = next_belief

        rollouts.append(AbstractRollout(rollout_transitions))

    return rollouts


def plan_to_rollout(
    spec: ProblemSpec, plan: List[Action], ab_0: AbstractBelief, store: AliasStore
) -> AbstractRollout:
    transitions = []
    ab = ab_0

    for unprocessed_action in plan:
        action_name, veffect_str = unprocessed_action.name.split(ACTION_EXT)
        action = Action(action_name, unprocessed_action.args)
        s = spec.get_action_schema(action.name)
        effect_items = copy.deepcopy(s.effects)
        if len(veffect_str) > 0:
            actives = [int(v) for v in veffect_str.split(VEFFECT_SEPARATOR)]
            for active, ve in zip(actives, s.verify_effects):
                if isinstance(ve, Atom) or isinstance(ve, Not):
                    if active:
                        effect_items.append(ve)
                    else:
                        effect_items.append(negate(ve))
                elif isinstance(ve, OneOf):
                    for oo_idx, oo_elem in enumerate(ve.components):
                        if oo_idx == active:
                            effect_items.append(oo_elem)
                        else:
                            effect_items.append(negate(oo_elem))
                else:
                    raise NotImplementedError

        action_effect = AbstractEffect(effect_items)
        ab_p = symbolic_update(ab, action, s, action_effect, store)
        transitions.append((action, ab, ab_p))
        ab = ab_p

    return AbstractRollout(transitions)


def get_stream_plan(
    a: Action, ab: AbstractBelief, continuous_arg: str, spec: ProblemSpec, store: AliasStore
):
    """Given a abstract belief and action along with a continuous argument of
    the action that we want to create a new continuous sample of, return the
    shortest sequence of stream executions that generates a new continuous_arg
    sample while satisfying the preconditions of the input action."""

    # Abandon all hope ye who enter
    action_schema = spec.get_action_schema(a.name)
    action_arg_map = {k: v for k, v in zip(action_schema.inputs, a.args)}

    new_arg = continuous_arg.replace(VARIABLE_PREFIX, OPT_OBJ)
    new_preconditions = []
    for p in action_schema.preconditions:
        new_preconditions.append(replace_arg(p, continuous_arg, new_arg))

    full_type_dict = defaultdict(list)
    for arg_type, arg in zip(action_schema.input_types, a.args):
        full_type_dict[arg_type].append(arg)

    # An open stream plan is a set of known facts and a sequence of stream calls
    open_stream_plans = [(full_type_dict, store.certified + ab.items, [], new_preconditions)]
    while len(open_stream_plans) > 0:
        type_dict, known, stream_plan, subbed_preconditions = open_stream_plans.pop(0)
        for stream_schema in spec.stream_schemas:
            for stream_arg_tuple in product(*[type_dict[t] for t in stream_schema.input_types]):
                # Substitute the args into the stream schema certified effects
                continuous_s_arg = stream_schema.output
                new_s_arg = continuous_s_arg.replace(VARIABLE_PREFIX, OPT_OBJ)
                new_subbed_preconditions = []
                for p in subbed_preconditions:
                    new_subbed_preconditions.append(replace_arg(p, continuous_s_arg, new_s_arg))

                subs = {k: v for k, v in zip(stream_schema.inputs, stream_arg_tuple)}

                if eval_expr(And(stream_schema.preconditions), subs, known, store.type_dict):
                    subs[continuous_s_arg] = new_s_arg
                    subbed_certified = [substitute(cert, subs) for cert in stream_schema.certified]
                    new_facts = symbolic_eff(
                        AbstractBelief(known), AbstractEffect(subbed_certified), store
                    )

                    if AbstractBelief(known) != new_facts:
                        new_type_dict = copy.deepcopy(type_dict)
                        new_type_dict[stream_schema.output_type] = [new_s_arg]
                        stream = Stream(stream_schema.name, stream_arg_tuple, new_s_arg)
                        new_stream_plan = (
                            new_type_dict,
                            new_facts.items,
                            stream_plan + [stream],
                            new_subbed_preconditions,
                        )
                        if eval_expr(
                            substitute(And(new_subbed_preconditions), action_arg_map),
                            {},
                            new_facts.items,
                            store.type_dict,
                        ):
                            return new_stream_plan[2]
                        else:
                            open_stream_plans.append(new_stream_plan)

    return None


def execute_stream_plan(
    stream_plan: List[Stream], spec: ProblemSpec, store: AliasStore
) -> AliasStore:
    """Execute a stream plan by sampling from the stream samplers."""
    new_arg_map = {}
    for stream in stream_plan:
        ss = spec.get_stream_schema(stream.name)
        input_map = {k: v for k, v in zip(ss.inputs, stream.inputs)}

        # Replace the input placeholders with the recently generated objects
        for k, v in input_map.items():
            if v in new_arg_map:
                input_map[k] = new_arg_map[v]

        sample_inputs = [input_map[k] for k in ss.inputs]
        assert None not in store.get_all(sample_inputs)
        output = ss.sample_fn(sample_inputs, store)
        output_sym = store.add_typed(output, ss.output_type)
        new_arg_map[stream.output] = output_sym
        subbed_cert = [
            substitute(cert, input_map | {ss.output: output_sym}) for cert in ss.certified
        ]
        store.certified += subbed_cert
    return store


def progressive_widening(
    a: Action, ab: AbstractBelief, spec: ProblemSpec, alpha: float, k: float, store: AliasStore
) -> AliasStore:
    action_schema = spec.get_action_schema(a.name)
    cont_stream_plans = {}
    # Split up the action input arguments into discrete and continuous depending on if they are generated by a stream
    discrete_args = [
        arg
        for (arg, t) in zip(action_schema.inputs, action_schema.input_types)
        if t not in spec.continuous_types
    ]
    continuous_args = [arg for arg in action_schema.inputs if arg not in discrete_args]

    discrete_component = tuple(
        [a.args[action_schema.inputs.index(discrete_arg)] for discrete_arg in discrete_args]
    )
    d_action = Action(a.name, discrete_component)

    store.sample_counts[d_action] = store.get_sample_count(d_action) + 1

    if len(continuous_args) > 0:
        if k * (store.get_sample_count(d_action) ** alpha) >= store.get_branching_factor(d_action):
            logging.info(
                "Progressive widening on action {}, {}>{}".format(
                    d_action,
                    k * (store.get_sample_count(d_action) ** alpha),
                    store.branching_factor[d_action],
                )
            )

            store.branching_factor[d_action] += 1
            for continuous_arg in continuous_args:
                stream_plan = get_stream_plan(a, ab, continuous_arg, spec, store)
                if stream_plan is not None:
                    cont_stream_plans[continuous_arg] = stream_plan
            if len(cont_stream_plans) > 0:
                # Choose a random continuous argument to widen and query the streams to generate this new continuous value
                selected_stream_plan = random.choice(list(cont_stream_plans.values()))
                store = execute_stream_plan(selected_stream_plan, spec, store)

    return store


def policy_search(
    b0: Belief,
    spec: ProblemSpec,
    F: AbstractTransitionModel,
    R: AbstractRewardModel,
    belief_map: Dict[AbstractBelief, List[Belief]],
    store: AliasStore,
    config: Dict[str, Any],
    save_dir: str,
) -> Tuple[AbstractTransitionModel, AbstractRewardModel, Dict[AbstractBelief, List[Belief]], bool]:
    """Sample trajectories according to the provided abstract policy."""

    a_b0 = b0.abstract(store)

    default_cost = int(
        -np.log(beta_quantile(0, 0, F.total_count()) * config["gamma"]) * 1 / (1 - config["gamma"])
    )

    cost_modifiers = compute_cost_modifiers(
        spec, F, config["learning_strategy"], config["gamma"], store
    )

    (domain_file, problem_file) = spec.save_pddl(
        a_b0, default_cost=default_cost, folder=save_dir, store=store
    )

    # _ = spec.save_ppddl(a_b0, cost_modifiers=cost_modifiers, folder=save_dir, store=store)

    output_sas_file = symk_translate(domain_file, problem_file)

    output_sas_file = inject_action_costs(
        output_sas_file,
        a_b0,
        action_costs=cost_modifiers,
        store=store,
    )
    plans = symk_search(output_sas_file, config)

    if len(plans) == 0:
        return F, R, belief_map, False
    else:
        rollouts = [plan_to_rollout(spec, plan, a_b0, store) for plan in plans]

    for step in range(config["batch_size"]):
        rollout = random.choice(rollouts)
        if len(rollout.transitions) == 0:
            continue

        # Sort transitions by num collected samples
        filtered_transitions = [t for t in rollout.transitions if t[1] in belief_map]
        assert len(filtered_transitions) > 0

        max_info_transition = sorted(
            filtered_transitions,
            key=lambda t: sum(F.get_transition(t[0], t[1]).ab_counts.values()),
        )[0]
        mi_a, mi_ab, mi_ab_p = max_info_transition
        action_schema = spec.get_action_schema(mi_a.name)
        sel_b = None
        if any([o is None for o in store.get_all(mi_a.args)]):
            logging.debug(
                "Step {} Skipping Action {} w/ Objects {} and Outcome {}".format(
                    step, mi_a, store.get_all(mi_a.args), mi_ab_p
                )
            )
            ab_p_belief_set = AbstractBeliefSet(ab_counts={mi_ab: 1e6}, belief_map={mi_ab: []})
        else:
            logging.debug("Step {} Sampling Action {} w/ Outcome {}".format(step, mi_a, mi_ab_p))
            logging.debug(
                "From abstract belief: {} with {} beliefs".format(mi_ab, len(belief_map[mi_ab]))
            )

            sel_b = random.choice(belief_map[mi_ab])
            ab_p_belief_set = action_schema.effects_fn(mi_a, sel_b, store)

        if not config["flat_sample"]:
            # Progressive widening
            store = progressive_widening(mi_a, mi_ab, spec, config["pwa"], config["pwk"], store)

        for ab_p, belief_set in ab_p_belief_set.belief_map.items():
            if ab_p == mi_ab_p:
                ab_p_belief_set.outcome_successes[mi_ab_p] = ab_p_belief_set.ab_counts[ab_p]

            belief_map[ab_p] += belief_set
            R.reward[ab_p] = spec.get_reward(ab_p, store)

        ab_p_belief_set.outcome_attempts[mi_ab_p] = ab_p_belief_set.total_count()

        F.update(mi_a, mi_ab, ab_p_belief_set)

    return F, R, belief_map, True
