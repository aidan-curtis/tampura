from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

####################################

VARIABLE_PREFIX = "?"
OBJ = "o_"
OPT_OBJ = "#"
ACTION_EXT = "_cm_"
VEFFECT_SEPARATOR = "."


@dataclass
class Expr:
    def __eq__(self, a2):
        return hash(self) == hash(a2)

    def __lt__(self, a2):
        return hash(self) < hash(a2)

    def __gt__(self, a2):
        return hash(self) > hash(a2)


@dataclass
class Quantifier(Expr):
    component: Any
    args: List[str]
    types: List[str]

    def __str__(self):
        typed_args = ", ".join(
            ["{} : {}".format(a, str(t)) for a, t in zip(self.args, self.types)]
        )
        return "{}_{{ {} }} [{}]".format(
            self.get_quant_str(), str(typed_args), str(self.component)
        )

    def pddl(self):
        typed_args = " ".join([f"{a} - {t}" for a, t in zip(self.args, self.types)])

        # Constructing the quantifier
        quant_str = "{} ({})".format(self.get_quant_str(), typed_args)

        return "({} {})".format(quant_str, self.component.pddl())

    def __hash__(self):
        return hash(tuple([self.component] + self.args + self.types))


@dataclass
class ForAll(Quantifier):
    def get_quant_str(self):
        return "forall"

    def __hash__(self):
        return hash(tuple([self.get_quant_str()] + [self.component] + self.args + self.types))


@dataclass
class Exists(Quantifier):
    def get_quant_str(self):
        return "exists"

    def __hash__(self):
        return hash(tuple([self.get_quant_str()] + [self.component] + self.args + self.types))


@dataclass
class Imply(Expr):
    a: Any
    b: Any

    def __str__(self):
        return f"{self.a} => {self.b}"

    def __hash__(self):
        return hash(tuple[self.a, self.b])

    def pddl(self):
        return f"(imply {self.a.pddl()} {self.b.pddl()})"


@dataclass
class When(Expr):
    a: Any
    b: Any

    def __str__(self):
        return f"when {self.a} then {self.b}"

    def __hash__(self):
        return hash(tuple([self.a, self.b]))

    def pddl(self):
        return f"(when {self.a.pddl()} {self.b.pddl()})"


@dataclass
class Not(Expr):
    component: Any

    def __str__(self):
        return "~({})".format(self.component)

    def __hash__(self):
        return hash(tuple(["~", self.component.__hash__()]))

    def pddl(self):
        return f"(not {self.component.pddl()})"


@dataclass
class Num(Expr):
    value: float

    def __str__(self):
        return self.pddl()

    def pddl(self):
        return str(self.value)


@dataclass
class Mult(Expr):
    components: List[Expr]

    def __str__(self):
        return self.pddl()

    def pddl(self):
        pddls = " ".join([c.pddl() for c in self.components])
        return f"(* {pddls})"


@dataclass
class Add(Expr):
    components: List[Expr]

    def __str__(self):
        return self.pddl()

    def pddl(self):
        pddls = " ".join([c.pddl() for c in self.components])
        return f"(+ {pddls})"


@dataclass
class And(Expr):
    components: Any

    def __str__(self):
        return " & ".join([str(c) for c in self.components])

    def pddl(self):
        return "(and " + " ".join([c.pddl() for c in self.components]) + ")"


@dataclass
class OneOf(Expr):
    components: Any

    def __str__(self):
        return " & ".join([str(c) for c in self.components])

    def pddl(self):
        return "(oneof " + " ".join([c.pddl() for c in self.components]) + ")"


@dataclass
class Or(Expr):
    components: Any

    def __str__(self):
        return " | ".join([str(c) for c in self.components])

    def pddl(self):
        return "(or " + " ".join([c.pddl() for c in self.components]) + ")"


@dataclass
class Atom(Expr):
    pred_name: str
    args: List[str] = field(default_factory=lambda: [])

    def __str__(self):
        if len(self.args) > 0:
            return "{}({})".format(self.pred_name, ", ".join([str(c) for c in self.args]))
        else:
            return "{}".format(self.pred_name)

    def __hash__(self):
        return hash(tuple([self.pred_name] + list(self.args)))

    def pddl(self):
        if self.args:
            return f"({self.pred_name} " + " ".join([f"{arg}" for arg in self.args]) + ")"
        else:
            return f"({self.pred_name})"


@dataclass
class Eq(Expr):
    a: str = 0
    b: str = 0

    def __str__(self):
        return f"{self.a}=={self.b}"

    def pddl(self):
        return f"(eq {self.a} {self.b})"

    def __hash__(self):
        return hash(tuple((self.a, self.b)))


@dataclass
class IncreaseCost(Expr):
    amount: int = 0.0  # current planner only supports integer costs

    def __str__(self):
        return self.pddl()

    def pddl(self):
        return f"(increase (total-cost) {self.amount})"

    def __hash__(self):
        return hash(self.amount)


def negate(e: Expr):
    if isinstance(e, Not):
        return e.component
    elif isinstance(e, Atom):
        return Not(e)
    else:
        raise NotImplementedError


####################################


@dataclass
class Predicate:
    name: Any
    arg_types: List[Any]
    grounding: Any = None

    def get_arg_names(self):
        arg_names = []
        counts = defaultdict(lambda: 0)
        for t in self.arg_types:
            arg_names.append("?{}{} - {}".format(t, str(counts[t]), t))
            counts[t] += 1
        return arg_names

    def pddl(self):
        arg_str = " ".join(self.get_arg_names())
        return f"({self.name} {arg_str})"

    def __hash__(self):
        return hash(tuple([self.name] + self.arg_types + [self.grounding]))

    def __lt__(self, o):
        return hash(self) < hash(o)


@dataclass
class Action:
    name: str = "default-action"
    args: List[Any] = field(default_factory=lambda: [])

    def __hash__(self):
        return hash(tuple([self.name] + list(self.args)))

    def __eq__(self, b):
        return hash(self) == hash(b)

    def __str__(self):
        return "{}({})".format(self.name, ", ".join(self.args))


# TODO: Move these three functions into the symbolic definitions
def substitute(value, subs) -> Any:
    if isinstance(value, str):
        if value in subs:
            value = subs[value]
    elif isinstance(value, And):
        return And([substitute(v, subs) for v in value.components])
    elif isinstance(value, Atom):
        return Atom(value.pred_name, [substitute(v, subs) for v in value.args])
    elif isinstance(value, Not):
        return Not(substitute(value.component, subs))
    elif isinstance(value, ForAll):
        return ForAll(substitute(value.component, subs), value.args, value.types)
    elif isinstance(value, Exists):
        return Exists(substitute(value.component, subs), value.args, value.types)
    elif isinstance(value, When):
        return When(substitute(value.a, subs), substitute(value.b, subs))
    elif isinstance(value, Eq):
        return Eq(substitute(value.a, subs), substitute(value.b, subs))
    else:
        print(f"{type(value)} not supported in substitute")
        raise NotImplementedError
    return value


def get_pred_names(expr: Expr) -> List[str]:
    if isinstance(expr, Not):
        return get_pred_names(expr.component)
    elif isinstance(expr, Atom):
        return [expr.pred_name]
    elif isinstance(expr, ForAll) or isinstance(expr, Exists):
        return get_pred_names(expr.component)
    elif isinstance(expr, When) or isinstance(expr, Eq):
        return get_pred_names(expr.a) + get_pred_names(expr.b)
    elif isinstance(expr, And) or isinstance(expr, Or) or isinstance(expr, OneOf):
        pred_names = []
        for e in expr.components:
            pred_names += get_pred_names(e)
        return pred_names
    elif isinstance(expr, str):
        return []
    else:
        print("get_pred_names type {} not implemented".format(type(expr)))
        raise NotImplementedError


def get_args(expr, prefix=VARIABLE_PREFIX) -> List[str]:
    """
    Given an expression, return the variables in that expression
    Input: Not(Atom("on", ["?o", "@goal"]))
    Output: ["?o"]
    """
    if isinstance(expr, Atom):
        return list(itertools.chain.from_iterable([get_args(c, prefix=prefix) for c in expr.args]))
    elif isinstance(expr, Not):
        return get_args(expr.component, prefix=prefix)
    elif isinstance(expr, str):
        return [expr] if expr.startswith(prefix) else []
    elif isinstance(expr, Quantifier):
        partial = get_args(expr.component, prefix=prefix)
        return [arg for arg in partial if arg not in expr.args]
    elif isinstance(expr, Imply):
        return list(set(get_args(expr.a) + get_args(expr.b)))
    elif isinstance(expr, Or) or isinstance(expr, And):
        return list(set(itertools.chain(*[get_args(component) for component in expr.components])))
    else:
        print(f"{type(expr)} not supported in get_args")
        raise NotImplementedError


def replace_arg(expr: Expr, old_arg: str, new_arg: str) -> Expr:
    if isinstance(expr, Atom):
        return Atom(expr.pred_name, [a if a != old_arg else new_arg for a in expr.args])
    elif isinstance(expr, Not):
        return Not(replace_arg(expr.component, old_arg, new_arg))
    elif isinstance(expr, And) or isinstance(expr, Or):
        return And([replace_arg(c, old_arg, new_arg) for c in expr.components])
    elif isinstance(expr, Exists) or isinstance(expr, ForAll):
        new_expr = copy.deepcopy(expr)
        new_expr.component = replace_arg(expr.component, old_arg, new_arg)
        return new_expr
    else:
        raise NotImplementedError


def eval_expr(
    expr: Expr, arg_map: Dict[str, str], facts: List[Expr], type_dict: Dict[str, List[str]]
) -> bool:
    if isinstance(expr, And):
        return all([eval_expr(q, arg_map, facts, type_dict) for q in expr.components])
    elif isinstance(expr, Or):
        return any([eval_expr(q, arg_map, facts, type_dict) for q in expr.components])
    elif isinstance(expr, Not):
        return not eval_expr(expr.component, arg_map, facts, type_dict)
    elif isinstance(expr, Atom):
        for atom in facts:
            if (
                isinstance(atom, Atom)
                and Atom(
                    expr.pred_name,
                    [substitute(a, arg_map) for a in expr.args],
                )
                == atom
            ):
                return True
        return False
    elif isinstance(expr, Imply):
        return (not eval_expr(expr.a, arg_map, facts, type_dict)) or eval_expr(
            expr.b, arg_map, facts, type_dict
        )

    elif isinstance(expr, Exists):
        for args in itertools.product(*[type_dict[t] for t in expr.types]):
            new_arg_map = {k: v for k, v in zip(expr.args, args)}
            if eval_expr(expr.component, {**new_arg_map, **arg_map}, facts, type_dict):
                return True
        return False
    elif isinstance(expr, ForAll):
        for args in itertools.product(*[type_dict[t] for t in expr.types]):
            new_arg_map = {k: v for k, v in zip(expr.args, args)}
            if not eval_expr(expr.component, {**new_arg_map, **arg_map}, facts, type_dict):
                return False
        return True
    elif isinstance(expr, Eq):
        return substitute(expr.a, arg_map) == substitute(expr.b, arg_map)
    else:
        print(f"{type(expr)} not supported in eval_expr")
        raise NotImplementedError
