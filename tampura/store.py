from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tampura.symbolic import OBJ, Action, Expr


@dataclass
class AliasStore:
    """A store for mapping between names and python objects."""

    als: Dict[str, Any] = field(default_factory=lambda: {})
    als_type: Dict[str, Any] = field(default_factory=lambda: {})
    alph_counter: Dict[str, int] = field(default_factory=lambda: {})
    certified: List[Expr] = field(default_factory=lambda: [])

    # Keeps track of the number of times an object was used
    # in an effects simulation. Important bookkeeping for
    # progressive widening
    sample_counts: Dict[Action, int] = field(default_factory=lambda: {})
    branching_factor: Dict[Action, int] = field(default_factory=lambda: {})

    def get_branching_factor(self, action: Action):
        if action not in self.branching_factor:
            self.branching_factor[action] = 0
        return self.branching_factor[action]

    def get_sample_count(self, action: Action):
        if action not in self.sample_counts:
            self.sample_counts[action] = 0
        return self.sample_counts[action]

    @property
    def type_dict(self):
        type_dict = defaultdict(list)
        for k, v in self.als_type.items():
            type_dict[v].append(k)
        return type_dict

    def get_alph_count(self, prefix):
        if prefix not in self.alph_counter:
            self.alph_counter[prefix] = 0

        name = self.alph_counter[prefix]
        self.alph_counter[prefix] += 1
        return name

    def refine(self, objects):
        new_als = {}
        new_als_type = {}
        for item in objects:
            new_als[item] = self.als[item]
            new_als_type[item] = self.als_type[item]
        self.als = new_als
        self.als_type = new_als_type

    def add_typed(self, el, type, prefix=OBJ):
        name = f"{prefix}{type[:2]}_{self.get_alph_count(type[:2])}"
        self.als[name] = el
        self.als_type[name] = type
        return name

    def add_objects(self, objects, type):
        for obj in objects:
            self.set(obj, obj, type)

    def get(self, key):
        if key in self.als:
            return self.als[key]
        else:
            return None

    def set(self, key, value, type):
        self.als[key] = value
        self.als_type[key] = type
        return key

    def get_all(self, keys):
        return [self.get(key) for key in keys]
