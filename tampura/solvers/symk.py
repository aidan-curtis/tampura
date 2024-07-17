from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from tampura.structs import Action


def parse_action_from_line(line: str) -> Optional[Action]:
    match = re.match(r"\(([^ ]+)(?: (.*?))?\s*\)", line)
    if match:
        action_name = match.group(1)
        action_args = match.group(2).split() if match.group(2) else []
        return Action(name=action_name, args=action_args)
    return None


def parse_actions_from_file(filename: str) -> List[Action]:
    actions = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            action = parse_action_from_line(line)
            if action:
                actions.append(action)
    return actions


def symk_translate(domain_file: str, problem_file: str) -> List[List[Action]]:
    # Extract the directory of the domain file
    domain_dir = os.path.dirname(domain_file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downward = os.path.join(script_dir, "../../third_party/symk/fast-downward.py")
    sas_file = os.path.join(domain_dir, "output.sas")
    translate_command = [
        "python",
        downward,
        "--sas-file",
        sas_file,
        "--translate",
        domain_file,
        problem_file,
        "--translate-options",
        "--keep-unimportant-variables",
        "--keep-unreachable-facts",
    ]
    logging.debug(" ".join(translate_command))
    out = subprocess.run(translate_command, capture_output=True, text=True)
    return sas_file


def symk_search(sas_file: str, config: Dict[str, Any]) -> List[List[Action]]:
    domain_dir = os.path.dirname(sas_file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downward = os.path.join(script_dir, "../../third_party/symk/fast-downward.py")
    search_command = [
        "python",
        downward,
        "--search-time-limit",
        "20",
        "--plan-file",
        os.path.join(domain_dir, "sas_plan"),
        sas_file,
        "--search-options",
        "--search",
        "symk-{}(simple={},plan_selection={}(num_plans={},dump_plans=false))".format(
            config["symk_direction"],
            str(config["symk_simple"]).lower(),
            config["symk_selection"],
            str(config["num_skeletons"]),
        ),
    ]

    logging.debug(" ".join(search_command))
    out = subprocess.run(search_command, capture_output=True, text=True)

    logging.debug(out.stdout)
    logging.debug(out.stderr)

    plan_files = [
        os.path.join(domain_dir, file)
        for file in os.listdir(domain_dir)
        if file.startswith("sas_plan")
    ]
    if len(plan_files) == 0:
        logging.info("Symk planning failure")
        return []

    plans = []
    for plan_file in plan_files:
        plans.append(parse_actions_from_file(plan_file))

    return plans
