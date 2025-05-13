import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Generic, Type, TypeVar

import yaml

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self.items: Dict[str, Type[T]] = {}

    def register(self, id: str, item_class: Type[T]):
        self.items[id] = item_class

    def get(self, id: str) -> Type[T]:
        return self.items[id]

    def list(self):
        return list(self.items.keys())


class EnvRegistry(Registry):
    pass


class PlannerRegistry(Registry):
    pass


env_registry = EnvRegistry()
planner_registry = PlannerRegistry()


def register_env(id: str, env_class):
    env_registry.register(id, env_class)


def get_env(id: str):
    return env_registry.get(id)


def register_planner(id: str, planner_class):
    planner_registry.register(id, planner_class)


def get_planner(id: str):
    return planner_registry.get(id)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="The config file to load from")
    parser.add_argument("--task", type=str),
    parser.add_argument("--planner", type=str)
    parser.add_argument(
        "--global-seed", help="The global rng seed set once before planner execution", type=int
    )
    parser.add_argument(
        "--vis",
        help="A flag enabling visualization of the pybullet execution",
        type=bool,
    )
    parser.add_argument(
        "--vis-graph",
        help="A flag enabling visualization of the learned transition graphs",
        type=bool,
    )
    parser.add_argument(
        "--print-options",
        help="Specifies what to print at each step of execution",
    )
    parser.add_argument(
        "--save-dir", help="File to load from", default="runs/run{}".format(str(time.time()))
    )
    parser.add_argument("--max-steps", help="Maximum number of steps allowed", type=int)

    parser.add_argument(
        "--batch-size", help="Number of samples from effect model before replanning.", type=int
    )
    parser.add_argument(
        "--num-skeletons", help="Number of symbolic skeletons to extract from symk", type=int
    )
    parser.add_argument(
        "--flat-sample",
        help="Sample all continuous controller input params once at the beginning.",
        type=bool,
    )
    parser.add_argument("--flat-width", help="Width when flat sampling", type=int)
    parser.add_argument("--pwa", help="Progressive widening alpha parameter", type=float)
    parser.add_argument("--pwk", help="Progressive widening k parameter", type=float)
    parser.add_argument("--gamma", help="POMDP decay parameter", type=float)
    parser.add_argument(
        "--envelope-threshold",
        help="Number of samples from effect model before replanning.",
        type=float,
    )
    parser.add_argument("--num-samples", help="Maximum number of steps allowed", type=int)
    parser.add_argument(
        "--learning-strategy", choices=["bayes_optimistic", "monte_carlo", "mdp_guided", "none"]
    )
    parser.add_argument("--decision-strategy", choices=["prob", "wao", "ao", "mlo", "none"])

    parser.add_argument("--symk-selection", choices=["unordered", "top_k"])

    parser.add_argument("--symk-direction", choices=["fw", "bw", "bd"])
    parser.add_argument("--symk-simple", type=bool)
    parser.add_argument("--from-scratch", type=bool)

    parser.add_argument(
        "--load",
        help="Location of the save folder to load from when visualizing",
    )
    return parser


class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def config_from_file(file_path) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_default_config(save_dir=None):
    return load_config(save_dir=save_dir)


def load_config(config_file="tampura/config/default.yml", save_dir=None, with_parser=False):
    arg_dict = {}

    if with_parser:
        parser = create_parser()
        arg_dict = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    if "config" in arg_dict:
        config_file = arg_dict["config"]

    script_directory = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir
    )
    config = config_from_file(os.path.join(script_directory, config_file))
    config.update(arg_dict)
    if "save_dir" not in config or config["save_dir"] is None:
        config["save_dir"] = "runs/run{}".format(time.time())

    if save_dir is not None:
        config["save_dir"] = os.path.join(save_dir, config["save_dir"])

    return config


def setup_logger(save_dir, log_level=logging.DEBUG):
    logger = logging.getLogger()

    # Reset logger if it has any handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Create a logs folder if it doesn't exist
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{time.time()}.log"))
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
