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


def load_config(config_file, arg_dict={}, save_dir=None):
    config = config_from_file(config_file)
    config.update(arg_dict)
    if "save_dir" not in config or config["save_dir"] is None:
        config["save_dir"] = "runs/run{}".format(time.time())

    if save_dir is not None:
        config["save_dir"] = os.path.join(save_dir, config["save_dir"])

    return config


def setup_logger(save_dir, log_level=logging.INFO):
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
