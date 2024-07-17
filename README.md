# TAMPURA

Partially Observable Task and motion planning with uncertainty and risk awareness. See our [paper](https://img.shields.io/badge/-Paper-87CEEB?style=flat&logo=read-the-docs) or [website](https://aidan-curtis.github.io/tampura.github.io/) for more details.

![alt text](figs/tasks.png)

## Install

Install the symbolic planner
```
    cd third_party/symk
    python build.py
```

Install the tampura package
```
    python -m pip install -e .
```

# Example Notebook

See `notebooks/grasping_env.ipynb` for a simple usage example.

# Robot environments

The robot environments from the paper are in a separate [tampura_environments](https://github.com/aidan-curtis/tampura_environments) repo
