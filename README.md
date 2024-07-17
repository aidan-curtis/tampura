# TAMPURA

Partially Observable Task and motion planning with uncertainty and risk awareness.

[![Paper](https://img.shields.io/badge/-Paper-0000FF?style=flat&logo=read-the-docs)](https://arxiv.org/abs/2403.10454)
[![Website](https://img.shields.io/badge/-Website-00FF00?style=flat&logo=Internet-Explorer)](https://aidan-curtis.github.io/tampura.github.io/)

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
