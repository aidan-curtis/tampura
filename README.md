# TAMPURA environments

Task and motion planning with uncertainty and risk awareness.

![alt text](figs/tasks.png)

## Install

Make sure to pull in submodules
```
    git submodule update --init --recursive
```

Install the symbolic planner
```
    cd third_party/symk
    python build.py
```

# Example Notebook

See notebooks/grasping_env.ipynb for a simple usage example.

# Robot environments

The robot environments from the paper are in a separate [tampura_environments](https://github.com/aidan-curtis/tampura_environments) repo