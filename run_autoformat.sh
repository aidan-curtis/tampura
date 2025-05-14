#!/bin/bash

# Exclude files in third_party folder for Black
python -m black . --exclude tampura/third_party/

# Exclude files in third_party folder for isort
isort . --skip tampura/third_party

# Exclude files in third_party folder for docformatter
docformatter -i -r . --exclude venv --exclude tampura/third_party
