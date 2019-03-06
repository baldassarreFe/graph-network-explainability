#!/usr/bin/env bash

# Sync
rsync -a -v --delete-before --exclude '*.pyc' --exclude '.idea' --exclude '.ipynb_checkpoints' --exclude '.pytest_cache' --exclude '*__pycache__*' ~/Desktop/tg-experiments isengard:/home/fedbal/

# Conda environment
conda env export | sed '/prefix: .*/ d' > environment.yml
conda env create -f environment.yml
