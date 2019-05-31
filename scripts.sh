#!/usr/bin/env bash

# Conda environment
conda env export | sed '/prefix: .*/ d' > conda.yaml
conda env create -n gn-exp -f conda.yaml
