# Solubility

## Task
We want to predict the water solubility (in `log mol/L`) of the organic molecules from their molecular structure. 

## Data
The molecules in [this dataset](../../data/delaney-processed.csv) are loaded and parsed using RDKit.
Train/test split is 70/30.

## Losses

Two losses are used for training: a global-level loss and a regularization term.
The two terms are added together in a weighted sum and constitute the final training objective.

### Global-level regression
The network should also output a global-level prediction corresponding to the log solubility of the molecule.
The loss on this prediction is computed as Mean Squared Error

No weighting is used to account for more/less common values.

### L1 regularization
The weights of the network are regularized with L1 regularization.

## Workflow

1. Create base folder
    ```bash
    SOLUBILITY=~/experiments/solubility/
    mkdir -p "$SOLUBILITY/"{runs,data}
    ```
2. Launch one experiment (from the root of the repo):
    ```bash
    python -m solubility.train --experiment config/solubility/train.yaml
    ```
    Or make a grid search over the hyperparameters (from the root of the repo):
    ```bash
    conda activate tg-experiments
    function train {
       python -m solubility.train \
               --experiment config/solubility/train.yaml \
                            "tags=[layers${3},lr${1},bias${4},size${5},wd${2},dr${7},e${6},${8}]" \
               --model "kwargs.num_layers=${3}" "kwargs.hidden_bias=${4}" "kwargs.hidden_node=${5}" "kwargs.dropout=${6}" "kwargs.aggregation=${8}"\
               --optimizer "kwargs.lr=${1}" \
               --session "losses.l1=${2}" "epochs=${6}"
    }    
    export -f train # use bash otherwise `export -f` won't work
    parallel --verbose --max-procs 6 --load 200% --delay 1 --noswap \
    'train {1} {2} {3} {4} {5} {6} {7} {8}' \
    `# Learning rate`   ::: .01 .001 \
    `# L1 loss`         ::: .0001 .001 .01 \
    `# Hidden layers`   ::: 3 4 5 10 \
    `# Hidden bias`     ::: yes no \
    `# Hidden node`     ::: 32 64 128 256 512 \
    `# Epochs`          ::: 50 75 100 \
    `# Dropout`         ::: yes no \
    `# Aggregation`     ::: mean sum
    ```

6. Query logs and visualize
   - Tensorboard: `tensorboard --logdir "$SOLUBILITY/runs"`
   - Find best model
   ```bash
   for f in */experiment.latest.yaml; do
       echo -e $(grep loss_sol_val < $f) $(dirname $f)
   done | sort -k 3 -g -r | cut -f2,3 | tail -n 5
   ```
