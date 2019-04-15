# Solubility

## Task

1. Create base folder
    ```bash
    SOLUBILITY=~/experiments/solubility/
    mkdir -p "$SOLUBILITY/"{runs,data}
    ```

2. Create tensorboard layout
    ```bash
    python -m solubility.layout --folder "$SOLUBILITY/runs"
    ```
    
3. Launch experiments (use bash)
    ```bash
    python -m solubility.train \
       --experiment ../config/solubility/train.yaml \
       --model ../config/solubility/minimal.yaml
       
    conda activate tg-experiments
    function train {
       python -m solubility.train \
               --experiment ../config/solubility/train.yaml \
                            "tags=[layers${3},lr${1},bias${4},size${5},wd${2},dr${7},e${6},${8}]" \
               --model "kwargs.num_layers=${3}" "kwargs.hidden_bias=${4}" "kwargs.hidden_node=${5}" "kwargs.dropout=${6}" "kwargs.aggregation=${8}"\
               --optimizer "kwargs.lr=${1}" \
               --session "losses.l1=${2}" "epochs=${6}"
    }    
    export -f train
    parallel --verbose --max-procs 6 --load 200% --delay 1 --noswap \
    'export CUDA_VISIBLE_DEVICES=$(( ({#} - 1) % 6 )) && train {1} {2} {3} {4} {5} {6} {7} {8}' \
    `# Learning rate`   ::: .01 \
    `# L1 loss`         ::: .0001 .001 \
    `# Hidden layers`   ::: 3 4 5\
    `# Hidden bias`     ::: yes no \
    `# Hidden node`     ::: 64 128 256 512 \
    `# Epochs`          ::: 50 \
    `# Dropout`         ::: yes \
    `# Aggregation`     ::: mean sum
    ```

6. Query logs and visualize
   - Tensorboard: `tensorboard --logdir "$SOLUBILITY/runs"`
