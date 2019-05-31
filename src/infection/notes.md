# Infection

## Task
1. One node of the graph is _infected_ and identified through a 1 in one element of its feature vector. 
   The remaining features are unused. The infection is spread to the neighbors (directed) of the infected nodes.
   
2. One node of the graph is _infected_ and identified through a 1 in the first element of its feature vector. Some nodes
   are immune and identified through a 1 in the second element of their feature vectors. 
   Again, the infection is spread to the neighbors, but some of them are immune.
   
The network should output a prediction of 1 for nodes that are infected and 0 for the others, effectively
identifying the neighbors of the infected node, the connection type and the immunity status. 
The network also outputs a graph-level prediction that should correspond to the total number of infected nodes.

## Data

### Training
100,000 graphs are generated using the [Barabási–Albert model](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model), 
every graph contains between 10 and 30 nodes. Up to 10% of the nodes are sick and up to 30% are immune. Edges are virtual with a percentage of up to 30%.  

### Testing
20,000 graphs for testing are generated in a similar way, but containing between 10 and 60 nodes. 
Up to 40% of the nodes are sick and up to 60% are immune. Edges are virtual with a percentage of up to 50%.

## Losses

Three losses are used for training: a node-level loss, a global-level loss and a regularization term.
The three terms are added together in a weighted sum and constitute the final training objective.

### Node-level classification
At node-level, the network has to output a single binary prediction of whether the node will be sick or not. 
The loss on this prediction is computed as Binary Cross Entropy.

### Global-level regression
The network should also output a global-level prediction corresponding to the total number of infected nodes.
The loss on this prediction is computed as Mean Squared Error

Since the total number of infected nodes in the training set is not homogeneously distributed, we weight the losses
computed on individual graphs using the negative log frequency of the true value. For example, if 15 is the 
ground-truth number of infected nodes after the spread for a given input graph, we weight the MSE of the network's
prediction as `-ln(# of graphs with 15 infected nodes in the training set / # number of graphs in the training set)`

### L1 regularization
The weights of the network are regularized with L1 regularization. 

## Workflow

1. Create base folder
    ```bash
    INFECTION=~/experiments/infection/
    mkdir -p "$INFECTION/"{runs,data}
    ```
2. Create dataset (plus a small one for debug)
    ```bash
    python -m infection.dataset generate ../config/infection/datasets.yaml "folder=${INFECTION}/data"
    python -m infection.dataset generate ../config/infection/datasets.yaml \
           "folder=${INFECTION}/smalldata" \
           datasets.{train,val}.num_samples=5000
    ```
4. Launch one experiment:
    ```bash
    python -m infection.train \
       --experiment ../config/infection/train.yaml \
       --model ../config/infection/minimal.yaml
    ```
    Or make a grid search over the hyperparameters:
    ```bash
    conda activate tg-experiments
    function train {
       python -m infection.train \
               --experiment ../config/infection/train.yaml \
                            "tags=[${1},lr${2},nodes${4},count${5},wd${3}]" \
               --model "../config/infection/${1}.yaml" \
               --optimizer "kwargs.lr=${2}" \
               --session "losses.l1=${3}" "losses.nodes=${4}" "losses.count=${5}"  
    }    
    export -f train # use bash otherwise `export -f` won't work
    parallel --eta --max-procs 6 --load 80% --noswap 'train {1} {2} {3} {4} {5}' \
    `# Architecture`   ::: infectionGN \
    `# Learning rate`  ::: .01 .001 \
    `# L1 loss`        ::: 0   .0001 \
    `# Infection loss` ::: 1 .1 \
    `# Count loss`     ::: 1 .1
    ```
6. Visualize logs: 
   ```bash
   tensorboard --logdir "$INFECTION/runs"
   ```
