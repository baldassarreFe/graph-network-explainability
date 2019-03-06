# Count Nodes

## Task
Count the number of nodes in the graph. It's a case of simple aggregation from nodes to global.

Cases:
1. Node features are all random, just count the number of nodes
2. Node features are a combination of informative features and random features. Only some of them are considered
   active and therefore counted. Specifically, if values of the informative features of a node are **all** 1, 
   that node should be counted, otherwise is skipped.  

## Generalization strategy
During training the network only sees _small_ graphs, i.e. graphs whose node count is below a threshold.
During testing the network is presented larger graphs, with a number of nodes outside the range used for training.

## Weighting function
Weighting is only necessary for case 2: 
- The number of nodes is selected at random as `n~Uniform(0, max_nodes)`
- Out of `K` features, `I` are considered independently informative and set to 0 or 1 with prob 0.5 
- The `I` informative features are evaluated through a logical AND, i.e. all must be 1 for the node to be counted
- For this reason, the target values for the dataset will follow a binomial distribution 
  
  ```t~Binomial(max_nodes, 0.5^I)```
  
  with mean `max_nodes * 0.5^I` and variance `max_nodes * 0.5^I (1 - 0.5^I)` 
- Graphs that have a very small or very large number or active nodes are rare and should be weighted more.
- A suitable weighting function is the negative log-likelihood of the target value:
  
  ```- ln Binomial(target | max_nodes, 0.5^I)``` 

## Full network
The full version of the graph network operates as such:
```
e_{s \to r}^{t+1} = ReLU [ f_e(e_{s \to r}^t) + f_s(n_s^t) + f_r(n_r^t) + f_u(u^t)]

n_i^{t+1} = ReLU [ g_n(n_i^t) + g_{in}(agg_s(e_{s \to i}^{t+1})) + g_{out}(agg_r(e_{i \to r}^{t+1})) + g_u(u^t)]

u_i^{t+1} = h_n(agg_i(n_i^{t+1})) + h_e(agg_{ij}(e_{i \to j}^{t+1})) + h_u(u^t)
```

using summation as an aggregation function

## Minimal network
For this task, the minimal version of the network should only use:
```
n_i^{t+1} = ReLU [ g_n(n_i^t) ]

u_i^{t+1} = h_n(agg_i(n_i^{t+1}))
```

## Workflow

1. Create base folder
    ```bash
    COUNT_NODES=~/experiments/count-nodes/
    mkdir -p "$COUNT_NODES/{runs,data}"
    ```
2. Create dataset
    ```bash
    python -m count_nodes.dataset --yaml <dataset.yml>
    ```
3. Create tensorboard layout
    ```bash
    python -m count_nodes.layout --folder "$COUNT_NODES/runs"
    ```
4. Launch experiments
    ```bash
    python -m count_nodes.train --yaml <model.yml> <train.yml> -- [<other config>]
    
    conda activate tg-experiments
    for i in 1 2 3 4 5; do
       for type in full minimal; do
           for lr in .01 .001; do
               for wd in 0 .01 .001; do
                   python -m count_nodes.train --yaml ../config/count_nodes/{train,${type}}.yml -- \
                     opts.session=${type}_lr${lr}_wd${wd}_${i} \
                     optimizer.lr=${lr} \
                     training.l1=${wd} training.epochs=40
               done
           done
       done
    done    
    ```
5. Visualize   
    ```bash
    tensorboard --logdir "$COUNT_NODES/runs"
    ```