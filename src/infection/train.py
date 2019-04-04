import tqdm
import yaml
import pyaml
import random
import textwrap
import multiprocessing
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sklearn.metrics
from munch import AutoMunch

import torch
import torch.utils.data
import torch.nn.functional as F
import torch_scatter
import torchgraphs as tg
from tensorboardX import SummaryWriter

from saver import Saver
from utils import git_info, cuda_info, parse_dotted, update_rec, set_seeds, import_, sort_dict, RunningWeightedAverage
from .dataset import InfectionDataset

parser = ArgumentParser()
parser.add_argument('--experiment', nargs='+', required=True)
parser.add_argument('--model', nargs='+', required=False, default=[])
parser.add_argument('--optimizer', nargs='+', required=False, default=[])
parser.add_argument('--session', nargs='+', required=False, default=[])

args = parser.parse_args()


# region Collecting phase
class Experiment(AutoMunch):
    @property
    def session(self):
        return self.sessions[-1]


experiment = Experiment()

# Experiment defaults
experiment.name = 'experiment'
experiment.tags = []
experiment.samples = 0
experiment.model = {'fn': None, 'args': [], 'kwargs': {}}
experiment.optimizer = {'fn': None, 'args': [], 'kwargs': {}}
experiment.sessions = []

# Session defaults
session = AutoMunch()
session.losses = {'nodes': 0, 'count': 0, 'l1': 0}
session.seed = random.randint(0, 99)
session.cpus = multiprocessing.cpu_count() - 1
session.device = 'cuda' if torch.cuda.is_available() else 'cpu'
session.log = {'when': []}
session.checkpoint = {'when': []}

# Experiment configuration
for string in args.experiment:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
    # If the current session is defined inside the experiment update the session instead
    if 'session' in update:
        update_rec(session, update.pop('session'))
    update_rec(experiment, update)

# Model from --model args
for string in args.model:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml document contains a single entry with key `model` use that one instead
            if update.keys() == {'model'}:
                update = update['model']
    update_rec(experiment.model, update)
    del update

# Optimizer from --optimizer args
for string in args.optimizer:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml document contains a single entry with key `optimizer` use that one instead
            if update.keys() == {'optimizer'}:
                update = update['optimizer']
    update_rec(experiment.optimizer, update)
    del update

# Session from --session args
for string in args.session:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml document contains a single entry with key `session` use that one instead
            if update.keys() == {'session'}:
                update = update['session']
    update_rec(session, update)
    del update

# Checks (some missing, others redundant)
if experiment.name is None or len(experiment.name) == 0:
    raise ValueError(f'Experiment name is empty: {experiment.name}')
if experiment.tags is None:
    raise ValueError('Experiment tags is None')
if experiment.model.fn is None:
    raise ValueError('Model constructor function not defined')
if experiment.optimizer.fn is None:
    raise ValueError('Optimizer constructor function not defined')
if session.cpus < 0:
    raise ValueError(f'Invalid number of cpus: {session.cpus}')
if any(l < 0 for l in session.losses.values()) or all(l == 0 for l in session.losses.values()):
    raise ValueError(f'Invalid losses: {session.losses}')
if len(experiment.sessions) > 0 and ('state_dict' not in experiment.model or 'state_dict' not in experiment.optimizer):
    raise ValueError(f'Model and optimizer state dicts are required to restore training')

# Experiment computed fields
experiment.epoch = sum((s.epochs for s in experiment.sessions), 0)

# Session computed fields
session.status = 'NEW'
session.datetime_started = None
session.datetime_completed = None
git = git_info()
if git is not None:
    session.git = git
if 'cuda' in session.device:
    session.cuda = cuda_info()

# Resolving paths
rand_id = ''.join(chr(random.randint(ord('A'), ord('Z'))) for _ in range(6))
session.data.folder = Path(session.data.folder.replace('{name}', experiment.name)).expanduser().resolve().as_posix()
session.log.folder = session.log.folder \
    .replace('{name}', experiment.name) \
    .replace('{tags}', '_'.join(experiment.tags)) \
    .replace('{rand}', rand_id)
if len(session.checkpoint.when) > 0:
    if len(session.log.when) > 0:
        session.log.folder = Path(session.log.folder).expanduser().resolve().as_posix()
    session.checkpoint.folder = session.checkpoint.folder \
        .replace('{name}', experiment.name) \
        .replace('{tags}', '_'.join(experiment.tags)) \
        .replace('{rand}', rand_id)
    session.checkpoint.folder = Path(session.checkpoint.folder).expanduser().resolve().as_posix()
if 'state_dict' in experiment.model:
    experiment.model.state_dict = Path(experiment.model.state_dict).expanduser().resolve().as_posix()
if 'state_dict' in experiment.optimizer:
    experiment.optimizer.state_dict = Path(experiment.optimizer.state_dict).expanduser().resolve().as_posix()

sort_dict(experiment, ['name', 'tags', 'epoch', 'samples', 'model', 'optimizer', 'sessions'])
sort_dict(session, ['epochs', 'batch_size', 'losses', 'seed', 'cpus', 'device', 'samples', 'status',
                    'datetime_started', 'datetime_completed', 'data', 'log', 'checkpoint', 'git', 'gpus'])
experiment.sessions.append(session)
pyaml.pprint(experiment, sort_dicts=False, width=200)
del session
# endregion

# region Building phase
# Seeds (set them after the random run id is generated)
set_seeds(experiment.session.seed)

# Model
model: torch.nn.Module = import_(experiment.model.fn)(*experiment.model.args, **experiment.model.kwargs)
if 'state_dict' in experiment.model:
    model.load_state_dict(torch.load(experiment.model.state_dict))
model.to(experiment.session.device)

# Optimizer
optimizer: torch.optim.Optimizer = import_(experiment.optimizer.fn)(
    model.parameters(), *experiment.optimizer.args, **experiment.optimizer.kwargs)
if 'state_dict' in experiment.optimizer:
    optimizer.load_state_dict(torch.load(experiment.optimizer.state_dict))

# Logger
if len(experiment.session.log.when) > 0:
    logger = SummaryWriter(experiment.session.log.folder)
    logger.add_text(
        'Experiment', textwrap.indent(pyaml.dump(experiment, safe=True, sort_dicts=False), '    '), experiment.samples)
else:
    logger = None

# Saver
if len(experiment.session.checkpoint.when) > 0:
    saver = Saver(experiment.session.checkpoint.folder)
    if experiment.epoch == 0:
        saver.save_experiment(experiment, suffix=f'e{experiment.epoch:04d}')
else:
    saver = None
# endregion

# Datasets and dataloaders
dataloader_kwargs = dict(
    num_workers=min(experiment.session.cpus, 1) if 'cuda' in experiment.session.device else experiment.session.cpus,
    pin_memory='cuda' in experiment.session.device,
    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)),
    batch_size=experiment.session.batch_size,
    collate_fn=tg.GraphBatch.collate,
)
dataset_train: InfectionDataset = torch.load(Path(experiment.session.data.folder) / 'train.pt')
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    shuffle=True,
    **dataloader_kwargs
)
dataset_val: InfectionDataset = torch.load(Path(experiment.session.data.folder) / 'val.pt')
dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    shuffle=False,
    **dataloader_kwargs
)
del dataloader_kwargs

# region Training
# Train and validation loops
experiment.session.status = 'RUNNING'
experiment.session.datetime_started = datetime.utcnow()

graphs_df = {k: [] for k in ['LossInfection', 'LossCount', 'Nodes', 'Edges', 'InfectedStart', 'InfectedEnd',
                             'InfectedSum', 'InfectedCount', 'AvgPrecision', 'AreaROC']}
nodes_df = {k: [] for k in ['Targets', 'Results']}

epoch_bar_postfix = {}
epoch_bar = tqdm.trange(1, experiment.session.epochs + 1, desc='Epochs', unit='e', leave=True)
for epoch_idx in epoch_bar:
    experiment.epoch += 1

    # region Training loop
    model.train()
    torch.set_grad_enabled(True)
    
    train_bar_postfix = {}
    loss_bce_avg = RunningWeightedAverage()
    loss_count_avg = RunningWeightedAverage()
    loss_l1_avg = RunningWeightedAverage()
    loss_total_avg = RunningWeightedAverage()

    train_bar = tqdm.tqdm(desc=f'Train {experiment.epoch}', total=len(dataloader_train.dataset), unit='g') 
    for graphs, targets in dataloader_train:
        graphs = graphs.to(experiment.session.device)
        targets = targets.to(experiment.session.device)
        results = model(graphs)

        loss_total = torch.tensor(0., device=experiment.session.device)

        if experiment.session.losses.nodes > 0:
            loss_bce = F.binary_cross_entropy_with_logits(
                results.node_features.squeeze(), targets.node_features.squeeze().float(), reduction='mean')
            loss_total += experiment.session.losses.nodes * loss_bce
            loss_bce_avg.add(loss_bce.item(), len(graphs))
            train_bar_postfix['Infected'] = f'{loss_bce.item():.5f}'
            if 'every batch' in experiment.session.log.when:
                logger.add_scalar('loss/train/infected', loss_bce.item(), global_step=experiment.samples)

        if experiment.session.losses.count > 0:
            loss_count = F.mse_loss(
                results.global_features.squeeze(), targets.global_features.squeeze().float(), reduction='mean')
            loss_total += experiment.session.losses.count * loss_count
            loss_count_avg.add(loss_count.item(), len(graphs))
            train_bar_postfix['Count'] = f'{loss_count.item():.5f}'
            if 'every batch' in experiment.session.log.when:
                logger.add_scalar('loss/train/count', loss_count.item(), global_step=experiment.samples)

        if experiment.session.losses.l1 > 0:
            loss_l1 = sum([p.abs().sum() for p in model.parameters()])
            loss_total += experiment.session.losses.l1 * loss_l1
            loss_l1_avg.add(loss_l1.item(), len(graphs))
            train_bar_postfix['L1'] = f'{loss_l1.item():.5f}'
            if 'every batch' in experiment.session.log.when:
                logger.add_scalar('loss/train/l1', loss_l1.item(), global_step=experiment.samples)

        loss_total_avg.add(loss_total.item(), len(graphs))
        train_bar_postfix['Total'] = f'{loss_total.item():.5f}'
        if 'every batch' in experiment.session.log.when:
            logger.add_scalar('loss/train/total', loss_total.item(), global_step=experiment.samples)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step(closure=None)

        experiment.samples += len(graphs)
        train_bar.update(len(graphs))
        train_bar.set_postfix(train_bar_postfix)
    train_bar.close()

    epoch_bar_postfix['Train'] = f'{loss_total_avg.get():.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)

    if 'every epoch' in experiment.session.log.when and 'every batch' not in experiment.session.log.when:
        logger.add_scalar('loss/train/total', loss_total_avg.get(), global_step=experiment.samples)
        if experiment.session.losses.nodes > 0:
            logger.add_scalar('loss/train/infected', loss_bce_avg.get(), global_step=experiment.samples)
        if experiment.session.losses.count > 0:
            logger.add_scalar('loss/train/count', loss_count_avg.get(), global_step=experiment.samples)
        if experiment.session.losses.l1 > 0:
            logger.add_scalar('loss/train/l1', loss_l1_avg.get(), global_step=experiment.samples)

    del train_bar, train_bar_postfix, loss_bce_avg, loss_count_avg, loss_l1_avg, loss_total_avg
    # endregion

    # region Validation loop
    model.eval()
    torch.set_grad_enabled(False)
    
    val_bar_postfix = {}
    loss_bce_avg = RunningWeightedAverage()
    loss_count_avg = RunningWeightedAverage()
    loss_total_avg = RunningWeightedAverage()
    loss_l1 = sum([p.abs().sum() for p in model.parameters()])
    val_bar_postfix['L1'] = f'{loss_l1.item():.5f}'

    val_bar = tqdm.tqdm(desc=f'Val {experiment.epoch}', total=len(dataloader_val.dataset), unit='g')
    for batch_idx, (graphs, targets) in enumerate(dataloader_val):
        graphs = graphs.to(experiment.session.device)
        targets = targets.to(experiment.session.device)
        results = model(graphs)

        loss_total = torch.tensor(0., device=experiment.session.device)

        if experiment.session.losses.nodes > 0:
            loss_bce = F.binary_cross_entropy_with_logits(
                results.node_features.squeeze(), targets.node_features.squeeze().float(), reduction='mean')
            loss_total += experiment.session.losses.nodes * loss_bce
            loss_bce_avg.add(loss_bce.item(), len(graphs))
            val_bar_postfix['Infected'] = f'{loss_bce.item():.5f}'

        if experiment.session.losses.count > 0:
            loss_count = F.mse_loss(
                results.global_features.squeeze(), targets.global_features.squeeze().float(), reduction='mean')
            loss_total += experiment.session.losses.count * loss_count
            loss_count_avg.add(loss_count.item(), len(graphs))
            val_bar_postfix['Count'] = f'{loss_count.item():.5f}'

        if experiment.session.losses.l1 > 0:
            loss_total += experiment.session.losses.l1 * loss_l1

        val_bar_postfix['Total'] = f'{loss_total.item():.5f}'
        loss_total_avg.add(loss_total.item(), len(graphs))

        # region Last epoch
        if epoch_idx == experiment.session.epochs:
            loss_bce_by_graph = torch_scatter.scatter_mean(
                F.binary_cross_entropy_with_logits(
                    results.node_features.squeeze(), targets.node_features.squeeze().float(), reduction='none'),
                index=tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph),
                dim=0,
                dim_size=graphs.num_graphs
            )
            loss_count_by_graph = F.mse_loss(
                results.global_features.squeeze(), targets.global_features.squeeze().float(), reduction='none')
            infected_by_graph_start_true = torch_scatter.scatter_add(
                graphs.node_features[:, 0].int(),
                index=tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph),
                dim=0,
                dim_size=graphs.num_graphs
            )
            infected_by_graph_sum_pred = torch_scatter.scatter_add(
                results.node_features.squeeze().sigmoid(),
                index=tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph),
                dim=0,
                dim_size=graphs.num_graphs
            )
            avg_prec_by_graph = []
            area_roc_by_graph = []
            for t, r in zip(targets.node_features_by_graph, results.node_features_by_graph):
                avg_prec_by_graph.append(sklearn.metrics.average_precision_score(
                    y_true=t.squeeze().cpu().int(),  # numpy does not work with torch.int8
                    y_score=r.squeeze().sigmoid().cpu())
                )
                try:
                    area_roc_by_graph.append(sklearn.metrics.roc_auc_score(
                        y_true=t.squeeze().cpu().int(),  # numpy does not work with torch.int8
                        y_score=r.squeeze().sigmoid().cpu())
                    )
                except ValueError:
                    # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
                    area_roc_by_graph.append(np.nan)
    
            nodes_df['Targets'].append(targets.node_features.squeeze().cpu().int())  # numpy doesn't convert torch.int8
            nodes_df['Results'].append(results.node_features.squeeze().sigmoid().cpu())
    
            graphs_df['LossInfection'].append(loss_bce_by_graph.cpu())
            graphs_df['LossCount'].append(loss_count_by_graph.cpu())
            graphs_df['Nodes'].append(graphs.num_nodes_by_graph.cpu())
            graphs_df['Edges'].append(graphs.num_edges_by_graph.cpu())
            graphs_df['InfectedStart'].append(infected_by_graph_start_true.cpu())
            graphs_df['InfectedEnd'].append(targets.global_features.squeeze().cpu())
            graphs_df['InfectedSum'].append(infected_by_graph_sum_pred.cpu())
            graphs_df['InfectedCount'].append(results.global_features.squeeze().cpu())
            graphs_df['AvgPrecision'].append(np.array(avg_prec_by_graph))
            graphs_df['AreaROC'].append(np.array(area_roc_by_graph))
        # endregion

        val_bar.update(len(graphs))
        val_bar.set_postfix(val_bar_postfix)
    val_bar.close()

    epoch_bar_postfix['Val'] = f'{loss_total_avg.get():.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)

    if (
            'every batch' in experiment.session.log.when or
            'every epoch' in experiment.session.log.when or
            'last epoch' in experiment.session.checkpoint.when and epoch_idx == experiment.session.epochs
    ):
        logger.add_scalar('loss/val/total', loss_total_avg.get(), global_step=experiment.samples)
        if experiment.session.losses.nodes > 0:
            logger.add_scalar('loss/val/infected', loss_bce_avg.get(), global_step=experiment.samples)
        if experiment.session.losses.count > 0:
            logger.add_scalar('loss/val/count', loss_count_avg.get(), global_step=experiment.samples)
        if experiment.session.losses.l1 > 0:
            logger.add_scalar('loss/val/l1', loss_l1.item(), global_step=experiment.samples)

    del val_bar, val_bar_postfix, loss_bce_avg, loss_count_avg, loss_l1, loss_total_avg, batch_idx
    # endregion

    # Saving
    if epoch_idx == experiment.session.epochs:
        experiment.session.status = 'DONE'
        experiment.session.datetime_completed = datetime.utcnow()
    if (
            'every batch' in experiment.session.checkpoint.when or
            'every epoch' in experiment.session.checkpoint.when or
            'last epoch' in experiment.session.checkpoint.when and epoch_idx == experiment.session.epochs
    ):
        saver.save(model, experiment, optimizer, suffix=f'e{experiment.epoch:04d}')
epoch_bar.close()
print()
del epoch_bar, epoch_bar_postfix, epoch_idx
# endregion

# region Final report
pd.options.display.float_format = '{:.2f}'.format

nodes_df = pd.DataFrame({k: np.concatenate(v) for k, v in nodes_df.items()})
experiment.average_precision = sklearn.metrics.average_precision_score(
    y_true=nodes_df.Targets, y_score=nodes_df.Results)
print('Average precision:', experiment.average_precision)
if logger is not None:
    logger.add_scalar('metrics/val/avg_precision', experiment.average_precision, global_step=experiment.samples)
    logger.add_pr_curve('infection', labels=nodes_df.Targets.values,
                        predictions=nodes_df.Results.values, global_step=experiment.samples)
# noinspection PyUnreachableCode
if False:
    import matplotlib.pyplot as plt
    precision, recall, _ = sklearn.metrics.precision_recall_curve(
        y_true=nodes_df.Targets, probas_pred=nodes_df.Results)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={experiment.average_precision:.2f}')
    plt.show()
del nodes_df

graphs_df = pd.DataFrame({k: np.concatenate(v) for k, v in graphs_df.items()}).rename_axis('GraphId').reset_index()
experiment.loss_count = graphs_df.LossCount.mean()
experiment.loss_infection = graphs_df.LossInfection.mean()
print('Count MSE:', experiment.loss_count)
print('Infection BCe:', experiment.loss_infection)

# Split the results based on whether the number of nodes was present in the training set or not
df_train_val = graphs_df \
    .groupby(np.where(graphs_df.Nodes < dataset_train.max_nodes,
                      f'Train [{dataset_train.min_nodes}, {dataset_train.max_nodes - 1})',
                      f'Val   [{dataset_train.max_nodes}, {dataset_val.max_nodes - 1})')) \
    .agg({'Nodes': ['min', 'max'], 'GraphId': 'count', 'LossInfection': 'mean', 'LossCount': 'mean'}) \
    .sort_index(ascending=True) \
    .rename_axis(index='Dataset') \
    .rename(str.capitalize, axis='columns', level=1)

# Split the results in ranges based on the number of nodes and compute the average loss per range
df_losses_by_node_range = graphs_df \
    .groupby(graphs_df.Nodes // 10) \
    .agg({'Nodes': ['min', 'max'], 'GraphId': 'count', 'LossInfection': 'mean', 'LossCount': 'mean'}) \
    .rename_axis(index='NodeRange') \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index') \
    .rename(str.capitalize, axis='columns', level=1)

# Split the results in ranges based on the number of nodes and keep the N worst predictions w.r.t. node-wise loss
df_worst_infection_loss_by_node_range = graphs_df \
    .groupby(graphs_df.Nodes // 10) \
    .apply(lambda df_gr: df_gr.nlargest(5, 'LossInfection').set_index('GraphId')) \
    .rename_axis(index={'Nodes': 'NodeRange'}) \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index', level=0)

# Split the results in ranges based on the number of nodes and keep the N worst predictions w.r.t. graph-wise loss
df_worst_count_loss_by_node_range = graphs_df \
    .groupby(graphs_df.Nodes // 10) \
    .apply(lambda df_gr: df_gr.nlargest(5, 'LossCount').set_index('GraphId')) \
    .rename_axis(index={'Nodes': 'NodeRange'}) \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index', level=0)

print(f"""
Generalization:
{df_train_val}\n
Losses by range:
{df_losses_by_node_range}\n
Worst infection predictions:
{df_worst_infection_loss_by_node_range}\n
Worst count predictions:
{df_worst_count_loss_by_node_range}
""")

if logger is not None:
    logger.add_text(
        'Generalization',
        textwrap.indent(df_train_val.to_string(), '    '),
        global_step=experiment.samples)
    logger.add_text(
        'Losses by range',
        textwrap.indent(df_losses_by_node_range.to_string(), '    '),
        global_step=experiment.samples)
    logger.add_text(
        'Worst infection predictions',
        textwrap.indent(df_worst_infection_loss_by_node_range.to_string(), '    '),
        global_step=experiment.samples),
    logger.add_text(
        'Worst count predictions',
        textwrap.indent(df_worst_count_loss_by_node_range.to_string(), '    '),
        global_step=experiment.samples)
del graphs_df, df_losses_by_node_range, df_worst_infection_loss_by_node_range, df_worst_count_loss_by_node_range

params = [f'{name}:\n{param.data.cpu().numpy().round(3)}' for name, param in model.named_parameters()]
print('Parameters:', *params, sep='\n\n')
if logger is not None:
    logger.add_text('Parameters', textwrap.indent('\n\n'.join(params), '    '), global_step=experiment.samples)
del params
# endregion

# region Cleanup
if logger is not None:
    logger.close()
exit()
# endregion
