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
session.l1 = 0
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
            # If the yaml file contains a single entry with key `model` use that one instead
            if update.keys() == {'model'}:
                update = update['model']
    update_rec(experiment.model, update)

# Optimizer from --optimizer args
for string in args.optimizer:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml file contains a single entry with key `optimizer` use that one instead
            if update.keys() == {'optimizer'}:
                update = update['optimizer']
    update_rec(experiment.optimizer, update)

# Session from --session args
for string in args.session:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml file contains a single entry with key `session` use that one instead
            if update.keys() == {'session'}:
                update = update['session']
    update_rec(session, update)

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
if session.l1 < 0:
    raise ValueError(f'Invalid factor for L1: {session.l1}')
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
sort_dict(session, ['epochs', 'batch_size', 'l1', 'seed', 'cpus', 'device', 'samples', 'status',
                    'datetime_started', 'datetime_completed', 'data', 'log', 'checkpoint', 'git', 'gpus'])
experiment.sessions.append(session)
pyaml.pprint(experiment, sort_dicts=False, width=200)
# endregion

# region Building phase
# Seeds (set them after the random run id is generated)
set_seeds(experiment.session.seed)

# Model
model: torch.nn.Module = import_(experiment.model.fn)(*experiment.model.args, **experiment.model.kwargs)
if 'state_dict' in experiment.model:
    model.load_state_dict(torch.load(experiment.model.state_dict))
model.to(session.device)

# Optimizer
optimizer: torch.optim.Optimizer = import_(experiment.optimizer.fn)(
    model.parameters(), *experiment.optimizer.args, **experiment.optimizer.kwargs)
if 'state_dict' in experiment.optimizer:
    optimizer.load_state_dict(torch.load(experiment.optimizer.state_dict))

# Logger
if len(experiment.session.log.when) > 0:
    logger = SummaryWriter(experiment.session.log.folder)
    logger.add_text(
        'experiment', textwrap.indent(pyaml.dump(experiment, safe=True, sort_dicts=False), '    '), experiment.samples)
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

# region Training
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

# Train and validation loops
experiment.session.status = 'RUNNING'
experiment.session.datetime_started = datetime.utcnow()

epoch_bar_postfix = {}
epoch_bar = tqdm.trange(1, experiment.session.epochs + 1, desc='Epochs', unit='e', leave=True)
for epoch_idx in epoch_bar:
    experiment.epoch += 1

    # region Training loop
    model.train()
    train_bar_postfix={}
    loss_bce_avg = RunningWeightedAverage()
    with tqdm.tqdm(desc=f'Train {experiment.epoch}', total=len(dataloader_train.dataset), unit='g') as bar:
        for graphs, targets in dataloader_train:
            graphs = graphs.to(experiment.session.device)
            targets = targets.node_features.squeeze().to(experiment.session.device)

            results = model(graphs).node_features.squeeze()
            loss_bce = F.binary_cross_entropy_with_logits(results, targets.float(), reduction='mean')
            loss_total = loss_bce
            if experiment.session.l1 > 0:
                loss_l1 = experiment.session.l1 * sum([p.abs().sum() for p in model.parameters()])
                loss_total += loss_l1
                train_bar_postfix['L1'] = f'{loss_l1.item():.5f}'
                if 'every batch' in experiment.session.log.when:
                    logger.add_scalar('loss/train/l1', loss_l1.item(), global_step=experiment.samples)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step(closure=None)

            experiment.samples += len(graphs)
            loss_bce_avg.add(loss_bce.item(), len(graphs))

            bar.update(len(graphs))
            train_bar_postfix['BCE'] = f'{loss_bce.item():.5f}'
            bar.set_postfix(train_bar_postfix)
            if 'every batch' in experiment.session.log.when:
                logger.add_scalar('loss/train/bce', loss_bce.item(), global_step=experiment.samples)
                logger.add_scalar('loss/train/total', loss_total.item(), global_step=experiment.samples)
    epoch_bar_postfix['Train'] = f'{loss_bce_avg.get():.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)
    # endregion

    # region Validation loop
    model.eval()
    loss_avg = RunningWeightedAverage()
    stats_df = {k: [] for k in ['Loss', 'Nodes', 'Edges', 'Predict', 'Target', 'AvgPrecision', 'AreaROC']}
    huge_dict = {'targets': [], 'results': []}
    with torch.no_grad():
        with tqdm.tqdm(desc=f'Val {experiment.epoch}', total=len(dataloader_val.dataset), unit='g') as bar:
            for graphs, targets in dataloader_val:
                graphs = graphs.to(experiment.session.device)
                targets = targets.to(experiment.session.device)

                results = model(graphs)
                losses = F.binary_cross_entropy_with_logits(
                    results.node_features.squeeze(), targets.node_features.squeeze().float(), reduction='none')
                loss_avg.add(losses.mean().item(), len(graphs))

                idx = tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph)
                losses_by_graph = torch_scatter.scatter_mean(
                    losses, index=idx, dim=0, dim_size=graphs.num_graphs)
                infected_by_graph_true = torch_scatter.scatter_add(
                    targets.node_features.squeeze(), index=idx, dim=0, dim_size=graphs.num_graphs)
                infected_by_graph_pred = torch_scatter.scatter_add(
                    results.node_features.squeeze().sigmoid(), index=idx, dim=0, dim_size=graphs.num_graphs)
                avg_prec_by_graph = []
                area_roc_by_graph = []
                for t, r in zip(targets.node_features_by_graph, results.node_features_by_graph):
                    # numpy does not work with torch.int8
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

                huge_dict['targets'].append(targets.node_features.squeeze().cpu())
                huge_dict['results'].append(results.node_features.squeeze().sigmoid().cpu())

                stats_df['Loss'].append(losses_by_graph.cpu())
                stats_df['Nodes'].append(graphs.num_nodes_by_graph.cpu())
                stats_df['Edges'].append(graphs.num_edges_by_graph.cpu())
                stats_df['Target'].append(infected_by_graph_true.cpu().int())  # numpy does not work with torch.int8
                stats_df['Predict'].append(infected_by_graph_pred.cpu())
                stats_df['AvgPrecision'].append(np.array(avg_prec_by_graph))
                stats_df['AreaROC'].append(np.array(area_roc_by_graph))

                bar.update(graphs.num_graphs)
                bar.set_postfix_str(f'Loss: {losses.mean().item():.9f}')

    logger.add_scalar('loss/val/bce', loss_avg.get(), global_step=experiment.samples)
    epoch_bar_postfix['Val'] = f'{loss_avg.get():.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)
    # endregion

    # Saving
    if epoch_idx == experiment.session.epochs:
        experiment.session.status = 'DONE'
        experiment.session.datetime_completed = datetime.utcnow()
    if (
            'every epoch' in experiment.session.checkpoint.when or
            'last epoch' in experiment.session.checkpoint.when and epoch_idx == experiment.session.epochs
    ):
        saver.save(model, experiment, optimizer, suffix=f'e{experiment.epoch:04d}')
epoch_bar.close()
# endregion

# region Final report
targets = torch.cat(huge_dict['targets']).int()  # numpy does not work with torch.int8
results = torch.cat(huge_dict['results'])
del huge_dict

experiment.average_precision = sklearn.metrics.average_precision_score(y_true=targets, y_score=results)
print('Average precision:', experiment.average_precision)
if False:
    import matplotlib.pyplot as plt
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=targets, probas_pred=results)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={experiment.average_precision:.2f}')
    plt.show()
if logger is not None:
    logger.add_scalar('val/avg_precision', experiment.average_precision, global_step=experiment.samples)
    logger.add_pr_curve('val/pr_curve', labels=targets, predictions=results, global_step=experiment.samples)
del targets, results

stats_df = pd.DataFrame({k: np.concatenate(v) for k, v in stats_df.items()}).rename_axis('GraphId').reset_index()

# Split the results based on whether the number of nodes was present in the training set or not
df_train_val = stats_df \
    .groupby(np.where(stats_df.Nodes < dataset_train.max_nodes,
                      f'Train [{dataset_train.min_nodes}, {dataset_train.max_nodes - 1})',
                      f'Val  [{dataset_train.max_nodes}, {dataset_val.max_nodes - 1})')) \
    .agg({'Nodes': ['min', 'max'], 'GraphId': 'count', 'Loss': 'mean'}) \
    .sort_index(ascending=False) \
    .rename_axis(index='Dataset') \
    .rename(str.capitalize, axis='columns', level=1)

# Split the results in ranges based on the number of nodes and compute the average loss per range
df_losses_by_node_range = stats_df \
    .groupby(stats_df.Nodes // 10) \
    .agg({'Nodes': ['min', 'max'], 'GraphId': 'count', 'Loss': 'mean'}) \
    .rename_axis(index='NodeRange') \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index') \
    .rename(str.capitalize, axis='columns', level=1)

# Split the results in ranges based on the number of nodes and keep the N worst predictions
df_worst_graphs_by_node_range = stats_df \
    .groupby(stats_df.Nodes // 10) \
    .apply(lambda df_gr: df_gr.nlargest(5, 'Loss').set_index('GraphId')) \
    .rename_axis(index={'Nodes': 'NodeRange'}) \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index', level=0)

# Split the results in ranges based on the number of nodes and keep the N best predictions
df_best_graphs_by_node_range = stats_df \
    .groupby(stats_df.Nodes // 10) \
    .apply(lambda df_gr: df_gr.nsmallest(3, 'Loss').set_index('GraphId')) \
    .rename_axis(index={'Nodes': 'NodeRange'}) \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index', level=0)

print('',
      df_train_val.to_string(float_format=lambda x: f'{x:.2f}'),
      df_losses_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
      df_best_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
      df_worst_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
      sep='\n\n')

if logger is not None:
    logger.add_text(
        'Generalization',
        textwrap.indent(df_train_val.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
        global_step=experiment.samples)
    logger.add_text(
        'Loss by range',
        textwrap.indent(df_losses_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
        global_step=experiment.samples)
    logger.add_text(
        'Best predictions',
        textwrap.indent(df_best_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
        global_step=experiment.samples)
    logger.add_text(
        'Worst predictions',
        textwrap.indent(df_worst_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
        global_step=experiment.samples)

params = [f'{name}:\n{param.data.cpu().numpy().round(3)}' for name, param in model.named_parameters()]
print('Parameters:', *params, sep='\n\n')
if logger is not None:
    logger.add_text('Parameters', textwrap.indent('\n\n'.join(params), '    '), global_step=experiment.samples)
# endregion

# region Cleanup
if logger is not None:
    logger.close()
exit()
# endregion
