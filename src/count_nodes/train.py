import random
import argparse
import textwrap

import pandas as pd
from pathlib import Path

import tqdm
import numpy as np
import torchgraphs as tg
from munch import Munch
from scipy import stats

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tensorboardX import SummaryWriter

from count_nodes.dataset import NodeCountDataset
from saver import Saver
from utils import load_class
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', nargs='+', default=[])
parser.add_argument('--dry-run', action='store_true')
args, rest = parser.parse_known_args()

config = Config()
for y in args.yaml:
    config.update_from_yaml(y)
if len(rest) > 0:
    if rest[0] != '--':
        rest = ' '.join(rest)
        print(f"Error: additional config must be separated by '--', got:\n{rest}")
        exit(1)
    config.update_from_cli(' '.join(rest[1:]))

print('Config summary:', config.toYAML(), sep='\n')
if args.dry_run:
    print('Dry run, exiting.')
    exit(0)
del args, rest

random.seed(config.opts.seed)
np.random.seed(config.opts.seed)
torch.random.manual_seed(config.opts.seed)

folder_base = (Path(config.opts.folder)).expanduser().resolve()
folder_data = folder_base / 'data'
folder_run = folder_base / 'runs' / config.opts.session
saver = Saver(folder_run)
logger = SummaryWriter(folder_run.as_posix())

ModelClass = load_class(config.model._class_)
net: nn.Module = ModelClass(**{k: v for k, v in config.model.items() if k != '_class_'})
net.to(config.opts.device)

OptimizerClass = load_class(config.optimizer._class_)
optimizer: optim.Optimizer = OptimizerClass(params=net.parameters(),
                                            **{k: v for k, v in config.optimizer.items() if k != '_class_'})

if config.training.restore:
    train_state = saver.load(model=net, optimizer=optimizer, device=config.training.device)
else:
    train_state = Munch(epochs=0, samples=0)

if config.opts.log:
    with open(folder_run / 'config.yml', mode='w') as f:
        f.write(config.toYAML())
    logger.add_text(
        'Config',
        textwrap.indent(config.toYAML(), '    '),
        global_step=train_state.samples)


def make_dataloader(dataset, shuffle) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        collate_fn=tg.GraphBatch.collate,
        num_workers=config.opts.cpus,
        shuffle=shuffle,
        pin_memory='cuda' in str(config.opts.device),
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1))
    )


dataset_train: NodeCountDataset = torch.load(folder_data / 'train.pt')
dataset_val: NodeCountDataset = torch.load(folder_data / 'val.pt')
dataset_test: NodeCountDataset = torch.load(folder_data / 'test.pt')
dataloader_train = make_dataloader(dataset_train, shuffle=True)
dataloader_val = make_dataloader(dataset_val, shuffle=False)
dataloader_test = make_dataloader(dataset_test, shuffle=False)

if dataset_train.informative_features > 0:
    binom = stats.binom(n=dataset_train.max_nodes, p=.5 ** dataset_train.informative_features)


    def weight_fn(targets):
        return targets.new_tensor(- binom.logpmf(targets.cpu().numpy()))
else:
    def weight_fn(targets):
        return torch.ones_like(targets)

epoch_bar_postfix = {}
epoch_start = train_state.epochs + 1
epoch_end = train_state.epochs + 1 + config.training.epochs
epoch_bar = tqdm.trange(epoch_start, epoch_end, desc='Training', unit='e', leave=True)
for epoch in epoch_bar:
    # Training loop
    net.train()
    loss_mse_train = 0
    train_bar_postfix = {}
    with tqdm.tqdm(desc=f'Train {epoch}', total=dataloader_train.dataset.num_samples, unit='g') as train_bar:
        for graphs, targets in dataloader_train:
            graphs = graphs.to(config.opts.device)
            targets = targets.float().to(config.opts.device)
            weights = weight_fn(targets)

            loss_total = 0
            results = net(graphs).global_features.squeeze()
            losses_mse = F.mse_loss(results, targets, reduction='none')
            losses_mse_weighted = losses_mse * weights
            loss_total += losses_mse_weighted.mean()

            if config.training.l1 > 0:
                loss_l1 = sum([p.abs().sum() for p in net.parameters()]) * config.training.l1
                loss_total += loss_l1
                train_bar_postfix['L1'] = f'{loss_l1.item():.5f}'
                if config.opts.log:
                    logger.add_scalar('loss/train/l1', loss_l1.item(), global_step=train_state.samples)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step(closure=None)

            train_state.samples += graphs.num_graphs
            loss_mse_train += losses_mse.sum().item()

            train_bar.update(graphs.num_graphs)
            train_bar_postfix['MSE'] = f'{losses_mse.mean().item():.5f}'
            train_bar.set_postfix(train_bar_postfix)
            if config.opts.log:
                logger.add_scalar('loss/train/all', loss_total.item(), global_step=train_state.samples)
                logger.add_scalar('loss/train/mse', losses_mse.mean().item(), global_step=train_state.samples)
    epoch_bar_postfix['train/mse'] = f'{loss_mse_train / dataloader_train.dataset.num_samples:.5f}'
    epoch_bar.set_postfix(epoch_bar_postfix)

    # Saving
    train_state.epochs += 1
    if config.training.save_every > 0 and epoch % config.training.save_every == 0:
        saver.save(id_=epoch, model=net, optimizer=optimizer, **train_state)

    # Validation loop
    net.eval()
    loss_mse_val = 0
    with torch.no_grad():
        with tqdm.tqdm(desc=f'Val {epoch}', total=dataloader_val.dataset.num_samples, unit='g') as val_bar:
            for graphs, targets in dataloader_val:
                graphs = graphs.to(config.opts.device)
                targets = targets.float().to(config.opts.device)

                results = net(graphs).global_features.squeeze()
                losses_mse = F.mse_loss(results, targets, reduction='none')

                loss_mse_val += losses_mse.sum().item()
                val_bar.update(graphs.num_graphs)
                val_bar.set_postfix_str(f'MSE: {losses_mse.mean().item():.5f}')
    if config.opts.log:
        logger.add_scalar(
            'loss/val/mse', loss_mse_val / dataloader_val.dataset.num_samples, global_step=train_state.samples)
    epoch_bar_postfix['val/mse'] = f'{loss_mse_val / dataloader_val.dataset.num_samples:.5f}'
    epoch_bar.set_postfix(epoch_bar_postfix)
epoch_bar.close()

net.eval()
df = {k: [] for k in ['Loss', 'Nodes', 'Edges', 'Predict', 'Target']}
with torch.no_grad():
    with tqdm.tqdm(desc='Test', total=dataloader_test.dataset.num_samples, leave=True, unit='g') as test_bar:
        for graphs, targets in dataloader_test:
            graphs = graphs.to(config.opts.device)
            targets = targets.float().to(config.opts.device)

            results = net(graphs).global_features.squeeze()
            losses_mse = F.mse_loss(results, targets, reduction='none')

            df['Loss'].append(losses_mse.cpu().numpy())
            df['Nodes'].append(graphs.num_nodes_by_graph.cpu().numpy())
            df['Edges'].append(graphs.num_edges_by_graph.cpu().numpy())
            df['Target'].append(targets.int().cpu().numpy())
            df['Predict'].append(results.cpu().numpy())

            test_bar.update(graphs.num_graphs)
            test_bar.set_postfix_str(f'MSE: {losses_mse.mean().item():.5f}')
df = pd.DataFrame({k: np.concatenate(v) for k, v in df.items()}).rename_axis('GraphId').reset_index()

# Split the results based on whether the number of nodes was present in the training set or not
df_train_test = df \
    .groupby(np.where(df.Nodes < dataset_train.max_nodes,
                      f'Train [{dataset_train.min_nodes}, {dataset_train.max_nodes - 1})',
                      f'Test  [{dataset_train.max_nodes}, {dataset_test.max_nodes - 1})')) \
    .agg({'Nodes': ['min', 'max'], 'GraphId': 'count', 'Loss': 'mean'}) \
    .sort_index(ascending=False) \
    .rename_axis(index='Dataset') \
    .rename(str.capitalize, axis='columns', level=1)

# Split the results in ranges based on the number of nodes and compute the average loss per range
df_losses_by_node_range = df \
    .groupby(df.Nodes // 10) \
    .agg({'Nodes': ['min', 'max'], 'GraphId': 'count', 'Loss': 'mean'}) \
    .rename_axis(index='NodeRange') \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index') \
    .rename(str.capitalize, axis='columns', level=1)

# Split the results in ranges based on the number of nodes and compute the average loss per range
df_worst_graphs_by_node_range = df \
    .groupby(df.Nodes // 10) \
    .apply(lambda df_gr: df_gr.nlargest(5, 'Loss').set_index('GraphId')) \
    .rename_axis(index={'Nodes': 'NodeRange'}) \
    .rename(lambda node_group_min: f'[{node_group_min * 10}, {node_group_min * 10 + 10})', axis='index', level=0)

print(
    df_train_test.to_string(float_format=lambda x: f'{x:.2f}'),
    df_losses_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
    df_worst_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
    sep='\n\n')
if config.opts.log:
    logger.add_text(
        'Generalization',
        textwrap.indent(df_train_test.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
        global_step=train_state.samples)
    logger.add_text(
        'Loss by range',
        textwrap.indent(df_losses_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
        global_step=train_state.samples)
    logger.add_text(
        'Samples',
        textwrap.indent(df_worst_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
        global_step=train_state.samples)

params = [f'{name}:\n{param.data.cpu().numpy().round(3)}' for name, param in net.named_parameters()]
print('Parameters:', *params, sep='\n\n')
if config.opts.log:
    logger.add_text('Parameters', textwrap.indent('\n\n'.join(params), '    '), global_step=train_state.samples)

logger.close()
