import tqdm
import socket
import random
import textwrap
import multiprocessing
from pathlib import Path

import torch
import torch.utils.data
import torch.nn.functional as F
import arrow
import numpy as np
import pandas as pd
import sklearn.metrics
import torch_scatter
import torchgraphs as tg

from sacred import Experiment
from sacred.observers import RunObserver
from sacred.observers.file_storage import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from saver import Saver
from utils import load_class
from .dataset import InfectionDataset

# region Sacred experiment set up
ex = Experiment('infection')
ex.captured_out_filter = apply_backspaces_and_linefeeds


class RunIdHack(RunObserver):
    # The run id is assigned by the first observer that handles the started event,
    # this hack makes sure that the run id matches the session name from the config
    def __init__(self, run_id):
        self.run_id = run_id
        self.priority = 1_000_000

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        return self.run_id


# noinspection PyUnusedLocal
@ex.config
def config_autos():
    opts = dict(
        cpus=multiprocessing.cpu_count() - 1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_every='_never_',
        log_every='_never_'
    )
    opts = dict(
        session=f'{socket.gethostname()}_{arrow.utcnow()}'
    )


@ex.config_hook
def cfg_hook(config, command_name, logger):
    updates = {}
    paths = config['paths']
    if 'root' in paths:
        updates['root'] = Path(paths['root']).expanduser().resolve()
    if 'data' in paths:
        updates['data'] = Path(paths['data']).expanduser().resolve()
    else:
        updates['data'] = Path(paths['root']).joinpath('data').expanduser().resolve()
    if 'runs' in paths:
        updates['runs'] = Path(paths['runs']).expanduser().resolve()
    else:
        updates['runs'] = Path(paths['root']).joinpath('runs').expanduser().resolve()

    ex.observers.append(RunIdHack(config['opts']['session']))
    if command_name != 'print_config' and config['opts']['log_every'] != '_never_':
        ex.observers.append(FileStorageObserver.create(updates['runs']))
    return dict(paths={n: p.as_posix() for n, p in updates.items()})
# endregion


# region Setup and helper functions
def set_seeds(_seed):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.random.manual_seed(_seed)


def load_model_opt(model, optimizer, opts):
    model_class = load_class(model['klass'])
    model = model_class(**model['params']).to(opts['device'])
    opt_class = load_class(optimizer['klass'])
    optimizer = opt_class(params=model.parameters(), **optimizer['params'])
    return model, optimizer


def prepare_saver(opts, paths):
    if opts['save_every'] != '_never_':
        saver = Saver(paths['runs'])
        should_save = lambda epoch_idx: epoch_idx % opts['save_every'] == 0
    else:
        saver = None
        should_save = lambda _: False
    return saver, should_save


def prepare_logger(training, opts, paths, _run):
    if opts['log_every'] != '_never_':
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(Path(paths['runs']).joinpath(opts['session']).as_posix())
        mod = training['batch_size'] if opts['log_every'] == '_batch_' else int(opts['log_every'])
        should_log = lambda samples: samples % mod == 0
    else:
        logger = None
        should_log = lambda _: False
    return logger, should_log


def get_dataloader(name, paths, training, opts, shuffle):
    path = Path(paths['data']).joinpath(name).with_suffix('.pt')
    dataset = torch.load(path)
    if not isinstance(dataset, InfectionDataset):
        raise ValueError(f'Not an InfectionDataset: {path}')
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=training['batch_size'],
        collate_fn=tg.GraphBatch.collate,
        num_workers=opts['cpus'],
        shuffle=shuffle,
        pin_memory='cuda' in str(opts['device']),
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1))
    )
# endregion


@ex.automain
def train(model, optimizer, training, opts, paths, _run, _seed):
    set_seeds(_seed)
    opts['device'] = torch.device(opts['device'])
    paths = {n: Path(p) for n, p in paths.items()}
    model, optimizer = load_model_opt(model, optimizer, opts)
    saver, should_save = prepare_saver(opts, paths)
    logger, should_log = prepare_logger(training, opts, paths, _run)

    _run.info['epochs'] = 0
    _run.info['samples'] = 0

    dataloader_train = get_dataloader('train', paths, training, opts, shuffle=True)
    dataloader_val = get_dataloader('val', paths, training, opts, shuffle=False)

    epoch_bar_postfix = {}
    epoch_start = _run.info['epochs'] + 1
    epoch_end = _run.info['epochs'] + 1 + training['epochs']
    epoch_bar = tqdm.trange(epoch_start, epoch_end, desc='Training', unit='e', leave=True)
    for epoch in epoch_bar:
        # region Training loop
        model.train()
        losses_batch = 0
        with tqdm.tqdm(desc=f'Train {epoch}', total=len(dataloader_train.dataset), unit='g') as bar:
            for graphs, targets in dataloader_train:
                graphs = graphs.to(opts['device'])
                targets = targets.node_features.squeeze().to(opts['device'])

                results = model(graphs).node_features.squeeze()
                loss = F.binary_cross_entropy_with_logits(results, targets.float(), reduction='mean')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)

                _run.info['samples'] += graphs.num_graphs
                losses_batch += loss.item() * graphs.num_graphs

                bar.update(graphs.num_graphs)
                bar.set_postfix_str(f'Loss: {loss.item():.9f}')
                if should_log(_run.info['samples']):
                    logger.add_scalar('loss/train', loss.item(), global_step=_run.info['samples'])
                    _run.log_scalar('loss/train', loss.item(), step=_run.info['samples'])
        epoch_bar_postfix['Train'] = f'{losses_batch / len(dataloader_train.dataset):.4f}'
        epoch_bar.set_postfix(epoch_bar_postfix)
        # endregion

        # Saving
        _run.info['epochs'] += 1
        if should_save(_run.info['epochs']):
            saver.save(id_=epoch, model=model, optimizer=optimizer, **_run.info)

        # region Validation loop
        model.eval()
        losses_batch = 0
        with torch.no_grad():
            with tqdm.tqdm(desc=f'Val {epoch}', total=len(dataloader_val.dataset), unit='g') as bar:
                for graphs, targets in dataloader_val:
                    graphs = graphs.to(opts['device'])
                    targets = targets.node_features.squeeze().to(opts['device'])

                    results = model(graphs).node_features.squeeze()
                    loss = F.binary_cross_entropy_with_logits(results, targets.float(), reduction='mean')

                    losses_batch += loss.item() * graphs.num_graphs
                    bar.update(graphs.num_graphs)
                    bar.set_postfix_str(f'Loss: {loss.item():.9f}')
        if should_log(_run.info['samples']):
            logger.add_scalar('loss/val', losses_batch / len(dataloader_val.dataset), global_step=_run.info['samples'])
            _run.log_scalar('loss/val', losses_batch / len(dataloader_val.dataset), step=_run.info['samples'])
        epoch_bar_postfix['Val'] = f'{losses_batch / len(dataloader_val.dataset):.4f}'
        epoch_bar.set_postfix(epoch_bar_postfix)
        # endregion
    epoch_bar.close()

    dataset_train_max_nodes = dataloader_train.dataset.max_nodes
    dataset_train_min_nodes = dataloader_train.dataset.min_nodes
    del dataloader_train, dataloader_val, optimizer

    # region Testing
    model.eval()
    dataloader_test = get_dataloader('test', paths, training, opts, shuffle=False)
    stats_df = {k: [] for k in ['Loss', 'Nodes', 'Edges', 'Predict', 'Target', 'AvgPrecision', 'AreaROC']}
    huge_dict = {'targets': [], 'results': []}
    with torch.no_grad():
        with tqdm.tqdm(desc='Test', total=len(dataloader_test.dataset), leave=True, unit='g') as bar:
            for graphs, targets in dataloader_test:
                graphs = graphs.to(opts['device'])
                targets = targets.to(opts['device'])

                results = model(graphs)
                losses = F.binary_cross_entropy_with_logits(
                    results.node_features.squeeze(), targets.node_features.squeeze().float(), reduction='none')

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
    dataset_test_max_nodes = dataloader_test.dataset.max_nodes
    del dataloader_test
    # endregion

    # region Final report
    targets = torch.cat(huge_dict['targets']).int()  # numpy does not work with torch.int8
    results = torch.cat(huge_dict['results'])
    del huge_dict
    _run.info['average_precision'] = sklearn.metrics.average_precision_score(y_true=targets, y_score=results)
    print('Average precision:', _run.info['average_precision'])
    if False:
        import matplotlib.pyplot as plt
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=targets, probas_pred=results)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={_run.info["average_precision"]:.2f}')
        plt.show()
    if logger is not None:
        logger.add_scalar('test/avg_precision', _run.info['average_precision'], global_step=_run.info['samples'])
        logger.add_pr_curve('test/pr_curve', labels=targets, predictions=results, global_step=_run.info['samples'])
    del targets, results

    stats_df = pd.DataFrame({k: np.concatenate(v) for k, v in stats_df.items()}).rename_axis('GraphId').reset_index()

    # Split the results based on whether the number of nodes was present in the training set or not
    df_train_test = stats_df \
        .groupby(np.where(stats_df.Nodes < dataset_train_max_nodes,
                          f'Train [{dataset_train_min_nodes}, {dataset_train_max_nodes - 1})',
                          f'Test  [{dataset_train_max_nodes}, {dataset_test_max_nodes - 1})')) \
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
          df_train_test.to_string(float_format=lambda x: f'{x:.2f}'),
          df_losses_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
          df_best_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
          df_worst_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'),
          sep='\n\n')

    if logger is not None:
        logger.add_text(
            'Generalization',
            textwrap.indent(df_train_test.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
            global_step=ex.info['samples'])
        logger.add_text(
            'Loss by range',
            textwrap.indent(df_losses_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
            global_step=ex.info['samples'])
        logger.add_text(
            'Best predictions',
            textwrap.indent(df_best_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
            global_step=ex.info['samples'])
        logger.add_text(
            'Worst predictions',
            textwrap.indent(df_worst_graphs_by_node_range.to_string(float_format=lambda x: f'{x:.2f}'), '    '),
            global_step=ex.info['samples'])

    params = [f'{name}:\n{param.data.cpu().numpy().round(3)}' for name, param in model.named_parameters()]
    print('Parameters:', *params, sep='\n\n')
    if logger is not None:
        logger.add_text('Parameters', textwrap.indent('\n\n'.join(params), '    '), global_step=ex.info['samples'])
    # endregion

    if logger is not None:
        logger.close()

    return _run.info['average_precision']
