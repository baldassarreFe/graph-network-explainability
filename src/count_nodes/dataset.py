import numpy as np
import networkx as nx

import torch
from torch.utils import data

import torchgraphs as tg


class NodeCountDataset(data.Dataset):
    def __init__(self, min_nodes, max_nodes, num_samples, informative_features,
                 edge_features_shape, node_features_shape, global_features_shape):
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.node_features_shape = node_features_shape
        self.edge_features_shape = edge_features_shape
        self.global_features_shape = global_features_shape
        self.informative_features = informative_features
        self.samples = [None] * self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        self.samples[item] = self.samples[item] if self.samples[item] is not None else self._random_sample()
        return self.samples[item]

    def _random_sample(self):
        num_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        num_edges = np.random.randint(0, num_nodes * (num_nodes - 1) + 1)
        g_nx = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
        g = tg.Graph.from_networkx(g_nx)
        g = g.evolve(
            node_features=torch.empty(num_nodes, self.node_features_shape).uniform_(-1, 1),
            edge_features=torch.empty(num_edges, self.edge_features_shape).uniform_(-1, 1),
            global_features=torch.empty(self.global_features_shape).uniform_(-1, 1)
        )

        if self.informative_features > 0:
            feats = np.random.rand(num_nodes, self.informative_features) > .5
            target = np.logical_and(*feats.transpose()).sum()
            g.node_features[:, :self.informative_features] = torch.from_numpy(feats.astype(np.float32)) * 2 - 1
        else:
            target = g.num_nodes

        return g, target


def create_and_save():
    import random
    import argparse
    from tqdm import tqdm
    from pathlib import Path
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

    print(config.toYAML())
    if args.dry_run:
        print('Dry run, exiting.')
        exit(0)
    del args, rest

    random.seed(config.opts.seed)
    np.random.seed(config.opts.seed)
    torch.random.manual_seed(config.opts.seed)

    folder = Path(config.opts.folder).expanduser().resolve() / 'data'
    folder.mkdir(parents=True, exist_ok=True)

    datasets_common = {k: v for k, v in config.datasets.items() if k not in {'_train_', '_val_', '_test_'}}
    for name in ['train', 'val', 'test']:
        path = folder / f'{name}.pt'
        dataset = NodeCountDataset(
            **datasets_common,
            **{k: v for k, v in config.datasets.get(f'_{name}_', {}).items()}
        )
        samples = len([s for s in tqdm(dataset, desc=name.capitalize(), unit='samples', leave=True)])
        torch.save(dataset, path)
        tqdm.write(f'{name.capitalize()}: saved {samples} samples in\t{path}')


if __name__ == '__main__':
    create_and_save()
