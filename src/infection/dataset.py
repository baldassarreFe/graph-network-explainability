import numpy as np
import networkx as nx
from pathlib import Path

import torch
from torch.utils import data

import torchgraphs as tg


class InfectionDataset(data.Dataset):
    def __init__(self, max_percent_immune, max_percent_sick, min_nodes, max_nodes, num_samples):
        if max_percent_sick + max_percent_immune > 1:
            raise ValueError(f"Cannot have a population with `max_percent_sick`={max_percent_sick}"
                             f"and `max_percent_immune`={max_percent_immune}")
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_percent_immune = max_percent_immune
        self.max_percent_sick = max_percent_sick
        self.node_features_shape = 4
        self.edge_features_shape = 2
        self.samples = [None] * self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        self.samples[item] = self.samples[item] if self.samples[item] is not None else self._random_sample()
        return self.samples[item]

    def _random_sample(self):
        num_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        g_nx = nx.barabasi_albert_graph(num_nodes, 2).to_directed()

        # At least one sick
        num_sick = np.random.randint(1, max(1, int(num_nodes * self.max_percent_sick)) + 1)
        num_immune = np.random.randint(0, int(num_nodes * self.max_percent_immune) + 1)
        sick, immune, atrisk = np.split(g_nx.nodes, [num_sick, num_sick + num_immune])

        node_features = torch.zeros(num_nodes, self.node_features_shape)
        node_features[sick, 0] = 1
        node_features[immune, 1] = 1
        node_features[:, 2:].uniform_()

        edge_features = torch.rand(g_nx.number_of_edges(), self.edge_features_shape)

        g = tg.Graph.from_networkx(g_nx).evolve(node_features=node_features, edge_features=edge_features)

        target = node_features.new_zeros(size=(num_nodes, 1))
        target[sick] = 1
        target[list({n for s in sick for n in g_nx.neighbors(s) if n not in immune})] = 1
        target = tg.Graph(node_features=target)

        return g, target


def create_and_save():
    import random
    import argparse
    from tqdm import tqdm
    from config_manager import Config

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
        dataset = InfectionDataset(
            **datasets_common,
            **{k: v for k, v in config.datasets.get(f'_{name}_', {}).items()}
        )
        samples = len([s for s in tqdm(dataset, desc=name.capitalize(), unit='samples', leave=True)])
        torch.save(dataset, path)
        tqdm.write(f'{name.capitalize()}: saved {samples} samples in\t{path}')

if __name__ == '__main__':
    create_and_save()
