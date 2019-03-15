from pathlib import Path
from typing import List, Tuple

import torch
import torch.utils.data
import numpy as np
import networkx as nx
import torchgraphs as tg


class InfectionDataset(torch.utils.data.Dataset):
    def __init__(self, max_percent_immune, max_percent_sick, min_nodes, max_nodes):
        if max_percent_sick + max_percent_immune > 1:
            raise ValueError(f"Cannot have a population with `max_percent_sick`={max_percent_sick}"
                             f"and `max_percent_immune`={max_percent_immune}")
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_percent_immune = max_percent_immune
        self.max_percent_sick = max_percent_sick
        self.node_features_shape = 4
        self.edge_features_shape = 2
        self.samples: List[Tuple[tg.Graph, tg.Graph]] = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def random_sample(self):
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

        target = torch.zeros((num_nodes, 1), dtype=torch.int8)
        target[sick] = 1
        target[list({n for s in sick for n in g_nx.neighbors(s) if n not in immune})] = 1
        target = tg.Graph(node_features=target)

        return g, target


def generate(cfg):
    from tqdm import trange
    from utils import set_seeds

    cfg.setdefault('seed', 0)
    set_seeds(cfg.seed)
    print(f'Random seed: {cfg.seed}')

    folder = Path(cfg.folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Saving datasets in: {folder}')

    with open(folder / 'datasets.yaml', 'w') as f:
        f.write(cfg.toYAML())
    for p, params in cfg.datasets.items():
        dataset = InfectionDataset(**{k: v for k, v in params.items() if k != 'num_samples'})
        dataset.samples = [dataset.random_sample() for _ in
                           trange(params.num_samples, desc=p.capitalize(), unit='samples', leave=True)]
        path = folder.joinpath(p).with_suffix('.pt')
        torch.save(dataset, path)
        print(f'{p.capitalize()}: saved {len(dataset)} samples in: {path}')


def describe(cfg):
    import pandas as pd
    target = Path(cfg.target).expanduser().resolve()
    if target.is_dir():
        paths = target.glob('*.pt')
    else:
        paths = [target]
    for p in paths:
        print(f"Loading dataset from: {p}")
        dataset = torch.load(p)
        if not isinstance(dataset, InfectionDataset):
            raise ValueError(f'Not an InfectionDataset: {p}')
        print(f"{p.with_suffix('').name.capitalize()} contains:\n"
              f"min_nodes: {dataset.min_nodes}\n"
              f"max_nodes: {dataset.max_nodes}\n"
              f"max_percent_immune: {dataset.max_percent_immune}\n"
              f"max_percent_sick: {dataset.max_percent_sick}\n"
              f"node_features_shape: {dataset.node_features_shape}\n"
              f"edge_features_shape: {dataset.edge_features_shape}\n"
              f"samples: {len(dataset)}")
        df = pd.DataFrame.from_records(
            {
                'num_nodes': g.num_nodes,
                'num_edges': g.num_edges,
                'degree': g.degree.float().mean().item(),
                'infected': g.node_features[:, 0].sum().item() / g.num_nodes,
                'immune': g.node_features[:, 1].sum().item() / g.num_nodes,
                'infected_post': t.node_features[:, 0].sum().item() / t.num_nodes,
            } for g, t in dataset.samples)
        print(f'\n{df.describe()}')


def main():
    from argparse import ArgumentParser
    from config import Config

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    sp_print = subparsers.add_parser('print', help='Print parsed configuration')
    sp_print.add_argument('config', nargs='*')
    sp_print.set_defaults(command=lambda c: print(c.toYAML()))

    sp_generate = subparsers.add_parser('generate', help='Generate new datasets')
    sp_generate.add_argument('config', nargs='*')
    sp_generate.set_defaults(command=generate)

    sp_describe = subparsers.add_parser('describe', help='Describe existing datasets')
    sp_describe.add_argument('config', nargs='*')
    sp_describe.set_defaults(command=describe)

    args = parser.parse_args()
    cfg = Config.build(*args.config)
    args.command(cfg)


if __name__ == '__main__':
    main()
