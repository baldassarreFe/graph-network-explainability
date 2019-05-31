import math
from collections import OrderedDict

import torch
from torch import nn

import torchgraphs as tg


class InfectionGN(nn.Module):
    def __init__(self, aggregation, bias):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict({
            'edge': tg.EdgeLinear(4, edge_features=2, bias=bias),
            'edge_relu': tg.EdgeReLU(),
            'node': tg.NodeLinear(8, node_features=4, bias=bias),
            'node_relu': tg.NodeReLU(),
        }))
        self.hidden = nn.Sequential(OrderedDict({
            'edge': tg.EdgeLinear(8, edge_features=4, sender_features=8, bias=bias),
            'edge_relu': tg.EdgeReLU(),
            'node': tg.NodeLinear(8, node_features=8, incoming_features=8, aggregation=aggregation, bias=bias),
            'node_relu': tg.NodeReLU()
        }))
        self.readout_nodes = tg.NodeLinear(1, node_features=8, bias=True)
        self.readout_globals = tg.GlobalLinear(1, node_features=8, aggregation='sum', bias=bias)

    def forward(self, graphs):
        graphs = self.encoder(graphs)
        graphs = self.hidden(graphs)
        nodes = self.readout_nodes(graphs).node_features
        globals = self.readout_globals(graphs).global_features

        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            num_edges_by_graph=None,
            edge_features=None,
            global_features=globals,
            senders=None,
            receivers=None
        )


def _reset_parameters(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            bound = 1 / math.sqrt(param.numel())
            nn.init.uniform_(param, -bound, bound)
        else:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))


def describe(cfg):
    from pathlib import Path
    from utils import import_
    klass = import_(cfg.model.klass)
    model = klass(*cfg.model.args, **cfg.model.kwargs)
    if 'state_dict' in cfg:
        model.load_state_dict(torch.load(Path(cfg.state_dict).expanduser().resolve()))
    print(model)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    for name, parameter in model.named_parameters():
        print(f'{name} {tuple(parameter.shape)}:')
        if 'state_dict' in cfg:
            print(parameter.numpy().round())
            print()


def main():
    from argparse import ArgumentParser
    from config import Config

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    sp_print = subparsers.add_parser('print', help='Print parsed configuration')
    sp_print.add_argument('config', nargs='*')
    sp_print.set_defaults(command=lambda c: print(c.toYAML()))

    sp_describe = subparsers.add_parser('describe', help='Describe a model')
    sp_describe.add_argument('config', nargs='*')
    sp_describe.set_defaults(command=describe)

    args = parser.parse_args()
    cfg = Config.build(*args.config)
    args.command(cfg)


if __name__ == '__main__':
    main()
