from collections import OrderedDict

import torch
from torch import nn

import torchgraphs as tg


def build_network(num_hidden):
    return SolubilityGN(num_hidden)


class SolubilityGN(nn.Module):
    def __init__(self, num_layers, hidden_bias, hidden_node, dropout):
        super().__init__()

        hidden_edge = hidden_node // 4
        hidden_global = hidden_node // 8
        
        self.encoder = nn.Sequential(OrderedDict({
            'edge': tg.EdgeLinear(hidden_edge, edge_features=6),
            'edge_relu': tg.EdgeReLU(),
            'node': tg.NodeLinear(hidden_node, node_features=47),
            'node_relu': tg.NodeReLU(),
            'global': tg.GlobalLinear(hidden_global, node_features=hidden_node,
                                      edge_features=hidden_edge, aggregation='mean'),
            'global_relu': tg.GlobalReLU(),
        }))
        if dropout:
            self.hidden = nn.Sequential(OrderedDict({
                f'hidden_{i}': nn.Sequential(OrderedDict({
                    'edge': tg.EdgeLinear(hidden_edge, edge_features=hidden_edge,
                                          sender_features=hidden_node, bias=hidden_bias),
                    'edge_relu': tg.EdgeReLU(),
                    'edge_dropout': tg.EdgeDroput(),
                    'node': tg.NodeLinear(hidden_node, node_features=hidden_node, incoming_features=hidden_edge,
                                          aggregation='mean', bias=hidden_bias),
                    'node_relu': tg.NodeReLU(),
                    'node_dropout': tg.EdgeDroput(),
                    'global': tg.GlobalLinear(hidden_global, node_features=hidden_node, edge_features=hidden_edge,
                                              global_features=hidden_global, aggregation='mean', bias=hidden_bias),
                    'global_relu': tg.GlobalReLU(),
                    'global_dropout': tg.EdgeDroput(),
                }))
                for i in range(num_layers)
            }))
        else:
            self.hidden = nn.Sequential(OrderedDict({
                f'hidden_{i}': nn.Sequential(OrderedDict({
                    'edge': tg.EdgeLinear(hidden_edge, edge_features=hidden_edge,
                                          sender_features=hidden_node, bias=hidden_bias),
                    'edge_relu': tg.EdgeReLU(),
                    'node': tg.NodeLinear(hidden_node, node_features=hidden_node, incoming_features=hidden_edge,
                                          aggregation='mean', bias=hidden_bias),
                    'node_relu': tg.NodeReLU(),
                    'global': tg.GlobalLinear(hidden_global, node_features=hidden_node, edge_features=hidden_edge,
                                              global_features=hidden_global, aggregation='mean', bias=hidden_bias),
                    'global_relu': tg.GlobalReLU(),
                }))
                for i in range(num_layers)
            }))
        self.readout_globals = tg.GlobalLinear(1, global_features=hidden_global, bias=True)

    def forward(self, graphs):
        graphs = self.encoder(graphs)
        graphs = self.hidden(graphs)
        globals = self.readout_globals(graphs).global_features

        return graphs.evolve(
            num_nodes=0,
            node_features=None,
            num_nodes_by_graph=None,
            num_edges=0,
            num_edges_by_graph=None,
            edge_features=None,
            global_features=globals,
            senders=None,
            receivers=None
        )


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
