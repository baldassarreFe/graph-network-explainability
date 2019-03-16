import torch_scatter

from torch import nn
import torch.nn.functional as F

import torchgraphs as tg


class FullGN(nn.Module):
    def __init__(
            self,
            in_edge_features_shape, in_node_features_shape,
            out_edge_features_shape, out_node_features_shape
    ):
        super().__init__()
        self.f_e = nn.Linear(in_edge_features_shape, out_edge_features_shape)
        self.f_s = nn.Linear(in_node_features_shape, out_edge_features_shape)
        self.f_r = nn.Linear(in_node_features_shape, out_edge_features_shape)

        self.g_n = nn.Linear(in_node_features_shape, out_node_features_shape)
        self.g_in = nn.Linear(out_edge_features_shape, out_node_features_shape)
        self.g_out = nn.Linear(out_edge_features_shape, out_node_features_shape)

    def forward(self, graphs: tg.GraphBatch):
        edges = F.relu(
            self.f_e(graphs.edge_features) +
            self.f_s(graphs.node_features).index_select(dim=0, index=graphs.senders) +
            self.f_r(graphs.node_features).index_select(dim=0, index=graphs.receivers)
        )
        nodes = (
                self.g_n(graphs.node_features) +
                self.g_in(torch_scatter.scatter_max(edges, graphs.receivers, dim=0, dim_size=graphs.num_nodes)[0]) +
                self.g_out(torch_scatter.scatter_max(edges, graphs.senders, dim=0, dim_size=graphs.num_nodes)[0])
        )
        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            num_edges_by_graph=None,
            edge_features=None,
            global_features=None,
            senders=None,
            receivers=None
        )


class MinimalGN(nn.Module):
    def __init__(
            self,
            in_edge_features_shape, in_node_features_shape,
            out_edge_features_shape, out_node_features_shape
    ):
        super().__init__()
        self.f_s = nn.Linear(in_node_features_shape, out_edge_features_shape)
        self.g_n = nn.Linear(in_node_features_shape, out_node_features_shape)
        self.g_in = nn.Linear(out_edge_features_shape, out_node_features_shape)

    def forward(self, graphs: tg.GraphBatch):
        edges = F.relu(self.f_s(graphs.node_features).index_select(dim=0, index=graphs.senders))
        nodes = (
                self.g_n(graphs.node_features) +
                self.g_in(torch_scatter.scatter_max(edges, graphs.receivers, dim=0, dim_size=graphs.num_nodes)[0])
        )
        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            edge_features=None,
            num_edges_by_graph=None,
            global_features=None,
            senders=None,
            receivers=None
        )


class SubMinimalGN(nn.Module):
    def __init__(
            self,
            in_edge_features_shape, in_node_features_shape,
            out_edge_features_shape, out_node_features_shape
    ):
        super().__init__()
        self.f_s = nn.Linear(in_node_features_shape, out_edge_features_shape)
        self.g_in = nn.Linear(out_edge_features_shape, out_node_features_shape)

    def forward(self, graphs: tg.GraphBatch):
        edges = F.relu(self.f_s(graphs.node_features).index_select(dim=0, index=graphs.senders))
        nodes = self.g_in(torch_scatter.scatter_max(edges, graphs.receivers, dim=0, dim_size=graphs.num_nodes)[0])
        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            edge_features=None,
            num_edges_by_graph=None,
            global_features=None,
            senders=None,
            receivers=None
        )


def describe(cfg):
    from utils import load_class
    klass = load_class(cfg.model.klass)
    model = klass(**cfg.model.params)
    print(model)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


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
