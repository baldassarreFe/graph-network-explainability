import math

import torch_scatter

import torch
from torch import nn
import torch.nn.functional as F

import torchgraphs as tg


class FullGN(nn.Module):
    def __init__(
            self,
            in_edge_features_shape, in_node_features_shape,
            out_edge_features_shape, out_node_features_shape,
            out_global_features_shape
    ):
        super().__init__()

        self.f_edge = nn.Parameter(torch.Tensor(out_edge_features_shape, in_edge_features_shape))
        self.f_sender = nn.Parameter(torch.Tensor(out_edge_features_shape, in_node_features_shape))
        self.f_receiver = nn.Parameter(torch.Tensor(out_edge_features_shape, in_node_features_shape))
        self.f_bias = nn.Parameter(torch.Tensor(out_edge_features_shape))

        self.g_node = nn.Parameter(torch.Tensor(out_node_features_shape, in_node_features_shape))
        self.g_in = nn.Parameter(torch.Tensor(out_node_features_shape, out_edge_features_shape))
        self.g_out = nn.Parameter(torch.Tensor(out_node_features_shape, out_edge_features_shape))
        self.g_bias = nn.Parameter(torch.Tensor(out_node_features_shape))

        self.h_nodes = nn.Parameter(torch.Tensor(out_node_features_shape, out_global_features_shape))
        self.h_edges = nn.Parameter(torch.Tensor(out_edge_features_shape, out_global_features_shape))
        self.h_bias = nn.Parameter(torch.Tensor(out_global_features_shape))

        _reset_parameters(self)

    def forward(self, graphs: tg.GraphBatch):
        edges = F.relu(
            graphs.edge_features @ self.f_edge.t() +
            (graphs.node_features @ self.f_sender.t()).index_select(dim=0, index=graphs.senders) +
            (graphs.node_features @ self.f_receiver.t()).index_select(dim=0, index=graphs.receivers) +
            self.f_bias
        )
        incoming_edges_agg = torch_scatter.scatter_max(edges, graphs.receivers, dim=0, dim_size=graphs.num_nodes)[0]
        outgoing_edges_agg = torch_scatter.scatter_max(edges, graphs.senders, dim=0, dim_size=graphs.num_nodes)[0]
        nodes = (
                graphs.node_features @ self.g_node.t() +
                incoming_edges_agg @ self.g_in.t() +
                outgoing_edges_agg @ self.g_out.t() +
                self.g_bias
        )
        edges_agg = torch_scatter.scatter_add(F.relu(edges), tg.utils.segment_lengths_to_ids(graphs.num_edges_by_graph),
                                              dim=0, dim_size=graphs.num_graphs)
        nodes_agg = torch_scatter.scatter_add(F.relu(nodes), tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph),
                                              dim=0, dim_size=graphs.num_graphs)
        globals = (
            edges_agg @ self.h_edges.t() +
            nodes_agg @ self.h_nodes.t() +
            self.h_bias
        )
        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            num_edges_by_graph=None,
            edge_features=None,
            global_features=globals,
            senders=None,
            receivers=None
        )


class MinimalGN(nn.Module):
    def __init__(
            self,
            in_edge_features_shape, in_node_features_shape,
            out_edge_features_shape, out_node_features_shape,
            out_global_features_shape
    ):
        super().__init__()

        self.f_sender = nn.Parameter(torch.Tensor(out_edge_features_shape, in_node_features_shape))
        self.f_bias = nn.Parameter(torch.Tensor(out_edge_features_shape))

        self.g_node = nn.Parameter(torch.Tensor(out_node_features_shape, in_node_features_shape))
        self.g_in = nn.Parameter(torch.Tensor(out_node_features_shape, out_edge_features_shape))
        self.g_bias = nn.Parameter(torch.Tensor(out_node_features_shape))

        self.h_nodes = nn.Parameter(torch.Tensor(out_node_features_shape, out_global_features_shape))
        self.h_bias = nn.Parameter(torch.Tensor(out_global_features_shape))

        _reset_parameters(self)

    def forward(self, graphs: tg.GraphBatch):
        edges = F.relu(
            (graphs.node_features @ self.f_sender.t()).index_select(dim=0, index=graphs.senders) +
            self.f_bias
        )
        incoming_edges_agg = (torch_scatter.scatter_max(edges, graphs.receivers, dim=0, dim_size=graphs.num_nodes)[0])
        nodes = (
                graphs.node_features @ self.g_node.t() +
                incoming_edges_agg @ self.g_in.t() +
                self.g_bias
        )
        nodes_agg = torch_scatter.scatter_add(F.relu(nodes), tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph),
                                              dim=0, dim_size=graphs.num_graphs)
        globals = (
                nodes_agg @ self.h_nodes.t() +
                self.h_bias
        )
        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            edge_features=None,
            num_edges_by_graph=None,
            global_features=globals,
            senders=None,
            receivers=None
        )


class SubMinimalGN(nn.Module):
    def __init__(
            self,
            in_edge_features_shape, in_node_features_shape,
            out_edge_features_shape, out_node_features_shape,
            out_global_features_shape):
        super().__init__()

        self.f_sender = nn.Parameter(torch.Tensor(out_edge_features_shape, in_node_features_shape))
        self.f_bias = nn.Parameter(torch.Tensor(out_edge_features_shape))

        self.g_in = nn.Parameter(torch.Tensor(out_node_features_shape, out_edge_features_shape))
        self.g_bias = nn.Parameter(torch.Tensor(out_node_features_shape))

        self.h_nodes = nn.Parameter(torch.Tensor(out_node_features_shape, out_global_features_shape))
        self.h_bias = nn.Parameter(torch.Tensor(out_global_features_shape))

        _reset_parameters(self)

    def forward(self, graphs: tg.GraphBatch):
        edges = F.relu(
            (graphs.node_features @ self.f_sender.t()).index_select(dim=0, index=graphs.senders) +
            self.f_bias
        )
        incoming_edges_agg = (torch_scatter.scatter_max(edges, graphs.receivers, dim=0, dim_size=graphs.num_nodes)[0])
        nodes = (
                incoming_edges_agg @ self.g_in.t() +
                self.g_bias
        )
        nodes_agg = torch_scatter.scatter_add(F.relu(nodes), tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph),
                                              dim=0, dim_size=graphs.num_graphs)
        globals = (
                nodes_agg @ self.h_nodes.t() +
                self.h_bias
        )
        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            edge_features=None,
            num_edges_by_graph=None,
            global_features=globals,
            senders=None,
            receivers=None
        )


def _reset_parameters(gn):
    for name, param in gn.named_parameters():
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
