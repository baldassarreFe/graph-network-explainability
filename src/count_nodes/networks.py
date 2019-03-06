import torch_scatter

from torch import nn
import torch.nn.functional as F

import torchgraphs as tg
from torchgraphs.utils import segment_lengths_to_ids


class FullGN(nn.Module):
    def __init__(
            self,
            in_edge_features_shape, in_node_features_shape, in_global_features_shape,
            out_edge_features_shape, out_node_features_shape, out_global_features_shape
    ):
        super().__init__()
        self.f_e = nn.Linear(in_edge_features_shape, out_edge_features_shape)
        self.f_s = nn.Linear(in_node_features_shape, out_edge_features_shape)
        self.f_r = nn.Linear(in_node_features_shape, out_edge_features_shape)
        self.f_u = nn.Linear(in_global_features_shape, out_edge_features_shape)

        self.g_n = nn.Linear(in_node_features_shape, out_node_features_shape)
        self.g_in = nn.Linear(out_edge_features_shape, out_node_features_shape)
        self.g_out = nn.Linear(out_edge_features_shape, out_node_features_shape)
        self.g_u = nn.Linear(in_global_features_shape, out_node_features_shape)

        self.h_n = nn.Linear(out_node_features_shape, out_global_features_shape)
        self.h_e = nn.Linear(out_edge_features_shape, out_global_features_shape)
        self.h_u = nn.Linear(in_global_features_shape, out_global_features_shape)

    def forward(self, graphs: tg.GraphBatch):
        edges = F.relu(
            self.f_e(graphs.edge_features) +
            self.f_s(graphs.node_features).index_select(dim=0, index=graphs.senders) +
            self.f_r(graphs.node_features).index_select(dim=0, index=graphs.receivers) +
            tg.utils.repeat_tensor(self.f_u(graphs.global_features), graphs.num_edges_by_graph)
        )
        nodes = F.relu(
            self.g_n(graphs.node_features) +
            self.g_in(torch_scatter.scatter_add(edges, graphs.receivers, dim=0, dim_size=graphs.num_nodes)) +
            self.g_out(torch_scatter.scatter_add(edges, graphs.senders, dim=0, dim_size=graphs.num_nodes)) +
            tg.utils.repeat_tensor(self.g_u(graphs.global_features), graphs.num_nodes_by_graph)
        )
        globals = (
                self.h_e(torch_scatter.scatter_add(
                    edges, segment_lengths_to_ids(graphs.num_edges_by_graph), dim=0, dim_size=graphs.num_graphs)) +
                self.h_n(torch_scatter.scatter_add(
                    nodes, segment_lengths_to_ids(graphs.num_nodes_by_graph), dim=0, dim_size=graphs.num_graphs)) +
                self.h_u(graphs.global_features)
        )
        return graphs.evolve(
            edge_features=edges,
            node_features=nodes,
            global_features=globals,
        )


class MinimalGN(nn.Module):
    def __init__(self, in_node_features_shape, out_node_features_shape, out_global_features_shape):
        super().__init__()
        self.g_n = nn.Linear(in_node_features_shape, out_node_features_shape)
        self.h_n = nn.Linear(out_node_features_shape, out_global_features_shape)

    def forward(self, graphs: tg.GraphBatch):
        nodes = F.relu(self.g_n(graphs.node_features))
        globals = self.h_n(torch_scatter.scatter_add(
            nodes, segment_lengths_to_ids(graphs.num_nodes_by_graph), dim=0, dim_size=graphs.num_graphs))
        return graphs.evolve(
            num_edges=0,
            edge_features=None,
            node_features=None,
            global_features=globals,
            senders=None,
            receivers=None
        )
