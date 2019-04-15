import torch
import torchgraphs as tg

from . import autograd_tricks as lrp


class EdgeLinearRelevance(tg.EdgeLinear):
    def forward(self, graphs: tg.GraphBatch) -> tg.GraphBatch:
        new_edges = torch.tensor(0)

        if self.W_edge is not None:
            new_edges = lrp.add(new_edges, lrp.linear_eps(graphs.edge_features, self.W_edge))
        if self.W_sender is not None:
            new_edges = lrp.add(
                new_edges,
                lrp.index_select(lrp.linear_eps(graphs.node_features, self.W_sender),
                                 dim=0, index=graphs.senders)
            )
        if self.W_receiver is not None:
            new_edges = lrp.add(
                new_edges,
                lrp.index_select(lrp.linear_eps(graphs.node_features, self.W_receiver),
                                 dim=0, index=graphs.receivers)
            )
        if self.W_global is not None:
            new_edges = lrp.add(
                new_edges,
                lrp.repeat_tensor(lrp.linear_eps(graphs.global_features, self.W_global),
                                  dim=0, repeats=graphs.num_edges_by_graph)
            )
        if self.bias is not None:
            new_edges = lrp.add(new_edges, self.bias)

        return graphs.evolve(edge_features=new_edges)


class NodeLinearRelevance(tg.NodeLinear):
    def __init__(self, out_features, node_features=None, incoming_features=None, outgoing_features=None,
                 global_features=None, aggregation=None, bias=True):
        super(NodeLinearRelevance, self).__init__(out_features, node_features, incoming_features,
                                                  outgoing_features, global_features, lrp.get_aggregation(aggregation),
                                                  bias)

    def forward(self, graphs: tg.GraphBatch) -> tg.GraphBatch:
        new_nodes = torch.tensor(0)

        if self.W_node is not None:
            new_nodes = lrp.add(
                new_nodes,
                lrp.linear_eps(graphs.node_features, self.W_node)
            )
        if self.W_incoming is not None:
            new_nodes = lrp.add(
                new_nodes,
                lrp.linear_eps(
                    self.aggregation(graphs.edge_features, dim=0, index=graphs.receivers, dim_size=graphs.num_nodes),
                    self.W_incoming)
            )
        if self.W_outgoing is not None:
            new_nodes = lrp.add(
                new_nodes,
                lrp.linear_eps(
                    self.aggregation(graphs.edge_features, dim=0, index=graphs.senders, dim_size=graphs.num_nodes),
                    self.W_outgoing)
            )
        if self.W_global is not None:
            new_nodes = lrp.add(
                new_nodes,
                lrp.repeat_tensor(lrp.linear_eps(graphs.global_features, self.W_global), dim=0,
                                  repeats=graphs.num_nodes_by_graph)
            )
        if self.bias is not None:
            new_nodes = lrp.add(new_nodes, self.bias)

        return graphs.evolve(node_features=new_nodes)


class GlobalLinearRelevance(tg.GlobalLinear):
    def __init__(self, out_features, node_features=None, edge_features=None, global_features=None,
                 aggregation=None, bias=True):
        super(GlobalLinearRelevance, self).__init__(out_features, node_features, edge_features,
                                                    global_features, lrp.get_aggregation(aggregation), bias)

    def forward(self, graphs: tg.GraphBatch) -> tg.GraphBatch:
        new_globals = torch.tensor(0)

        if self.W_node is not None:
            index = tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph)
            new_globals = lrp.add(
                new_globals,
                lrp.linear_eps(self.aggregation(graphs.node_features, dim=0, index=index, dim_size=graphs.num_graphs),
                               self.W_node)
            )
        if self.W_edges is not None:
            index = tg.utils.segment_lengths_to_ids(graphs.num_edges_by_graph)
            new_globals = lrp.add(
                new_globals,
                lrp.linear_eps(self.aggregation(graphs.edge_features, dim=0, index=index, dim_size=graphs.num_graphs),
                               self.W_edges)
            )
        if self.W_global is not None:
            new_globals = lrp.add(
                new_globals,
                lrp.linear_eps(graphs.global_features, self.W_global)
            )
        if self.bias is not None:
            new_globals = lrp.add(new_globals, self.bias)

        return graphs.evolve(global_features=new_globals)


class EdgeReLURelevance(tg.EdgeFunction):
    def __init__(self):
        super(EdgeReLURelevance, self).__init__(lrp.relu)


class NodeReLURelevance(tg.NodeFunction):
    def __init__(self):
        super(NodeReLURelevance, self).__init__(lrp.relu)


class GlobalReLURelevance(tg.GlobalFunction):
    def __init__(self):
        super(GlobalReLURelevance, self).__init__(lrp.relu)
