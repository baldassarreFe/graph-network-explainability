import torch
import torchgraphs as tg

from . import autograd_tricks as guidedbp


class EdgeLinearGuidedBP(tg.EdgeLinear):
    def forward(self, graphs: tg.GraphBatch) -> tg.GraphBatch:
        new_edges = torch.tensor(0)

        if self.W_edge is not None:
            new_edges = guidedbp.add(new_edges, guidedbp.linear(graphs.edge_features, self.W_edge))
        if self.W_sender is not None:
            new_edges = guidedbp.add(
                new_edges,
                guidedbp.index_select(guidedbp.linear(graphs.node_features, self.W_sender),
                                      dim=0, index=graphs.senders)
            )
        if self.W_receiver is not None:
            new_edges = guidedbp.add(
                new_edges,
                guidedbp.index_select(guidedbp.linear(graphs.node_features, self.W_receiver),
                                      dim=0, index=graphs.receivers)
            )
        if self.W_global is not None:
            new_edges = guidedbp.add(
                new_edges,
                guidedbp.repeat_tensor(guidedbp.linear(graphs.global_features, self.W_global),
                                       dim=0, repeats=graphs.num_edges_by_graph)
            )
        if self.bias is not None:
            new_edges = guidedbp.add(new_edges, self.bias)

        return graphs.evolve(edge_features=new_edges)


class NodeLinearGuidedBP(tg.NodeLinear):
    def __init__(self, out_features, node_features=None, incoming_features=None, outgoing_features=None,
                 global_features=None, aggregation=None, bias=True):
        super(NodeLinearGuidedBP, self).__init__(out_features, node_features, incoming_features,
                                                  outgoing_features, global_features,
                                                  guidedbp.get_aggregation(aggregation),
                                                  bias)

    def forward(self, graphs: tg.GraphBatch) -> tg.GraphBatch:
        new_nodes = torch.tensor(0)

        if self.W_node is not None:
            new_nodes = guidedbp.add(
                new_nodes,
                guidedbp.linear(graphs.node_features, self.W_node)
            )
        if self.W_incoming is not None:
            new_nodes = guidedbp.add(
                new_nodes,
                guidedbp.linear(
                    self.aggregation(graphs.edge_features, dim=0, index=graphs.receivers, dim_size=graphs.num_nodes),
                    self.W_incoming)
            )
        if self.W_outgoing is not None:
            new_nodes = guidedbp.add(
                new_nodes,
                guidedbp.linear(
                    self.aggregation(graphs.edge_features, dim=0, index=graphs.senders, dim_size=graphs.num_nodes),
                    self.W_outgoing)
            )
        if self.W_global is not None:
            new_nodes = guidedbp.add(
                new_nodes,
                guidedbp.repeat_tensor(guidedbp.linear(graphs.global_features, self.W_global), dim=0,
                                       repeats=graphs.num_nodes_by_graph)
            )
        if self.bias is not None:
            new_nodes = guidedbp.add(new_nodes, self.bias)

        return graphs.evolve(node_features=new_nodes)


class GlobalLinearGuidedBP(tg.GlobalLinear):
    def __init__(self, out_features, node_features=None, edge_features=None, global_features=None,
                 aggregation=None, bias=True):
        super(GlobalLinearGuidedBP, self).__init__(out_features, node_features, edge_features,
                                                    global_features, guidedbp.get_aggregation(aggregation), bias)

    def forward(self, graphs: tg.GraphBatch) -> tg.GraphBatch:
        new_globals = torch.tensor(0)

        if self.W_node is not None:
            index = tg.utils.segment_lengths_to_ids(graphs.num_nodes_by_graph)
            new_globals = guidedbp.add(
                new_globals,
                guidedbp.linear(self.aggregation(graphs.node_features, dim=0, index=index, dim_size=graphs.num_graphs),
                                self.W_node)
            )
        if self.W_edges is not None:
            index = tg.utils.segment_lengths_to_ids(graphs.num_edges_by_graph)
            new_globals = guidedbp.add(
                new_globals,
                guidedbp.linear(self.aggregation(graphs.edge_features, dim=0, index=index, dim_size=graphs.num_graphs),
                                self.W_edges)
            )
        if self.W_global is not None:
            new_globals = guidedbp.add(
                new_globals,
                guidedbp.linear(graphs.global_features, self.W_global)
            )
        if self.bias is not None:
            new_globals = guidedbp.add(new_globals, self.bias)

        return graphs.evolve(global_features=new_globals)


class EdgeReLUGuidedBP(tg.EdgeFunction):
    def __init__(self):
        super(EdgeReLUGuidedBP, self).__init__(guidedbp.relu)


class NodeReLUGuidedBP(tg.NodeFunction):
    def __init__(self):
        super(NodeReLUGuidedBP, self).__init__(guidedbp.relu)


class GlobalReLUGuidedBP(tg.GlobalFunction):
    def __init__(self):
        super(GlobalReLUGuidedBP, self).__init__(guidedbp.relu)
