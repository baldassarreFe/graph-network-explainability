import itertools

import torch

import relevance

import torchgraphs as tg

torch.set_grad_enabled(False)
relevance.register_function(torch.nn.Linear, relevance.LinearEpsilon)

graphs = tg.GraphBatch.from_graphs([tg.Graph(
    node_features=torch.rand(10, 9),
    edge_features=torch.rand(6, 5),
    senders=torch.tensor([0, 0, 1, 2, 4, 5]),
    receivers=torch.tensor([1, 2, 2, 4, 3, 3]),
    global_features=torch.rand(7)
)])

print('bias nf ef gf R_out R_in'.replace(' ', '\t'))
print('-' * 45)

for bias, nf, ef, gf in itertools.product(
        [True, False, 'zero'],
        [9, None],
        [5, None],
        [7, None]
):
    if nf is ef is gf is None:
        continue
    net = tg.GlobalLinear(3, node_features=nf, edge_features=ef, global_features=gf, bias=bool(bias), aggregation='sum')

    if bias == 'zero':
        net.bias.zero_()

    ctx = {}
    out = relevance.forward_relevance(net, graphs, ctx=ctx)
    torch.testing.assert_allclose(out.node_features, net(graphs).node_features)

    rel_out = graphs.evolve(
        global_features=torch.ones_like(out.global_features) * (out.global_features != 0).float()
    )

    rel_in = relevance.backward_relevance(net, rel_out, ctx=ctx)

    # If bias==0 then relevance is conserved at a graph level
    print(
        {True: '  x ', False: '  - ', 'zero': '  0 '}[bias], nf if nf else '-', ef if ef else '-', gf if gf else '-',
        rel_out.global_features.sum().item(),
        ((rel_in.global_features.sum() if rel_in.global_features is not None else 0) +
         (rel_in.edge_features.sum() if rel_in.edge_features is not None else 0) +
         (rel_in.node_features.sum() if rel_in.node_features is not None else 0)).item(),
        sep='\t'
    )
