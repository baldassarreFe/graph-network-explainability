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

print('bias ef sf rf gf out input'.replace(' ', '\t'))
print('-' * 45)

for bias, ef, sf, rf, gf in itertools.product(
        [True, False, 'zero'],
        [5, None],
        [9, None],
        [9, None],
        [7, None]
):
    if ef is sf is rf is gf is None:
        continue
    net = tg.EdgeLinear(3, edge_features=ef, sender_features=sf, receiver_features=rf, global_features=gf, bias=bool(bias))
    if bias == 'zero':
        net.bias.zero_()

    ctx = {}
    out = relevance.forward_relevance(net, graphs, ctx=ctx)
    torch.testing.assert_allclose(out.edge_features, net(graphs).edge_features)

    rel_out = graphs.evolve(
        edge_features=torch.ones_like(out.edge_features) * (out.edge_features != 0).float()
    )

    rel_in = relevance.backward_relevance(net, rel_out, ctx=ctx)

    # If bias==0 then relevance is conserved at a graph level
    print(
        {True: '  x ', False: '  - ', 'zero': '  0 '}[bias], ef if ef else '-', sf if sf else '-',
        rf if rf else '-', gf if gf else '-',
        rel_out.edge_features.sum().item(),
        ((rel_in.global_features.sum() if rel_in.global_features is not None else 0) +
         (rel_in.edge_features.sum() if rel_in.edge_features is not None else 0) +
         (rel_in.node_features.sum() if rel_in.node_features is not None else 0)).item(),
        sep='\t'
    )
