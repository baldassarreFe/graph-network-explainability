import torch
import torch_scatter
import torchgraphs as tg

import textwrap

from . import autograd_tricks as lrp



def patch():
    torch.add = lrp.add
    torch.cat = lrp.cat
    torch.index_select = lrp.index_select

    tg.utils.repeat_tensor = lrp.repeat_tensor

    torch_scatter.scatter_add = lrp.scatter_add
    torch_scatter.scatter_mean = lrp.scatter_mean
    torch_scatter.scatter_max = lrp.scatter_max

    torch.nn.functional.linear = lrp.linear_eps


def computational_graph(op):
    if op is None:
        return 'None'
    res = f'{op.__class__.__name__} at {hex(id(op))}:'
    if op.__class__.__name__ == 'AccumulateGrad':
        res += f'variable at {hex(id(op.variable))}'
    for op in op.next_functions:
        res += '\n-' + textwrap.indent(computational_graph(op[0]), ' ')
    return res
