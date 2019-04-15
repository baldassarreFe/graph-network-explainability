import numpy as np

import torch
import torch_scatter


def hook(grad):
    return grad.clamp(min=0)


def add(a, b):
    out = torch.add(a, b)
    out.register_hook(hook)
    return out


def sum(tensor, dim=None, keepdim=False):
    out = torch.sum(tensor, dim=dim, keepdim=keepdim)
    out.register_hook(hook)
    return out


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    out = torch_scatter.scatter_add(src, index, dim, out, dim_size, fill_value)
    out.register_hook(hook)
    return out


def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    out = torch_scatter.scatter_mean(src, index, dim, out, dim_size, fill_value)
    out.register_hook(hook)
    return out


def scatter_max(src, index, dim=-1, dim_size=None, fill_value=0):
    out = torch_scatter.scatter_max(src, index, dim, None, dim_size, fill_value)
    out[0].register_hook(hook)
    return out


def linear(input, weight, bias=None):
    out = torch.nn.functional.linear(input, weight, bias)
    out.register_hook(hook)
    return out


def index_select(src, dim, index):
    out = torch.index_select(src, dim, index)
    out.register_hook(hook)
    return out


def cat(tensors, dim=0):
    out = torch.cat(tensors, dim)
    out.register_hook(hook)
    return out


def repeat_tensor(src, repeats, dim=0):
    idx = src.new_tensor(np.arange(len(repeats)).repeat(repeats.cpu().numpy()), dtype=torch.long)
    return index_select(src, dim, idx)


def relu(input):
    out = torch.nn.functional.relu(input)
    out.register_hook(hook)
    return out


def get_aggregation(name):
    if name in ('add', 'sum'):
        return scatter_add
    elif name in ('mean', 'avg'):
        return scatter_mean
    elif name == 'max':
        from functools import wraps

        @wraps(scatter_max)
        def wrapper(*args, **kwargs):
            return scatter_max(*args, **kwargs)[0]

        return wrapper
