import warnings
import numpy as np

import torch
import torch_scatter


class AddRelevance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out = a + b
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    def backward(ctx, rel_out):
        a, b, out = ctx.saved_tensors
        if ((out == 0) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        rel_a = torch.where(out != 0, rel_out * a / out, out.new_tensor(0))
        rel_b = torch.where(out != 0, rel_out * b / out, out.new_tensor(0))
        return rel_a, rel_b


def add(a, b):
    return AddRelevance.apply(a, b)


class SumPooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, dim, keepdim):
        out = torch.sum(src, dim=dim, keepdim=keepdim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.save_for_backward(src, out)
        return out

    @staticmethod
    def backward(ctx, rel_out):
        src, out = ctx.saved_tensors
        if ((out == 0) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        rel_out = torch.where(out != 0, rel_out / out, out.new_tensor(0))
        if not ctx.keepdim and ctx.dim is not None:
            rel_out.unsqueeze_(ctx.dim)
        return rel_out * src, None, None


def sum(tensor, dim=None, keepdim=False):
    return SumPooling.apply(tensor, dim, keepdim)



class ScatterAddRelevance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, out, idx, dim):
        torch_scatter.scatter_add(src, idx, dim=dim, out=out)
        ctx.dim = dim
        ctx.save_for_backward(src, idx, out)
        return out

    @staticmethod
    def backward(ctx, rel_out):
        src, idx, out = ctx.saved_tensors
        if ((out == 0) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        rel_out = torch.where(out != 0, rel_out / out, out.new_tensor(0))
        rel_src = torch.index_select(rel_out, ctx.dim, idx) * src
        return rel_src, None, None, None


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, _, dim = torch_scatter.utils.gen.gen(src, index, dim, out, dim_size, fill_value)
    return ScatterAddRelevance.apply(src, out, index, dim)


class ScatterMeanRelevance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, dim_size, fill_value):
        sums = torch_scatter.scatter_add(src, idx, dim, None, dim_size, fill_value)
        count = torch_scatter.scatter_add(torch.ones_like(src), idx, dim, None, dim_size, fill_value=0)
        out = sums / count.clamp(min=1)
        ctx.dim = dim
        ctx.save_for_backward(src, idx, sums)
        return out

    @staticmethod
    def backward(ctx, rel_out):
        src, idx, sums = ctx.saved_tensors
        if ((sums == 0) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        rel_out = torch.where(sums != 0, rel_out / sums, sums.new_tensor(0))
        rel_src = torch.index_select(rel_out, ctx.dim, idx) * src
        return rel_src, None, None, None, None


def scatter_mean(src, index, dim=-1, dim_size=None, fill_value=0):
    return ScatterMeanRelevance.apply(src, index, dim, dim_size, fill_value)


class ScatterMaxRelevance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, dim_size, fill_value):
        out, idx_maxes = torch_scatter.scatter_max(src, idx, dim=dim, dim_size=dim_size, fill_value=fill_value)
        ctx.dim = dim
        ctx.dim_size = src.shape[dim]
        ctx.save_for_backward(idx, out, idx_maxes)
        return out, idx_maxes

    @staticmethod
    def backward(ctx, rel_out, rel_idx_maxes):
        idx, out, idx_maxes = ctx.saved_tensors
        if ((out == 0) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        rel_out = torch.where(out != 0, rel_out, out.new_tensor(0))

        # Where idx_maxes==-1 set idx=0 so that the indexes are valid for scatter_add
        # The corresponding relevance should already be 0, but set it relevance=0 to be sure
        rel_out = torch.where(idx_maxes != -1, rel_out, torch.zeros_like(rel_out))
        idx_maxes = torch.where(idx_maxes != -1, idx_maxes, torch.zeros_like(idx_maxes))

        rel_src = torch_scatter.scatter_add(rel_out, idx_maxes, dim=ctx.dim, dim_size=ctx.dim_size)
        return rel_src, None, None, None, None


def scatter_max(src, index, dim=-1, dim_size=None, fill_value=0):
    return ScatterMaxRelevance.apply(src, index, dim, dim_size, fill_value)


class LinearEpsilonRelevance(torch.autograd.Function):
    eps = 1e-16

    @staticmethod
    def forward(ctx, input, weight, bias):
        Z = weight.t()[None, :, :] * input[:, :, None]
        Zs = Z.sum(dim=1, keepdim=True)
        if bias is not None:
            Zs += bias[None, None, :]
        ctx.save_for_backward(Z, Zs)
        return Zs.squeeze(dim=1)

    @staticmethod
    def backward(ctx, rel_out):
        Z, Zs = ctx.saved_tensors
        eps = rel_out.new_tensor(LinearEpsilonRelevance.eps)
        Zs += torch.where(Zs >= 0, eps, -eps)
        return (rel_out[:, None, :] * Z / Zs).sum(dim=2), None, None


def linear_eps(input, weight, bias=None):
    return LinearEpsilonRelevance.apply(input, weight, bias)


class IndexSelectRelevance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, dim, idx):
        out = torch.index_select(src, dim, idx)
        ctx.dim = dim
        ctx.dim_size = src.shape[dim]
        ctx.save_for_backward(src, idx, out)
        return out

    @staticmethod
    def backward(ctx, rel_out):
        src, idx, out = ctx.saved_tensors
        return torch_scatter.scatter_add(rel_out, idx, dim=ctx.dim, dim_size=ctx.dim_size), None, None


def index_select(src, dim, index):
    return IndexSelectRelevance.apply(src, dim, index)


class CatRelevance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, *tensors):
        ctx.dim = dim
        ctx.sizes = [t.shape[dim] for t in tensors]
        return torch.cat(tensors, dim)

    @staticmethod
    def backward(ctx, rel_out):
        return (None, *torch.split_with_sizes(rel_out, dim=ctx.dim, split_sizes=ctx.sizes))


def cat(tensors, dim=0):
    return CatRelevance.apply(dim, *tensors)


def repeat_tensor(src, repeats, dim=0):
    idx = src.new_tensor(np.arange(len(repeats)).repeat(repeats.cpu().numpy()), dtype=torch.long)
    return torch.index_select(src, dim, idx)


class ReLuRelevance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = input.clamp(min=0)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, rel_out):
        return rel_out


def relu(input):
    return ReLuRelevance.apply(input)


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
