import warnings

import torch
import torch_scatter


class RelevanceFunction(object):
    def __init__(self):
        self.saved_stuff = None


class IndexSelect(RelevanceFunction):
    def forward(self, input, idx, dim):
        self.saved_stuff = idx, dim, input.shape[dim]
        return torch.index_select(input, dim=dim, index=idx)

    def relevance(self, rel_out):
        idx, dim, input_dim = self.saved_stuff
        return torch_scatter.scatter_add(rel_out, idx, dim=dim, dim_size=input_dim)


class SumPooling(RelevanceFunction):
    def forward(self, input):
        output = input.sum()
        self.saved_stuff = input, output
        return output

    def relevance(self, rel_out):
        input, output = self.saved_stuff
        if output == 0:
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
            return torch.zeros_like(input)
        return rel_out * input / output


class ScatterAdd(RelevanceFunction):
    def forward(self, input, idx, dim, dim_size):
        output = torch_scatter.scatter_add(input, idx, dim=dim, dim_size=dim_size)
        self.saved_stuff = input, idx, dim, output
        return output

    def relevance(self, rel_out):
        input, idx, dim, output = self.saved_stuff
        if ((output == 0) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        return torch.index_select(rel_out / output, dim, idx) * input


class ScatterMean(RelevanceFunction):
    def forward(self, input, idx, dim_size):
        sums = torch_scatter.scatter_add(input, idx, dim=0, dim_size=dim_size)
        counts = torch_scatter.scatter_add(torch.ones_like(input), idx, dim=0, dim_size=dim_size)
        self.saved_stuff = input, idx, sums
        return sums / counts.clamp(min=1)

    def relevance(self, rel_out):
        input, idx, sums = self.saved_stuff
        if ((sums == 0) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        return torch.index_select(rel_out / sums, 0, idx) * input


class ScatterMax(RelevanceFunction):
    def forward(self, input, idx, dim, dim_size):
        output, idx_maxes = torch_scatter.scatter_max(input, idx, dim=dim, dim_size=dim_size)
        self.saved_stuff = idx, dim, input.shape[dim], output, idx_maxes
        return output, idx_maxes

    def relevance(self, rel_out):
        idx, dim, input_dim, output, idx_maxes = self.saved_stuff
        if ((idx_maxes == -1) & (rel_out > 0)).any():
            warnings.warn('Relevance that is propagated back through an output of 0 will be lost')
        # Where idx_maxes==-1 set idx=0 so that the indexes are valid for scatter_add
        # The corresponding relevance should already be 0, but set it relevance=0 to be sure
        rel_out = torch.where(idx_maxes != -1, rel_out, torch.zeros_like(rel_out))
        idx_maxes = torch.where(idx_maxes != -1, idx_maxes, torch.zeros_like(idx_maxes))

        return torch_scatter.scatter_add(rel_out, idx_maxes, dim=dim, dim_size=input_dim)


class Cat(RelevanceFunction):
    def forward(self, *inputs, dim=0):
        self.saved_stuff = dim, [x.shape[dim] for x in inputs]
        return torch.cat(inputs, dim=dim)

    def relevance(self, rel_out):
        dim, shapes = self.saved_stuff
        return torch.split(rel_out, split_size_or_sections=shapes, dim=dim)


class Dense(RelevanceFunction):
    def __init__(self, W, b):
        super(Dense, self).__init__()
        self.W = W
        self.b = b

    def forward(self, x):
        self.saved_stuff = x
        output = x @ self.W.t() + self.b
        return output

    def relevance(self, rel_out):
        x = self.saved_stuff
        return x * ((rel_out / (x @ self.W.clamp(min=0).t() + 10e-6)) @ self.W.clamp(min=0))
