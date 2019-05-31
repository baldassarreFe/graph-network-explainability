"""
class DenseW2(RelevanceFunction):
    @staticmethod
    def forward_relevance(module, inputs, ctx):
        output = inputs @ module.weight.t()
        if module.bias is not None:
            output += module.bias
        return output

    @staticmethod
    def backward_relevance(module, relevance_outputs, ctx):
        return relevance_outputs @ (module.weight.pow(2) / (module.weight.pow(2).sum(dim=1, keepdim=True) + 10e-6))


class DenseZPlus(RelevanceFunction):
    @staticmethod
    def forward_relevance(module, inputs, ctx):
        ctx['inputs'] = inputs
        output = inputs @ module.weight.t()
        if module.bias is not None:
            output += module.bias
        return output

    @staticmethod
    def backward_relevance(module, relevance_outputs, ctx):
        inputs = ctx['inputs']
        return inputs * (
                (relevance_outputs / (inputs @ module.weight.clamp(min=0).t() + 10e-6)) @
                module.weight.clamp(min=0)
        )
"""
