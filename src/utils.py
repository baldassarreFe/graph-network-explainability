def load_class(fullname):
    import importlib
    package, name = fullname.rsplit('.', maxsplit=1)
    package = importlib.import_module(package)
    return getattr(package, name)


def set_seeds(seed):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)