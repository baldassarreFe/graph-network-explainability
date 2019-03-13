from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn, optim


class Saver(object):
    def __init__(self, folder: Union[str, Path]):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def save(self, *, id_: str = '', model: Optional[nn.Module] = None, optimizer: Optional[optim.Optimizer] = None,
             **rest):
        if model is not None:
            if isinstance(model, nn.DataParallel):
                model = model.module
            rest['model'] = model.state_dict()
        if optimizer is not None:
            rest['optimizer'] = optimizer.state_dict()
        torch.save(rest, self.folder / f'checkpoint_{id_}.pt')

    def load(self, *, id_: str = '', model: Optional[nn.Module] = None, optimizer: Optional[optim.Optimizer] = None,
             device: Union[str, torch.device] = None):
        rest = torch.load(self.folder / f'checkpoint_{id_}.pt', map_location=device)
        if model is not None:
            if isinstance(model, nn.DataParallel):
                model = model.module
            model.load_state_dict(rest['model'])
            del rest['model']
        if optimizer is not None:
            model.load_state_dict(rest['optimizer'])
            del rest['optimizer']
        return rest
