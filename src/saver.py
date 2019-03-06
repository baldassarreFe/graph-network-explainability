from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn, optim


class Saver(object):
    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True)

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

    '''
    def save_model(self, model: nn.model, id=''):
        if isinstance(model, nn.DataParallel):
            model = model.module
        torch.save(model.state_dict(), self.folder / f'model_{id}.pt')

    def load_model(self, model: nn.model, id='', device=None):
        state_dict = torch.load(self.folder / f'model_{id}.pt', map_location=device)
        model.load_state_dict(state_dict)
        return model.to(device)

    def save_training(self, optimizer: optim.Optimizer, **kwargs):
        torch.save({'_optimizer': optimizer.state_dict(), **kwargs}, self.folder / f'training_{id}.pt')

    def load_training(self, optimizer: optim.Optimizer):
        loaded = torch.load(self.folder / f'training_{id}.pt')
        optimizer.load_state_dict(loaded['_optimizer'])
        del loaded['_optimizer']
        return optimizer, loaded
    '''
