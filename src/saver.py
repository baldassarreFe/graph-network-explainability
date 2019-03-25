import os
from pathlib import Path
from typing import Union

import pyaml
import torch


class Saver(object):
    def __init__(self, folder: Union[str, Path]):
        self.base_folder = Path(folder).expanduser().resolve()
        self.checkpoint_folder = self.base_folder / 'checkpoints'
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, suffix=None, is_best=False):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        name = 'model.pt' if suffix is None else f'model.{suffix}.pt'
        model_path = self.checkpoint_folder / name
        torch.save(model.state_dict(), model_path)

        latest_path = self.base_folder / 'model.latest.pt'
        if latest_path.exists():
            os.unlink(latest_path)
        os.link(model_path, latest_path)

        if is_best:
            best_path = self.base_folder / 'model.best.pt'
            if best_path.exists():
                os.unlink(latest_path)
            os.link(model_path, best_path)

        return model_path.as_posix()

    def save_optimizer(self, optimizer, suffix=None):
        name = 'optimizer.pt' if suffix is None else f'optimizer.{suffix}.pt'
        optimizer_path = self.checkpoint_folder / name
        torch.save(optimizer.state_dict(), optimizer_path)

        latest_path = self.base_folder / 'optimizer.latest.pt'
        if latest_path.exists():
            os.unlink(latest_path)
        os.link(optimizer_path, latest_path)

        return optimizer_path.as_posix()

    def save_experiment(self, experiment, suffix=None):
        name = 'experiment.yaml' if suffix is None else f'experiment.{suffix}.yaml'
        experiment_path = self.checkpoint_folder / name
        with open(experiment_path, 'w') as f:
            pyaml.dump(experiment, f, safe=True, sort_dicts=False)

        latest_path = self.base_folder / 'experiment.latest.yaml'
        if latest_path.exists():
            os.unlink(latest_path)
        os.link(experiment_path, latest_path)

        return experiment_path.as_posix()

    def save(self, model, experiment, optimizer, suffix=None, is_best=False):
        experiment.model.state_dict = self.save_model(model, suffix=suffix, is_best=is_best)
        experiment.optimizer.state_dict = self.save_optimizer(optimizer, suffix=suffix)
        return {
            'model': experiment.model.state_dict,
            'optimizer': experiment.optimizer.state_dict,
            'experiment': self.save_experiment(experiment, suffix=suffix)
        }
