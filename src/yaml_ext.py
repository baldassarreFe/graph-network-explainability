import torch
import yaml

_init = False


class TorchDevice:
    _tag = '!torch.device'
    _class = torch.device

    @staticmethod
    def represent(dumper, device):
        return dumper.represent_scalar(TorchDevice._tag, str(device))

    @staticmethod
    def construct(loader, node):
        return torch.device(loader.construct_scalar(node))

    @classmethod
    def register(cls):
        yaml.add_representer(cls._class, cls.represent, yaml.SafeDumper)
        yaml.add_constructor(cls._tag, cls.construct, yaml.SafeLoader)


def init_ext():
    global _init
    if not _init:
        TorchDevice.register()
        _init = True
