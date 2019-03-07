import contextlib
from collections import Mapping, defaultdict

import yaml
from munch import Munch


class Config(Munch):
    def update_from_yaml(self, filename: str):
        with open(filename, mode='r') as f:
            config = Munch.fromDict(yaml.safe_load(f))
        self.update_from(config)

    def update_from_cli(self, cli: str):
        """Update the current config from a string of named arguments.

        Accepted formats:
        - opts.device cpu -> updates config.opts.device = 'cpu'
        - training.save no -> updates config.training.save = False
        - training.epochs=10 -> updates config.training.epochs = 10
        """

        def recursively_defaultdict():
            return defaultdict(recursively_defaultdict)

        config = recursively_defaultdict()

        cli = cli.replace('=', ' ').split()
        for name_dotted, value in zip(cli[::2], cli[1::2]):
            c = config
            name_head, *name_rest = name_dotted.split('.')
            while len(name_rest) > 0:
                c = c[name_head]
                name_head, *name_rest = name_rest
            c[name_head] = yaml.safe_load(value)

        self.update_from(config)

    def update_from(self, new_config):
        Config._update_rec(self, new_config)
        Config._config_git(self)
        Config._config_session(self)
        Config._config_hardware(self)

    @staticmethod
    def _update_rec(old_config, new_config):
        new_config = Munch.fromDict(new_config)
        for k in new_config.keys():
            if k in old_config and isinstance(old_config[k], Mapping) and isinstance(new_config[k], Mapping):
                Config._update_rec(old_config[k], new_config[k])
            else:
                old_config[k] = new_config[k]

    @staticmethod
    def _config_hardware(config):
        if 'opts' in config:
            if 'cpus' in config.opts and config.opts.cpus == '_auto_':
                import multiprocessing
                config.opts.cpus = multiprocessing.cpu_count() - 1
            if 'device' in config.opts and config.opts.device == '_auto_':
                import torch
                config.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _config_git(config):
        if 'opts' in config:
            with contextlib.suppress(ImportError):
                import git
                with contextlib.suppress(git.InvalidGitRepositoryError):
                    repo = git.Repo(search_parent_directories=True)
                    config.opts.commit = repo.head.object.hexsha

    @staticmethod
    def _config_session(config):
        if 'opts' in config:
            if 'session' in config.opts and config.opts.session == '_auto_':
                import arrow
                import socket
                config.opts.session = f'{socket.gethostname()}_{arrow.utcnow()}'
            if 'seed' in config.opts and config.opts.seed == '_auto_':
                import random
                config.opts.seed = random.randint(0, 100)
