from collections import Mapping, defaultdict
from pathlib import Path
from typing import Union

import yaml
import munch


class Config(munch.Munch):

    def __setattr__(self, key, value):
        if isinstance(value, Mapping):
            value = Config.fromDict(value)
        super(Config, self).__setattr__(key, value)

    @staticmethod
    def build(*new_configs, **cfg_args):
        c = Config()
        for new_config in new_configs:
            if isinstance(new_config, Mapping):
                Config._update_rec(c, new_config)
            elif isinstance(new_config, str) and '=' in new_config:
                Config._update_rec(c, Config.from_dotted(new_config))
            elif Path(new_config).suffix in {'.yml', '.yaml'}:
                Config._update_rec(c, Config.from_yaml(new_config))
            elif Path(new_config).suffix == '.json':
                Config._update_rec(c, Config.from_json(new_config))
        Config._update_rec(c, cfg_args)
        return c

    @staticmethod
    def from_yaml(file: Union[str, Path]):
        with open(file, 'r') as f:
            return Config.fromDict(yaml.safe_load(f))

    @staticmethod
    def from_json(file: Union[str, Path]):
        import json
        with open(file, 'r') as f:
            return Config.fromDict(json.load(f))

    @staticmethod
    def from_dotted(dotted_str: str):
        """Parse a string of named arguments that use dots to indicate hierarchy, e.g. `name=test opts.cpus=4`
        """

        def recursively_defaultdict():
            return defaultdict(recursively_defaultdict)

        config = recursively_defaultdict()

        for name_dotted, value in (pair.split('=') for pair in dotted_str.split(' ')):
            c = config
            name_head, *name_rest = name_dotted.lstrip('-').split('.')
            while len(name_rest) > 0:
                c = c[name_head]
                name_head, *name_rest = name_rest
            c[name_head] = yaml.safe_load(value)
        return Config.fromDict(config)

    @staticmethod
    def _update_rec(old_config, new_config):
        for k in new_config.keys():
            if k in old_config and isinstance(old_config[k], Mapping) and isinstance(new_config[k], Mapping):
                Config._update_rec(old_config[k], new_config[k])
            else:
                setattr(old_config, k, new_config[k])
