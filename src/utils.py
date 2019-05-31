import itertools
import subprocess
import collections
from typing import Mapping, Iterable, MutableMapping

import yaml
from munch import munchify, AutoMunch


def git_info():
    try:
        import git
        try:
            result = {}
            repo = git.Repo(search_parent_directories=True)
            try:
                result['url'] = repo.remote(name='origin').url
            except ValueError:
                result['url'] = 'git:/' + repo.working_dir
            result['commit'] = repo.head.commit.hexsha
            result['dirty'] = repo.is_dirty()
            if repo.is_dirty():
                result['diffs'] = [str(diff) for diff in repo.head.commit.diff(other=None, create_patch=True)]
            if len(repo.untracked_files) > 0:
                result['untracked_files'] = repo.untracked_files
            return result
        except (git.InvalidGitRepositoryError, ValueError):
            pass
    except ImportError:
        return None


def cuda_info():
    from xml.etree import ElementTree
    try:
        nvidia_smi_xml = subprocess.check_output(['nvidia-smi', '-q', '-x']).decode()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None

    driver = ''
    gpus = []
    for child in ElementTree.fromstring(nvidia_smi_xml):
        if child.tag == 'driver_version':
            driver = child.text
        elif child.tag == 'gpu':
            gpus.append({
                'model': child.find('product_name').text,
                'utilization': child.find('utilization').find('gpu_util').text,
                'memory_used': child.find('fb_memory_usage').find('used').text,
                'memory_total': child.find('fb_memory_usage').find('total').text,
            })

    return {'driver': driver, 'gpus': gpus}


def parse_dotted(string):
    result_dict = {}
    for kv_pair in string.split(' '):
        sub_dict = result_dict
        name_dotted, value = kv_pair.split('=')
        name_head, *name_rest = name_dotted.split('.')
        while len(name_rest) > 0:
            sub_dict = sub_dict.setdefault(name_head, {})
            name_head, *name_rest = name_rest
        sub_dict[name_head] = yaml.safe_load(value)
    return result_dict


def update_rec(target, source):
    for k in source.keys():
        if k in target and isinstance(target[k], Mapping) and isinstance(source[k], Mapping):
            update_rec(target[k], source[k])
        else:
            # AutoMunch should do its job, but sometimes it doesn't
            target[k] = munchify(source[k], AutoMunch)


def import_(fullname):
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


def sort_dict(mapping: MutableMapping, order: Iterable):
    for key in itertools.chain(filter(mapping.__contains__, order), set(mapping) - set(order)):
        mapping[key] = mapping.pop(key)
    return mapping


class RunningWeightedAverage(object):
    def __init__(self):
        self.total_weight = 0
        self.total_weighted_value = 0

    def add(self, value, weight):
        if weight <= 0:
            raise ValueError()
        self.total_weighted_value += value * weight
        self.total_weight += weight

    def get(self):
        if self.total_weight == 0:
            return 0
        return self.total_weighted_value / self.total_weight

    def __repr__(self):
        return f'{self.get() (self.total_weight)}'
