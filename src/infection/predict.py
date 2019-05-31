# TODO this might be incompatible with recent code changes

import tqdm
import yaml
import pyaml
import multiprocessing
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from munch import AutoMunch, Munch

import torch
import torch.utils.data
import torch.nn.functional as F
import torchgraphs as tg

from utils import parse_dotted, update_rec, import_
from .dataset import InfectionDataset

parser = ArgumentParser()
parser.add_argument('--model', nargs='+', required=True)
parser.add_argument('--data', nargs='+', required=True, default=[])
parser.add_argument('--options', nargs='+', required=False, default=[])
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()


# region Collecting phase

# Defaults
model = Munch(fn=None, args=[], kwargs={}, state_dict=None)
data = []
options = AutoMunch()
options.cpus = multiprocessing.cpu_count() - 1
options.device = 'cuda' if torch.cuda.is_available() else 'cpu'
options.output = args.output

# Model from --model args
for string in args.model:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml file contains an entry with key `model` use that one instead
            if 'model' in update.keys():
                update = update['model']
    update_rec(model, update)

# Data from --data args
for path in args.data:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        data.extend(path.glob('*.pt'))
    elif path.is_file() and path.suffix == '.pt':
        data.append(path)
    else:
        raise ValueError(f'Invalid data: {path}')

# Options from --options args
for string in args.options:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
    update_rec(options, update)

# Resolving paths
model.state_dict = Path(model.state_dict).expanduser().resolve()
options.output = Path(options.output).expanduser().resolve()

# Checks (some missing, others redundant)
if model.fn is None:
    raise ValueError('Model constructor function not defined')
if model.state_dict is None:
    raise ValueError(f'Model state dict is required to predict')
if len(data) == 0:
    raise ValueError(f'No data to predict')
if options.cpus < 0:
    raise ValueError(f'Invalid number of cpus: {options.cpus}')
if options.output.exists() and not options.output.is_dir():
    raise ValueError(f'Invalid output path {options.output}')


pyaml.pprint({'model': model, 'options': options, 'data': data}, sort_dicts=False, width=200)
# endregion

# region Building phase
# Model
net: torch.nn.Module = import_(model.fn)(*model.args, **model.kwargs)
net.load_state_dict(torch.load(model.state_dict))
net.to(options.device)

# Output folder
options.output.mkdir(parents=True, exist_ok=True)
# endregion

# region Training
# Dataset and dataloader
dataset_predict: InfectionDataset = torch.load(data[0])
dataloader_predict = torch.utils.data.DataLoader(
    dataset_predict,
    shuffle=False,
    num_workers=min(options.cpus, 1) if 'cuda' in options.device else options.cpus,
    pin_memory='cuda' in options.device,
    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)),
    batch_size=options.batch_size,
    collate_fn=tg.GraphBatch.collate,
)

# region Predict
net.eval()
torch.set_grad_enabled(False)
i = 0
with tqdm.tqdm(desc='Predict', total=len(dataloader_predict.dataset), unit='g') as bar:
    for graphs, *_ in dataloader_predict:
        graphs = graphs.to(options.device)

        results = net(graphs)
        results.node_features.sigmoid_()

        for result in results:
            torch.save(result.cpu(), options.output / f'output_{i:06d}.pt')
            i += 1

        bar.update(graphs.num_graphs)
# endregion
