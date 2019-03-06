import argparse
from pathlib import Path

import tensorflow as tf

from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2

layout_summary = summary_lib.custom_scalar_pb(layout_pb2.Layout(category=[
    layout_pb2.Category(
        title='Losses',
        chart=[
            layout_pb2.Chart(
                title='Train', multiline=layout_pb2.MultilineChartContent(tag=['loss/train/mse', 'loss/train/l1'])),
            layout_pb2.Chart(
                title='Val', multiline=layout_pb2.MultilineChartContent(tag=['loss/train/mse', 'loss/val/mse'])),
        ])
]))

parser = argparse.ArgumentParser()
parser.add_argument('folder', help='The log folder to place the layout in')
args = parser.parse_args()

folder = (Path(args.folder) / 'layout').expanduser().resolve()
with tf.summary.FileWriter(folder) as writer:
    writer.add_summary(layout_summary)

print('Layout saved to', folder)
