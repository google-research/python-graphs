# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Create control flow graph visualizations for the test components.


Usage:
python -m python_graphs.control_flow_visualizer
"""

import inspect
import os

from absl import app
from absl import flags
from absl import logging  # pylint: disable=unused-import

from python_graphs import control_flow
from python_graphs import control_flow_graphviz
from python_graphs import control_flow_test_components as tc
from python_graphs import program_utils

FLAGS = flags.FLAGS


def render_functions(functions):
  for name, function in functions:
    logging.info(name)
    graph = control_flow.get_control_flow_graph(function)
    path = '/tmp/control_flow_graphs/{}.png'.format(name)
    source = program_utils.getsource(function)  # pylint: disable=protected-access
    control_flow_graphviz.render(graph, include_src=source, path=path)


def render_filepaths(filepaths):
  for filepath in filepaths:
    filename = os.path.basename(filepath).split('.')[0]
    logging.info(filename)
    with open(filepath, 'r') as f:
      source = f.read()
    graph = control_flow.get_control_flow_graph(source)
    path = '/tmp/control_flow_graphs/{}.png'.format(filename)
    control_flow_graphviz.render(graph, include_src=source, path=path)


def main(argv):
  del argv  # Unused.

  functions = [
      (name, fn)
      for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction)
  ]
  render_functions(functions)

  # Add filepaths here to visualize their functions.
  filepaths = [
      __file__,
  ]
  render_filepaths(filepaths)


if __name__ == '__main__':
  app.run(main)
