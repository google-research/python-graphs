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

"""Example generating a control flow graph from a Python function.

Generates an image visualizing the control flow graph for each of the functions
in control_flow_test_components.py. Saves the resulting images to the directory
`out`.

Usage:
python -m examples.control_flow_example
"""

import inspect
import os

from absl import app

from python_graphs import control_flow
from python_graphs import control_flow_graphviz
from python_graphs import control_flow_test_components as tc
from python_graphs import program_utils


def plot_control_flow_graph(fn, path):
  graph = control_flow.get_control_flow_graph(fn)
  source = program_utils.getsource(fn)
  control_flow_graphviz.render(graph, include_src=source, path=path)


def main(argv) -> None:
  del argv  # Unused

  # Create the output directory.
  os.makedirs('out', exist_ok=True)

  # For each function in control_flow_test_components.py, visualize its
  # control flow graph. Save the results in the output directory.
  for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
    path = f'out/{name}_cfg.png'
    plot_control_flow_graph(fn, path)
  print('Done. See the `out` directory for the results.')


if __name__ == '__main__':
  app.run(main)
