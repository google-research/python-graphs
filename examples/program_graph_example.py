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

"""Example generating a complete program graph from a Python function.

Generates an image visualizing the complete program graph for each function
in program_graph_test_components.py. Saves the resulting images to the directory
`out`.

Usage:
python -m examples.program_graph_example
"""

import inspect
import os

from absl import app
from python_graphs import program_graph
from python_graphs import program_graph_graphviz
from python_graphs import program_graph_test_components as tc


def main(argv) -> None:
  del argv  # Unused

  # Create the output directory.
  os.makedirs('out', exist_ok=True)

  # For each function in program_graph_test_components.py, visualize its
  # program graph. Save the results in the output directory.
  for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
    path = f'out/{name}-program-graph.png'
    graph = program_graph.get_program_graph(fn)
    program_graph_graphviz.render(graph, path=path)
  print('Done. See the `out` directory for the results.')


if __name__ == '__main__':
  app.run(main)
