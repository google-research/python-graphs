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

r"""Create program graph visualizations for the test components.


Usage:
python -m python_graphs.program_graph_visualizer
"""

import inspect

from absl import app
from absl import logging  # pylint: disable=unused-import

from python_graphs import control_flow_test_components as tc
from python_graphs import program_graph
from python_graphs import program_graph_graphviz


def render_functions(functions):
  for name, function in functions:
    logging.info(name)
    graph = program_graph.get_program_graph(function)
    path = '/tmp/program_graphs/{}.png'.format(name)
    program_graph_graphviz.render(graph, path=path)


def main(argv):
  del argv  # Unused.

  functions = [
      (name, fn)
      for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction)
  ]
  render_functions(functions)


if __name__ == '__main__':
  app.run(main)
