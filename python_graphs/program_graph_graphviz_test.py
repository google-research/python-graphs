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

"""Tests for program_graph_graphviz.py."""

import inspect

from absl.testing import absltest
from python_graphs import control_flow_test_components as tc
from python_graphs import program_graph
from python_graphs import program_graph_graphviz


class ControlFlowGraphvizTest(absltest.TestCase):

  def test_to_graphviz_for_all_test_components(self):
    for unused_name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
      graph = program_graph.get_program_graph(fn)
      program_graph_graphviz.to_graphviz(graph)


if __name__ == '__main__':
  absltest.main()
