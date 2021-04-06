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

"""Tests for control_flow_graphviz.py."""

import inspect

from absl.testing import absltest
from python_graphs import control_flow
from python_graphs import control_flow_graphviz
from python_graphs import control_flow_test_components as tc


class ControlFlowGraphvizTest(absltest.TestCase):

  def test_to_graphviz_for_all_test_components(self):
    for unused_name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
      graph = control_flow.get_control_flow_graph(fn)
      control_flow_graphviz.to_graphviz(graph)

  def test_get_label_multi_op_expression(self):
    graph = control_flow.get_control_flow_graph(tc.multi_op_expression)
    block = graph.get_block_by_source('1 + 2 * 3')
    self.assertEqual(
        control_flow_graphviz.get_label(block).strip(),
        'return (1 + (2 * 3))\\l')


if __name__ == '__main__':
  absltest.main()
