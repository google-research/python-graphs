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

"""Tests for data_flow.py."""

import inspect

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest
import gast as ast

from python_graphs import control_flow
from python_graphs import control_flow_test_components as tc
from python_graphs import data_flow
from python_graphs import program_utils


class DataFlowTest(absltest.TestCase):

  def test_get_while_loop_variables(self):
    root = program_utils.program_to_ast(tc.nested_while_loops)
    graph = control_flow.get_control_flow_graph(root)

    # node = graph.get_ast_node_by_type(ast.While)
    # TODO(dbieber): data_flow.get_while_loop_variables(node, graph)

    analysis = data_flow.LivenessAnalysis()
    for block in graph.get_exit_blocks():
      analysis.visit(block)

    for block in graph.get_blocks_by_ast_node_type_and_label(
        ast.While, 'test_block'):
      logging.info(block.get_label('liveness_out'))

  def test_liveness_simple_while_loop(self):
    def simple_while_loop():
      a = 2
      b = 10
      x = 1
      while x < b:
        tmp = x + a
        x = tmp + 1

    program_node = program_utils.program_to_ast(simple_while_loop)
    graph = control_flow.get_control_flow_graph(program_node)

    # TODO(dbieber): Use unified query system.
    while_node = [
        node for node in ast.walk(program_node)
        if isinstance(node, ast.While)][0]
    loop_variables = data_flow.get_while_loop_variables(while_node, graph)
    self.assertEqual(loop_variables, {'x'})

  def test_data_flow_nested_loops(self):
    def fn():
      count = 0
      for x in range(10):
        for y in range(10):
          if x == y:
            count += 1
      return count

    program_node = program_utils.program_to_ast(fn)
    graph = control_flow.get_control_flow_graph(program_node)

    # Perform the analysis.
    analysis = data_flow.LastAccessAnalysis()
    analysis.visit(graph.start_block.control_flow_nodes[0])
    for node in graph.get_enter_control_flow_nodes():
      analysis.visit(node)

    # Verify correctness.
    node = graph.get_control_flow_node_by_source('count += 1')
    last_accesses_in = node.get_label('last_access_in')
    last_accesses_out = node.get_label('last_access_out')
    self.assertLen(last_accesses_in['write-count'], 2)  # += 1, = 0
    self.assertLen(last_accesses_in['read-count'], 1)  # += 1
    self.assertLen(last_accesses_out['write-count'], 1)  # += 1
    self.assertLen(last_accesses_out['read-count'], 1)  # += 1

  def test_last_accesses_analysis(self):
    root = program_utils.program_to_ast(tc.nested_while_loops)
    graph = control_flow.get_control_flow_graph(root)

    analysis = data_flow.LastAccessAnalysis()
    analysis.visit(graph.start_block.control_flow_nodes[0])

    for node in graph.get_enter_control_flow_nodes():
      analysis.visit(node)

    for block in graph.blocks:
      for cfn in block.control_flow_nodes:
        self.assertTrue(cfn.has_label('last_access_in'))
        self.assertTrue(cfn.has_label('last_access_out'))

    node = graph.get_control_flow_node_by_source('y += 5')
    last_accesses = node.get_label('last_access_out')
    # TODO(dbieber): Add asserts that these are the correct accesses.
    self.assertLen(last_accesses['write-x'], 2)  # x = 1, x += 6
    self.assertLen(last_accesses['read-x'], 1)  # x < 2

    node = graph.get_control_flow_node_by_source('return x')
    last_accesses = node.get_label('last_access_out')
    self.assertLen(last_accesses['write-x'], 2)  # x = 1, x += 6
    self.assertLen(last_accesses['read-x'], 1)  # x < 2

  def test_liveness_analysis_all_test_components(self):
    for unused_name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
      root = program_utils.program_to_ast(fn)
      graph = control_flow.get_control_flow_graph(root)

      analysis = data_flow.LivenessAnalysis()
      for block in graph.get_exit_blocks():
        analysis.visit(block)

  def test_last_access_analysis_all_test_components(self):
    for unused_name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
      root = program_utils.program_to_ast(fn)
      graph = control_flow.get_control_flow_graph(root)

      analysis = data_flow.LastAccessAnalysis()
      for node in graph.get_enter_control_flow_nodes():
        analysis.visit(node)


if __name__ == '__main__':
  absltest.main()
