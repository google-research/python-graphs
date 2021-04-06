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

"""Tests for control_flow.py."""

import inspect

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest
import gast as ast
from python_graphs import control_flow
from python_graphs import control_flow_test_components as tc
from python_graphs import instruction as instruction_module
from python_graphs import program_utils
import six


class ControlFlowTest(absltest.TestCase):

  def get_block(self, graph, selector):
    if isinstance(selector, control_flow.BasicBlock):
      return selector
    elif isinstance(selector, six.string_types):
      return graph.get_block_by_source(selector)

  def assertSameBlock(self, graph, selector1, selector2):
    block1 = self.get_block(graph, selector1)
    block2 = self.get_block(graph, selector2)
    self.assertEqual(block1, block2)

  def assertExitsTo(self, graph, selector1, selector2):
    block1 = self.get_block(graph, selector1)
    block2 = self.get_block(graph, selector2)
    self.assertTrue(block1.exits_to(block2))

  def assertNotExitsTo(self, graph, selector1, selector2):
    block1 = self.get_block(graph, selector1)
    block2 = self.get_block(graph, selector2)
    self.assertFalse(block1.exits_to(block2))

  def assertRaisesTo(self, graph, selector1, selector2):
    block1 = self.get_block(graph, selector1)
    block2 = self.get_block(graph, selector2)
    self.assertTrue(block1.raises_to(block2))

  def assertNotRaisesTo(self, graph, selector1, selector2):
    block1 = self.get_block(graph, selector1)
    block2 = self.get_block(graph, selector2)
    self.assertFalse(block1.raises_to(block2))

  def test_control_flow_straight_line_code(self):
    graph = control_flow.get_control_flow_graph(tc.straight_line_code)
    self.assertSameBlock(graph, 'x = 1', 'y = x + 2')
    self.assertSameBlock(graph, 'x = 1', 'z = y * 3')
    self.assertSameBlock(graph, 'x = 1', 'return z')

  def test_control_flow_simple_if_statement(self):
    graph = control_flow.get_control_flow_graph(tc.simple_if_statement)
    x1_block = 'x = 1'
    y2_block = 'y = 2'
    xy_block = 'x > y'
    y3_block = 'y = 3'
    return_block = 'return y'
    self.assertSameBlock(graph, x1_block, y2_block)
    self.assertSameBlock(graph, x1_block, xy_block)
    self.assertExitsTo(graph, xy_block, y3_block)
    self.assertExitsTo(graph, xy_block, return_block)
    self.assertExitsTo(graph, y3_block, return_block)
    self.assertNotExitsTo(graph, y3_block, x1_block)
    self.assertNotExitsTo(graph, return_block, x1_block)
    self.assertNotExitsTo(graph, return_block, y3_block)

  def test_control_flow_simple_for_loop(self):
    graph = control_flow.get_control_flow_graph(tc.simple_for_loop)
    x1_block = 'x = 1'
    iter_block = 'range'
    target_block = 'y'
    body_block = 'y + 3'
    return_block = 'return z'
    self.assertSameBlock(graph, x1_block, iter_block)
    self.assertExitsTo(graph, iter_block, target_block)
    self.assertExitsTo(graph, target_block, body_block)
    self.assertNotExitsTo(graph, body_block, return_block)
    self.assertExitsTo(graph, target_block, return_block)

  def test_control_flow_simple_while_loop(self):
    graph = control_flow.get_control_flow_graph(tc.simple_while_loop)
    x1_block = 'x = 1'
    test_block = 'x < 2'
    body_block = 'x += 3'
    return_block = 'return x'

    self.assertExitsTo(graph, x1_block, test_block)
    self.assertExitsTo(graph, test_block, body_block)
    self.assertExitsTo(graph, body_block, test_block)
    self.assertNotExitsTo(graph, body_block, return_block)
    self.assertExitsTo(graph, test_block, return_block)

  def test_control_flow_break_in_while_loop(self):
    graph = control_flow.get_control_flow_graph(tc.break_in_while_loop)
    # This is just one block since there's no edge from the while loop end
    # back to the while loop test, and so the 'x = 1' line can be merged with
    # the test.
    x1_and_test_block = 'x < 2'
    body_block = 'x += 3'
    return_block = 'return x'

    self.assertExitsTo(graph, x1_and_test_block, body_block)
    self.assertExitsTo(graph, body_block, return_block)
    self.assertNotExitsTo(graph, body_block, x1_and_test_block)
    self.assertExitsTo(graph, x1_and_test_block, return_block)

  def test_control_flow_nested_while_loops(self):
    graph = control_flow.get_control_flow_graph(tc.nested_while_loops)
    x1_block = 'x = 1'
    outer_test_block = 'x < 2'
    y3_block = 'y = 3'
    inner_test_block = 'y < 4'
    y5_block = 'y += 5'
    x6_block = 'x += 6'
    return_block = 'return x'

    self.assertExitsTo(graph, x1_block, outer_test_block)
    self.assertExitsTo(graph, outer_test_block, y3_block)
    self.assertExitsTo(graph, outer_test_block, return_block)
    self.assertExitsTo(graph, y3_block, inner_test_block)
    self.assertExitsTo(graph, inner_test_block, y5_block)
    self.assertExitsTo(graph, inner_test_block, x6_block)
    self.assertExitsTo(graph, y5_block, inner_test_block)
    self.assertExitsTo(graph, x6_block, outer_test_block)

  def test_control_flow_exception_handling(self):
    graph = control_flow.get_control_flow_graph(tc.exception_handling)
    self.assertSameBlock(graph, 'before_stmt0', 'before_stmt1')
    self.assertExitsTo(graph, 'before_stmt1', 'try_block')
    self.assertNotExitsTo(graph, 'before_stmt0', 'except_block1')
    self.assertNotExitsTo(graph, 'before_stmt1', 'final_block_stmt0')
    self.assertRaisesTo(graph, 'try_block', 'error_type')
    self.assertRaisesTo(graph, 'error_type', 'except_block2_stmt0')
    self.assertExitsTo(graph, 'except_block1', 'after_stmt0')

    self.assertRaisesTo(graph, 'after_stmt0', 'except_block2_stmt0')
    self.assertNotRaisesTo(graph, 'try_block', 'except_block2_stmt0')

  def test_control_flow_try_with_loop(self):
    graph = control_flow.get_control_flow_graph(tc.try_with_loop)
    self.assertSameBlock(graph, 'for_body0', 'for_body1')
    self.assertSameBlock(graph, 'except_body0', 'except_body1')

    self.assertExitsTo(graph, 'before_stmt0', 'iterator')
    self.assertExitsTo(graph, 'iterator', 'target')
    self.assertExitsTo(graph, 'target', 'for_body0')
    self.assertExitsTo(graph, 'for_body1', 'target')
    self.assertExitsTo(graph, 'target', 'after_stmt0')

    self.assertRaisesTo(graph, 'iterator', 'except_body0')
    self.assertRaisesTo(graph, 'target', 'except_body0')
    self.assertRaisesTo(graph, 'for_body1', 'except_body0')

  def test_control_flow_break_in_finally(self):
    graph = control_flow.get_control_flow_graph(tc.break_in_finally)

    # The exception handlers are tried sequentially until one matches.
    self.assertRaisesTo(graph, 'try0', 'Exception0')
    self.assertExitsTo(graph, 'Exception0', 'Exception1')
    self.assertExitsTo(graph, 'Exception1', 'finally_stmt0')
    # If the finally block were to finish and the exception hadn't matched, then
    # the exception would exit to the FunctionDef's raise_block. However, the
    # break statement prevents the finally from finishing and so the exception
    # is lost when the break statement is reached.
    # TODO(dbieber): Add the following assert.
    # raise_block = graph.get_raise_block('break_in_finally')
    # self.assertNotExitsFromEndTo(graph, 'finally_stmt1', raise_block)
    # The finally block can of course still raise an exception of its own, so
    # the following is still true:
    # TODO(dbieber): Add the following assert.
    # self.assertRaisesTo(graph, 'finally_stmt1', raise_block)

    # An exception in the except handlers could flow to the finally block.
    self.assertRaisesTo(graph, 'Exception0', 'finally_stmt0')
    self.assertRaisesTo(graph, 'exception0_stmt0', 'finally_stmt0')
    self.assertRaisesTo(graph, 'Exception1', 'finally_stmt0')

    # The break statement flows to after0, rather than to the loop header.
    self.assertNotExitsTo(graph, 'finally_stmt1', 'target0')
    self.assertExitsTo(graph, 'finally_stmt1', 'after0')

  def test_control_flow_for_loop_with_else(self):
    graph = control_flow.get_control_flow_graph(tc.for_with_else)
    self.assertExitsTo(graph, 'target', 'for_stmt0')
    self.assertSameBlock(graph, 'for_stmt0', 'condition')

    # If break is encountered, then the else clause is skipped.
    self.assertExitsTo(graph, 'condition', 'after_stmt0')

    # The else clause executes if the loop completes without reaching the break.
    self.assertExitsTo(graph, 'target', 'else_stmt0')
    self.assertNotExitsTo(graph, 'target', 'after_stmt0')

  def test_control_flow_lambda(self):
    graph = control_flow.get_control_flow_graph(tc.create_lambda)
    self.assertNotExitsTo(graph, 'before_stmt0', 'args')
    self.assertNotExitsTo(graph, 'before_stmt0', 'output')

  def test_control_flow_generator(self):
    graph = control_flow.get_control_flow_graph(tc.generator)
    self.assertExitsTo(graph, 'target', 'yield_statement')
    self.assertSameBlock(graph, 'yield_statement', 'after_stmt0')

  def test_control_flow_inner_fn_while_loop(self):
    graph = control_flow.get_control_flow_graph(tc.fn_with_inner_fn)
    self.assertExitsTo(graph, 'x = 10', 'True')
    self.assertExitsTo(graph, 'True', 'True')
    self.assertSameBlock(graph, 'True', 'True')

  def test_control_flow_example_class(self):
    graph = control_flow.get_control_flow_graph(tc.ExampleClass)
    self.assertSameBlock(graph, 'method_stmt0', 'method_stmt1')

  def test_control_flow_return_outside_function(self):
    with self.assertRaises(RuntimeError) as error:
      control_flow.get_control_flow_graph('return x')
    self.assertContainsSubsequence(str(error.exception),
                                   'outside of a function frame')

  def test_control_flow_continue_outside_loop(self):
    control_flow.get_control_flow_graph('for i in j: continue')
    with self.assertRaises(RuntimeError) as error:
      control_flow.get_control_flow_graph('if x: continue')
    self.assertContainsSubsequence(str(error.exception),
                                   'outside of a loop frame')

  def test_control_flow_break_outside_loop(self):
    control_flow.get_control_flow_graph('for i in j: break')
    with self.assertRaises(RuntimeError) as error:
      control_flow.get_control_flow_graph('if x: break')
    self.assertContainsSubsequence(str(error.exception),
                                   'outside of a loop frame')

  def test_control_flow_for_all_test_components(self):
    for unused_name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
      control_flow.get_control_flow_graph(fn)

  def test_control_flow_for_all_test_components_ast_to_instruction(self):
    """All INSTRUCTION_AST_NODES in an AST correspond to one Instruction.

    This assumes that a simple statement can't contain another simple statement.
    However, Yield nodes are the exception to this as they are contained within
    Expr nodes.

    We omit Yield nodes from INSTRUCTION_AST_NODES despite them being listed
    as simple statements in the Python docs.
    """
    for unused_name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
      node = program_utils.program_to_ast(fn)
      graph = control_flow.get_control_flow_graph(node)
      for n in ast.walk(node):
        if not isinstance(n, instruction_module.INSTRUCTION_AST_NODES):
          continue
        control_flow_nodes = list(graph.get_control_flow_nodes_by_ast_node(n))
        self.assertLen(control_flow_nodes, 1, ast.dump(n))

  def test_control_flow_reads_and_writes_appear_once(self):
    """Asserts each read and write in an Instruction is unique in the graph.

    Note that in the case of AugAssign, the same Name AST node is used once as
    a read and once as a write.
    """
    for unused_name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
      reads = set()
      writes = set()
      node = program_utils.program_to_ast(fn)
      graph = control_flow.get_control_flow_graph(node)
      for instruction in graph.get_instructions():
        # Check that all reads are unique.
        for read in instruction.get_reads():
          if isinstance(read, tuple):
            read = read[1]
          self.assertIsInstance(read, ast.Name, 'Unexpected read type.')
          self.assertNotIn(read, reads,
                           instruction_module.access_name(read))
          reads.add(read)

        # Check that all writes are unique.
        for write in instruction.get_writes():
          if isinstance(write, tuple):
            write = write[1]
          if isinstance(write, six.string_types):
            continue
          self.assertIsInstance(write, ast.Name)
          self.assertNotIn(write, writes,
                           instruction_module.access_name(write))
          writes.add(write)


if __name__ == '__main__':
  absltest.main()
