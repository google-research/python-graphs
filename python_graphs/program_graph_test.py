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

"""Tests for program_graph.py."""

import collections
import inspect
import time

from absl import logging
from absl.testing import absltest
import gast as ast

from python_graphs import control_flow_test_components as cftc
from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb
from python_graphs import program_graph_test_components as pgtc
from python_graphs import program_utils


def test_components():
  """Generates functions from two sets of test components.

  Yields:
    Functions from the program graph and control flow test components files.
  """
  for unused_name, fn in inspect.getmembers(pgtc, predicate=inspect.isfunction):
    yield fn

  for unused_name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
    yield fn


class ProgramGraphTest(absltest.TestCase):

  def assertEdge(self, graph, n1, n2, edge_type):
    """Asserts that an edge of type edge_type exists from n1 to n2 in graph."""
    edge = pb.Edge(id1=n1.id, id2=n2.id, type=edge_type)
    self.assertIn(edge, graph.edges)

  def assertNoEdge(self, graph, n1, n2, edge_type):
    """Asserts that no edge of type edge_type exists from n1 to n2 in graph."""
    edge = pb.Edge(id1=n1.id, id2=n2.id, type=edge_type)
    self.assertNotIn(edge, graph.edges)

  def test_get_program_graph_test_components(self):
    self.analyze_get_program_graph(test_components(), start=0)

  def analyze_get_program_graph(self, program_generator, start=0):
    # TODO(dbieber): Remove the counting and logging logic from this method,
    # and instead just get_program_graph for each program in the generator.
    # The counting and logging logic is for development purposes only.
    num_edges = 0
    num_edges_by_type = collections.defaultdict(int)
    num_nodes = 0
    num_graphs = 1
    times = {}
    for index, program in enumerate(program_generator):
      if index < start:
        continue
      start_time = time.time()
      graph = program_graph.get_program_graph(program)
      end_time = time.time()
      times[index] = end_time - start_time
      num_edges += len(graph.edges)
      for edge in graph.edges:
        num_edges_by_type[edge.type] += 1
      num_nodes += len(graph.nodes)
      num_graphs += 1
      if index % 100 == 0:
        logging.debug(sorted(times.items(), key=lambda kv: -kv[1])[:10])
    logging.info('%d %d %d', num_edges, num_nodes, num_graphs)
    logging.info('%f %f', num_edges / num_graphs, num_nodes / num_graphs)
    for edge_type in num_edges_by_type:
      logging.info('%s %f', edge_type,
                   num_edges_by_type[edge_type] / num_graphs)

    logging.info(times)
    logging.info(sorted(times.items(), key=lambda kv: -kv[1])[:10])

  def test_last_lexical_use_edges_function_call(self):
    graph = program_graph.get_program_graph(pgtc.function_call)
    read = graph.get_node_by_source_and_identifier('return z', 'z')
    write = graph.get_node_by_source_and_identifier(
        'z = function_call_helper(x, y)', 'z')
    self.assertEdge(graph, read, write, pb.EdgeType.LAST_LEXICAL_USE)

  def test_last_write_edges_function_call(self):
    graph = program_graph.get_program_graph(pgtc.function_call)
    write_z = graph.get_node_by_source_and_identifier(
        'z = function_call_helper(x, y)', 'z')
    read_z = graph.get_node_by_source_and_identifier('return z', 'z')
    self.assertEdge(graph, read_z, write_z, pb.EdgeType.LAST_WRITE)

    write_y = graph.get_node_by_source_and_identifier('y = 2', 'y')
    read_y = graph.get_node_by_source_and_identifier(
        'z = function_call_helper(x, y)', 'y')
    self.assertEdge(graph, read_y, write_y, pb.EdgeType.LAST_WRITE)

  def test_last_read_edges_assignments(self):
    graph = program_graph.get_program_graph(pgtc.assignments)
    write_a0 = graph.get_node_by_source_and_identifier('a, b = 0, 0', 'a')
    read_a0 = graph.get_node_by_source_and_identifier('c = 2 * a + 1', 'a')
    write_a1 = graph.get_node_by_source_and_identifier('a = c + 3', 'a')
    self.assertEdge(graph, write_a1, read_a0, pb.EdgeType.LAST_READ)
    self.assertNoEdge(graph, write_a0, read_a0, pb.EdgeType.LAST_READ)

    read_a1 = graph.get_node_by_source_and_identifier('return a, b, c, d', 'a')
    self.assertEdge(graph, read_a1, read_a0, pb.EdgeType.LAST_READ)

  def test_last_read_last_write_edges_repeated_identifier(self):
    graph = program_graph.get_program_graph(pgtc.repeated_identifier)
    write_x0 = graph.get_node_by_source_and_identifier('x = 0', 'x')

    stmt1 = graph.get_node_by_source('x = x + 1').ast_node
    read_x0 = graph.get_node_by_ast_node(stmt1.value.left)
    write_x1 = graph.get_node_by_ast_node(stmt1.targets[0])

    stmt2 = graph.get_node_by_source('x = (x + (x + x)) + x').ast_node
    read_x1 = graph.get_node_by_ast_node(stmt2.value.left.left)
    read_x2 = graph.get_node_by_ast_node(stmt2.value.left.right.left)
    read_x3 = graph.get_node_by_ast_node(stmt2.value.left.right.right)
    read_x4 = graph.get_node_by_ast_node(stmt2.value.right)
    write_x2 = graph.get_node_by_ast_node(stmt2.targets[0])

    read_x5 = graph.get_node_by_source_and_identifier('return x', 'x')

    self.assertEdge(graph, write_x1, read_x0, pb.EdgeType.LAST_READ)
    self.assertEdge(graph, read_x1, read_x0, pb.EdgeType.LAST_READ)
    self.assertEdge(graph, read_x2, read_x1, pb.EdgeType.LAST_READ)
    self.assertEdge(graph, read_x3, read_x2, pb.EdgeType.LAST_READ)
    self.assertEdge(graph, read_x4, read_x3, pb.EdgeType.LAST_READ)
    self.assertEdge(graph, write_x2, read_x4, pb.EdgeType.LAST_READ)
    self.assertEdge(graph, read_x5, read_x4, pb.EdgeType.LAST_READ)

    self.assertEdge(graph, read_x0, write_x0, pb.EdgeType.LAST_WRITE)
    self.assertEdge(graph, write_x1, write_x0, pb.EdgeType.LAST_WRITE)
    self.assertEdge(graph, read_x2, write_x1, pb.EdgeType.LAST_WRITE)
    self.assertEdge(graph, read_x3, write_x1, pb.EdgeType.LAST_WRITE)
    self.assertEdge(graph, read_x4, write_x1, pb.EdgeType.LAST_WRITE)
    self.assertEdge(graph, write_x2, write_x1, pb.EdgeType.LAST_WRITE)
    self.assertEdge(graph, read_x5, write_x2, pb.EdgeType.LAST_WRITE)

  def test_computed_from_edges(self):
    graph = program_graph.get_program_graph(pgtc.assignments)
    target_c = graph.get_node_by_source_and_identifier('c = 2 * a + 1', 'c')
    from_a = graph.get_node_by_source_and_identifier('c = 2 * a + 1', 'a')
    self.assertEdge(graph, target_c, from_a, pb.EdgeType.COMPUTED_FROM)

    target_d = graph.get_node_by_source_and_identifier('d = b - c + 2', 'd')
    from_b = graph.get_node_by_source_and_identifier('d = b - c + 2', 'b')
    from_c = graph.get_node_by_source_and_identifier('d = b - c + 2', 'c')
    self.assertEdge(graph, target_d, from_b, pb.EdgeType.COMPUTED_FROM)
    self.assertEdge(graph, target_d, from_c, pb.EdgeType.COMPUTED_FROM)

  def test_calls_edges(self):
    graph = program_graph.get_program_graph(pgtc)
    call = graph.get_node_by_source('function_call_helper(x, y)')
    self.assertIsInstance(call.node, ast.Call)
    function_call_helper_def = graph.get_node_by_function_name(
        'function_call_helper')
    assignments_def = graph.get_node_by_function_name('assignments')
    self.assertEdge(graph, call, function_call_helper_def, pb.EdgeType.CALLS)
    self.assertNoEdge(graph, call, assignments_def, pb.EdgeType.CALLS)

  def test_formal_arg_name_edges(self):
    graph = program_graph.get_program_graph(pgtc)
    x = graph.get_node_by_source_and_identifier('function_call_helper(x, y)',
                                                'x')
    y = graph.get_node_by_source_and_identifier('function_call_helper(x, y)',
                                                'y')
    function_call_helper_def = graph.get_node_by_function_name(
        'function_call_helper')
    arg0_ast_node = function_call_helper_def.node.args.args[0]
    arg0 = graph.get_node_by_ast_node(arg0_ast_node)
    arg1_ast_node = function_call_helper_def.node.args.args[1]
    arg1 = graph.get_node_by_ast_node(arg1_ast_node)
    self.assertEdge(graph, x, arg0, pb.EdgeType.FORMAL_ARG_NAME)
    self.assertEdge(graph, y, arg1, pb.EdgeType.FORMAL_ARG_NAME)
    self.assertNoEdge(graph, x, arg1, pb.EdgeType.FORMAL_ARG_NAME)
    self.assertNoEdge(graph, y, arg0, pb.EdgeType.FORMAL_ARG_NAME)

  def test_returns_to_edges(self):
    graph = program_graph.get_program_graph(pgtc)
    call = graph.get_node_by_source('function_call_helper(x, y)')
    return_stmt = graph.get_node_by_source('return arg0 + arg1')
    self.assertEdge(graph, return_stmt, call, pb.EdgeType.RETURNS_TO)

  def test_syntax_information(self):
    # TODO(dbieber): Test that program graphs correctly capture syntax
    # information. Do this once representation of syntax in program graphs
    # stabilizes.
    pass

  def test_ast_acyclic(self):
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
      graph = program_graph.get_program_graph(fn)
      ast_nodes = set()
      worklist = [graph.root]
      while worklist:
        current = worklist.pop()
        self.assertNotIn(
            current, ast_nodes,
            'ProgramGraph AST cyclic. Function {}\nAST {}'.format(
                name, graph.dump_tree()))
        ast_nodes.add(current)
        worklist.extend(graph.children(current))

  def test_neighbors_children_consistent(self):
    for unused_name, fn in inspect.getmembers(
        cftc, predicate=inspect.isfunction):
      graph = program_graph.get_program_graph(fn)
      for node in graph.all_nodes():
        if node.node_type == pb.NodeType.AST_NODE:
          children0 = set(graph.outgoing_neighbors(node, pb.EdgeType.FIELD))
          children1 = set(graph.children(node))
          self.assertEqual(children0, children1)

  def test_walk_ast_descendants(self):
    for unused_name, fn in inspect.getmembers(
        cftc, predicate=inspect.isfunction):
      graph = program_graph.get_program_graph(fn)
      for node in graph.walk_ast_descendants():
        self.assertIn(node, graph.all_nodes())

  def test_roundtrip_ast(self):
    for unused_name, fn in inspect.getmembers(
        cftc, predicate=inspect.isfunction):
      ast_representation = program_utils.program_to_ast(fn)
      graph = program_graph.get_program_graph(fn)
      ast_reproduction = graph.to_ast()
      self.assertEqual(ast.dump(ast_representation), ast.dump(ast_reproduction))

  def test_reconstruct_missing_ast(self):
    for unused_name, fn in inspect.getmembers(
        cftc, predicate=inspect.isfunction):
      graph = program_graph.get_program_graph(fn)
      ast_original = graph.root.ast_node
      # Remove the AST.
      for node in graph.all_nodes():
        node.ast_node = None
      # Reconstruct it.
      graph.reconstruct_ast()
      ast_reproduction = graph.root.ast_node
      # Check reconstruction.
      self.assertEqual(ast.dump(ast_original), ast.dump(ast_reproduction))
      # Check that all AST_NODE nodes are set.
      for node in graph.all_nodes():
        if node.node_type == pb.NodeType.AST_NODE:
          self.assertIsInstance(node.ast_node, ast.AST)
          self.assertIs(graph.get_node_by_ast_node(node.ast_node), node)
      # Check that old AST nodes are no longer referenced.
      self.assertFalse(graph.contains_ast_node(ast_original))

  def test_remove(self):
    graph = program_graph.get_program_graph(pgtc.assignments)

    for edge in list(graph.edges)[:]:
      # Remove the edge.
      graph.remove_edge(edge)
      self.assertNotIn(edge, graph.edges)
      self.assertNotIn((edge, edge.id2), graph.neighbors_map[edge.id1])
      self.assertNotIn((edge, edge.id1), graph.neighbors_map[edge.id2])

      if edge.type == pb.EdgeType.FIELD:
        self.assertNotIn(edge.id2, graph.child_map[edge.id1])
        self.assertNotIn(edge.id2, graph.parent_map)

      # Add the edge again.
      graph.add_edge(edge)
      self.assertIn(edge, graph.edges)
      self.assertIn((edge, edge.id2), graph.neighbors_map[edge.id1])
      self.assertIn((edge, edge.id1), graph.neighbors_map[edge.id2])

      if edge.type == pb.EdgeType.FIELD:
        self.assertIn(edge.id2, graph.child_map[edge.id1])
        self.assertIn(edge.id2, graph.parent_map)


if __name__ == '__main__':
  absltest.main()
