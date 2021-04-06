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

"""Tests for program_graph_analysis.py."""

from absl.testing import absltest
import gast as ast
import networkx as nx

from python_graphs import program_graph
from python_graphs.analysis import program_graph_analysis as pga


class ProgramGraphAnalysisTest(absltest.TestCase):

  def setUp(self):
    super(ProgramGraphAnalysisTest, self).setUp()
    self.singleton = self.create_singleton_graph()
    self.disconnected = self.create_disconnected_graph()
    self.cycle_3 = self.create_cycle_3()
    self.chain_4 = self.create_chain_4()
    self.wide_tree = self.create_wide_tree()

  def create_singleton_graph(self):
    """Returns a graph with one node and zero edges."""
    graph = program_graph.ProgramGraph()
    node = program_graph.make_node_from_syntax('singleton_node')
    graph.add_node(node)
    graph.root_id = node.id
    return graph

  def create_disconnected_graph(self):
    """Returns a disconnected graph with two nodes and zero edges."""
    graph = program_graph.ProgramGraph()
    a = program_graph.make_node_from_syntax('a')
    b = program_graph.make_node_from_syntax('b')
    graph.add_node(a)
    graph.add_node(b)
    graph.root_id = a.id
    return graph

  def create_cycle_3(self):
    """Returns a 3-cycle graph, A -> B -> C -> A."""
    graph = program_graph.ProgramGraph()
    a = program_graph.make_node_from_syntax('A')
    b = program_graph.make_node_from_ast_value('B')
    c = program_graph.make_node_from_syntax('C')
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    graph.add_new_edge(a, b)
    graph.add_new_edge(b, c)
    graph.add_new_edge(c, a)
    graph.root_id = a.id
    return graph

  def create_chain_4(self):
    """Returns a chain of 4 nodes, A -> B -> C -> D."""
    graph = program_graph.ProgramGraph()
    a = program_graph.make_node_from_syntax('A')
    b = program_graph.make_node_from_ast_value('B')
    c = program_graph.make_node_from_syntax('C')
    d = program_graph.make_node_from_ast_value('D')
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    graph.add_node(d)
    graph.add_new_edge(a, b)
    graph.add_new_edge(b, c)
    graph.add_new_edge(c, d)
    graph.root_id = a.id
    return graph

  def create_wide_tree(self):
    """Returns a tree where the root has 4 children that are all leaves."""
    graph = program_graph.ProgramGraph()
    root = program_graph.make_node_from_syntax('root')
    graph.add_node(root)
    graph.root_id = root.id
    for i in range(4):
      leaf = program_graph.make_node_from_ast_value(i)
      graph.add_node(leaf)
      graph.add_new_edge(root, leaf)
    return graph

  def ids_from_cycle_3(self):
    """Returns a triplet of IDs from the 3-cycle graph in cycle order."""
    root = self.cycle_3.root
    id_a = root.id
    id_b = self.cycle_3.outgoing_neighbors(root)[0].id
    id_c = self.cycle_3.incoming_neighbors(root)[0].id
    return id_a, id_b, id_c

  def test_num_nodes_returns_expected(self):
    self.assertEqual(pga.num_nodes(self.singleton), 1)
    self.assertEqual(pga.num_nodes(self.disconnected), 2)
    self.assertEqual(pga.num_nodes(self.cycle_3), 3)
    self.assertEqual(pga.num_nodes(self.chain_4), 4)
    self.assertEqual(pga.num_nodes(self.wide_tree), 5)

  def test_num_edges_returns_expected(self):
    self.assertEqual(pga.num_edges(self.singleton), 0)
    self.assertEqual(pga.num_edges(self.disconnected), 0)
    self.assertEqual(pga.num_edges(self.cycle_3), 3)
    self.assertEqual(pga.num_edges(self.chain_4), 3)
    self.assertEqual(pga.num_edges(self.wide_tree), 4)

  def test_ast_height_returns_expected_for_constructed_expression_ast(self):
    # Testing the expression "1".
    # Height 3: Module -> Expr -> Num.
    ast_node = ast.Module(
        body=[ast.Expr(value=ast.Constant(value=1, kind=None))],
        type_ignores=[])
    self.assertEqual(pga.ast_height(ast_node), 3)

    # Testing the expression "1 + 1".
    # Height 4: Module -> Expr -> BinOp -> Num.
    ast_node = ast.Module(
        body=[
            ast.Expr(
                value=ast.BinOp(
                    left=ast.Constant(value=1, kind=None),
                    op=ast.Add(),
                    right=ast.Constant(value=1, kind=None)))
        ],
        type_ignores=[])
    self.assertEqual(pga.ast_height(ast_node), 4)

    # Testing the expression "a + 1".
    # Height 5: Module -> Expr -> BinOp -> Name -> Load.
    ast_node = ast.Module(
        body=[
            ast.Expr(
                value=ast.BinOp(
                    left=ast.Name(
                        id='a',
                        ctx=ast.Load(),
                        annotation=None,
                        type_comment=None),
                    op=ast.Add(),
                    right=ast.Constant(value=1, kind=None)))
        ],
        type_ignores=[])
    self.assertEqual(pga.ast_height(ast_node), 5)

    # Testing the expression "a.b + 1".
    # Height 6: Module -> Expr -> BinOp -> Attribute -> Name -> Load.
    ast_node = ast.Module(
        body=[
            ast.Expr(
                value=ast.BinOp(
                    left=ast.Attribute(
                        value=ast.Name(
                            id='a',
                            ctx=ast.Load(),
                            annotation=None,
                            type_comment=None),
                        attr='b',
                        ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(value=1, kind=None)))
        ],
        type_ignores=[])
    self.assertEqual(pga.ast_height(ast_node), 6)

  def test_ast_height_returns_expected_for_constructed_function_ast(self):
    # Testing the function declaration "def foo(n): return".
    # Height 5: Module -> FunctionDef -> arguments -> Name -> Param.
    ast_node = ast.Module(
        body=[
            ast.FunctionDef(
                name='foo',
                args=ast.arguments(
                    args=[
                        ast.Name(
                            id='n',
                            ctx=ast.Param(),
                            annotation=None,
                            type_comment=None)
                    ],
                    posonlyargs=[],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[]),
                body=[ast.Return(value=None)],
                decorator_list=[],
                returns=None,
                type_comment=None)
        ],
        type_ignores=[])
    self.assertEqual(pga.ast_height(ast_node), 5)

    # Testing the function declaration "def foo(n): return n + 1".
    # Height 6: Module -> FunctionDef -> Return -> BinOp -> Name -> Load.
    ast_node = ast.Module(
        body=[
            ast.FunctionDef(
                name='foo',
                args=ast.arguments(
                    args=[
                        ast.Name(
                            id='n',
                            ctx=ast.Param(),
                            annotation=None,
                            type_comment=None)
                    ],
                    posonlyargs=[],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[]),
                body=[
                    ast.Return(
                        value=ast.BinOp(
                            left=ast.Name(
                                id='n',
                                ctx=ast.Load(),
                                annotation=None,
                                type_comment=None),
                            op=ast.Add(),
                            right=ast.Constant(value=1, kind=None)))
                ],
                decorator_list=[],
                returns=None,
                type_comment=None)
        ],
        type_ignores=[],
    )
    self.assertEqual(pga.ast_height(ast_node), 6)

  def test_ast_height_returns_expected_for_parsed_ast(self):
    # Height 3: Module -> Expr -> Num.
    self.assertEqual(pga.ast_height(ast.parse('1')), 3)

    # Height 6: Module -> Expr -> BinOp -> Attribute -> Name -> Load.
    self.assertEqual(pga.ast_height(ast.parse('a.b + 1')), 6)

    # Height 6: Module -> FunctionDef -> Return -> BinOp -> Name -> Load.
    self.assertEqual(pga.ast_height(ast.parse('def foo(n): return n + 1')), 6)

    # Height 9: Module -> FunctionDef -> If -> Return -> BinOp -> Call
    #                  -> BinOp -> Name -> Load.
    # Adding whitespace before "def foo" causes an IndentationError in parse().
    ast_node = ast.parse("""def foo(n):
                              if n <= 0:
                                return 0
                              else:
                                return 1 + foo(n - 1)
                         """)
    self.assertEqual(pga.ast_height(ast_node), 9)

  def test_graph_ast_height_returns_expected(self):
    # Height 6: Module -> FunctionDef -> Return -> BinOp -> Name -> Load.
    def foo1(n):
      return n + 1

    graph = program_graph.get_program_graph(foo1)
    self.assertEqual(pga.graph_ast_height(graph), 6)

    # Height 9: Module -> FunctionDef -> If -> Return -> BinOp -> Call
    #                  -> BinOp -> Name -> Load.
    def foo2(n):
      if n <= 0:
        return 0
      else:
        return 1 + foo2(n - 1)

    graph = program_graph.get_program_graph(foo2)
    self.assertEqual(pga.graph_ast_height(graph), 9)

  def test_degrees_returns_expected(self):
    self.assertCountEqual(pga.degrees(self.singleton), [0])
    self.assertCountEqual(pga.degrees(self.disconnected), [0, 0])
    self.assertCountEqual(pga.degrees(self.cycle_3), [2, 2, 2])
    self.assertCountEqual(pga.degrees(self.chain_4), [1, 2, 2, 1])
    self.assertCountEqual(pga.degrees(self.wide_tree), [4, 1, 1, 1, 1])

  def test_in_degrees_returns_expected(self):
    self.assertCountEqual(pga.in_degrees(self.singleton), [0])
    self.assertCountEqual(pga.in_degrees(self.disconnected), [0, 0])
    self.assertCountEqual(pga.in_degrees(self.cycle_3), [1, 1, 1])
    self.assertCountEqual(pga.in_degrees(self.chain_4), [0, 1, 1, 1])
    self.assertCountEqual(pga.in_degrees(self.wide_tree), [0, 1, 1, 1, 1])

  def test_out_degrees_returns_expected(self):
    self.assertCountEqual(pga.out_degrees(self.singleton), [0])
    self.assertCountEqual(pga.out_degrees(self.disconnected), [0, 0])
    self.assertCountEqual(pga.out_degrees(self.cycle_3), [1, 1, 1])
    self.assertCountEqual(pga.out_degrees(self.chain_4), [1, 1, 1, 0])
    self.assertCountEqual(pga.out_degrees(self.wide_tree), [4, 0, 0, 0, 0])

  def test_diameter_returns_expected_if_connected(self):
    self.assertEqual(pga.diameter(self.singleton), 0)
    self.assertEqual(pga.diameter(self.cycle_3), 1)
    self.assertEqual(pga.diameter(self.chain_4), 3)
    self.assertEqual(pga.diameter(self.wide_tree), 2)

  def test_diameter_throws_exception_if_disconnected(self):
    with self.assertRaises(nx.exception.NetworkXError):
      pga.diameter(self.disconnected)

  def test_program_graph_to_nx_undirected_has_correct_edges(self):
    id_a, id_b, id_c = self.ids_from_cycle_3()
    nx_graph = pga._program_graph_to_nx(self.cycle_3, directed=False)
    self.assertCountEqual(nx_graph.nodes(), [id_a, id_b, id_c])
    expected_adj = {
        id_a: {
            id_b: {},
            id_c: {}
        },
        id_b: {
            id_a: {},
            id_c: {}
        },
        id_c: {
            id_a: {},
            id_b: {}
        },
    }
    self.assertEqual(nx_graph.adj, expected_adj)

  def test_program_graph_to_nx_directed_has_correct_edges(self):
    id_a, id_b, id_c = self.ids_from_cycle_3()
    nx_digraph = pga._program_graph_to_nx(self.cycle_3, directed=True)
    self.assertCountEqual(nx_digraph.nodes(), [id_a, id_b, id_c])
    expected_adj = {
        id_a: {
            id_b: {}
        },
        id_b: {
            id_c: {}
        },
        id_c: {
            id_a: {}
        },
    }
    self.assertEqual(nx_digraph.adj, expected_adj)

  def test_max_betweenness_returns_expected(self):
    self.assertAlmostEqual(pga.max_betweenness(self.singleton), 0)
    self.assertAlmostEqual(pga.max_betweenness(self.disconnected), 0)
    self.assertAlmostEqual(pga.max_betweenness(self.cycle_3), 0)

    # Middle nodes are in 2 shortest paths, normalizer = (4-1)*(4-2)/2 = 3
    self.assertAlmostEqual(pga.max_betweenness(self.chain_4), 2 / 3)

    # Root is in 6 shortest paths, normalizer = (5-1)*(5-2)/2 = 6
    self.assertAlmostEqual(pga.max_betweenness(self.wide_tree), 6 / 6)


if __name__ == '__main__':
  absltest.main()
