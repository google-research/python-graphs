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

"""Functions to analyze program graphs.

Computes properties such as the height of a program graph's AST.
"""

import gast as ast
import networkx as nx


def num_nodes(graph):
  """Returns the number of nodes in a ProgramGraph."""
  return len(graph.all_nodes())


def num_edges(graph):
  """Returns the number of edges in a ProgramGraph."""
  return len(graph.edges)


def ast_height(ast_node):
  """Computes the height of an AST from the given node.

  Args:
    ast_node: An AST node.

  Returns:
    The height of the AST starting at ast_node. A leaf node or single-node AST
    has a height of 1.
  """
  max_child_height = 0
  for child_node in ast.iter_child_nodes(ast_node):
    max_child_height = max(max_child_height, ast_height(child_node))
  return 1 + max_child_height


def graph_ast_height(graph):
  """Computes the height of the AST of a ProgramGraph.

  Args:
    graph: A ProgramGraph.

  Returns:
    The height of the graph's AST. A single-node AST has a height of 1.
  """
  return ast_height(graph.to_ast())


def degrees(graph):
  """Returns a list of node degrees in a ProgramGraph.

  Args:
    graph: A ProgramGraph.

  Returns:
    An (unsorted) list of node degrees (in-degree plus out-degree).
  """
  return [len(graph.neighbors(node)) for node in graph.all_nodes()]


def in_degrees(graph):
  """Returns a list of node in-degrees in a ProgramGraph.

  Args:
    graph: A ProgramGraph.

  Returns:
    An (unsorted) list of node in-degrees.
  """
  return [len(graph.incoming_neighbors(node)) for node in graph.all_nodes()]


def out_degrees(graph):
  """Returns a list of node out-degrees in a ProgramGraph.

  Args:
    graph: A ProgramGraph.

  Returns:
    An (unsorted) list of node out-degrees.
  """
  return [len(graph.outgoing_neighbors(node)) for node in graph.all_nodes()]


def _program_graph_to_nx(program_graph, directed=False):
  """Converts a ProgramGraph to a NetworkX graph.

  Args:
    program_graph: A ProgramGraph.
    directed: Whether the graph should be treated as a directed graph.

  Returns:
    A NetworkX graph that can be analyzed by the networkx module.
  """
  # Create a dict-of-lists representation, where {0: [1]} represents a directed
  # edge from node 0 to node 1.
  dict_of_lists = {}
  for node in program_graph.all_nodes():
    neighbor_ids = [neighbor.id
                    for neighbor in program_graph.outgoing_neighbors(node)]
    dict_of_lists[node.id] = neighbor_ids
  return nx.DiGraph(dict_of_lists) if directed else nx.Graph(dict_of_lists)


def diameter(graph):
  """Returns the diameter of a ProgramGraph.

  Note: this is very slow for large graphs.

  Args:
    graph: A ProgramGraph.

  Returns:
    The diameter of the graph. A single-node graph has diameter 0. The graph is
    treated as an undirected graph.

  Raises:
    networkx.exception.NetworkXError: Raised if the graph is not connected.
  """
  nx_graph = _program_graph_to_nx(graph, directed=False)
  return nx.algorithms.distance_measures.diameter(nx_graph)


def max_betweenness(graph):
  """Returns the maximum node betweenness centrality in a ProgramGraph.

  Note: this is very slow for large graphs.

  Args:
    graph: A ProgramGraph.

  Returns:
    The maximum betweenness centrality value among all nodes in the graph. The
    graph is treated as an undirected graph.
  """
  nx_graph = _program_graph_to_nx(graph, directed=False)
  return max(nx.algorithms.centrality.betweenness_centrality(nx_graph).values())
