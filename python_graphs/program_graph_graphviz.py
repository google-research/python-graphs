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

"""Graphviz visualizations of Program Graphs."""

from absl import logging  # pylint: disable=unused-import
import pygraphviz
from python_graphs import program_graph_dataclasses as pb
import six


def to_graphviz(graph):
  """Creates a graphviz representation of a ProgramGraph.

  Args:
    graph: A ProgramGraph object to visualize.
  Returns:
    A pygraphviz object representing the ProgramGraph.
  """
  g = pygraphviz.AGraph(strict=False, directed=True)
  for unused_key, node in graph.nodes.items():
    node_attrs = {}
    if node.ast_type:
      node_attrs['label'] = six.ensure_binary(node.ast_type, 'utf-8')
    else:
      node_attrs['shape'] = 'point'
    node_type_colors = {
    }
    if node.node_type in node_type_colors:
      node_attrs['color'] = node_type_colors[node.node_type]
      node_attrs['colorscheme'] = 'svg'

    g.add_node(node.id, **node_attrs)
  for edge in graph.edges:
    edge_attrs = {}
    edge_attrs['label'] = edge.type.name
    edge_colors = {
        pb.EdgeType.LAST_READ: 'red',
        pb.EdgeType.LAST_WRITE: 'red',
    }
    if edge.type in edge_colors:
      edge_attrs['color'] = edge_colors[edge.type]
      edge_attrs['colorscheme'] = 'svg'
    g.add_edge(edge.id1, edge.id2, **edge_attrs)
  return g


def render(graph, path='/tmp/graph.png'):
  g = to_graphviz(graph)
  g.draw(path, prog='dot')
