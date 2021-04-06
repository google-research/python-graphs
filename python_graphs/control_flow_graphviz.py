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

"""Graphviz render for control flow graphs."""

from absl import logging  # pylint: disable=unused-import
import astunparse
import gast as ast
import pygraphviz

LEFT_ALIGN = '\l'  # pylint: disable=anomalous-backslash-in-string


def render(graph, include_src=None, path='/tmp/graph.png'):
  g = to_graphviz(graph, include_src=include_src)
  g.draw(path, prog='dot')


def trim(line, max_length=30):
  if len(line) <= max_length:
    return line
  return line[:max_length - 3] + '...'


def unparse(node):
  source = astunparse.unparse(node)
  trimmed_source = '\n'.join(trim(line) for line in source.split('\n'))
  return (
      trimmed_source.strip()
      .rstrip(' \n')
      .lstrip(' \n')
      .replace('\n', LEFT_ALIGN)
  )


def write_as_str(write):
  if isinstance(write, ast.AST):
    return unparse(write)
  else:
    return write


def get_label_for_instruction(instruction):
  if instruction.source is not None:
    line = ', '.join(write for write in instruction.get_write_names())
    line += ' <- ' + instruction.source
    return line
  else:
    return unparse(instruction.node)


def get_label(block):
  """Gets the source code for a control flow basic block."""
  lines = []
  for control_flow_node in block.control_flow_nodes:
    instruction = control_flow_node.instruction
    line = get_label_for_instruction(instruction)
    if line.strip():
      lines.append(line)

  return LEFT_ALIGN.join(lines) + LEFT_ALIGN


def to_graphviz(graph, include_src=None):
  """To graphviz."""
  g = pygraphviz.AGraph(strict=False, directed=True)
  for block in graph.blocks:
    node_attrs = {}
    label = get_label(block)
    # We only show the <entry>, <exit>, <start>, <raise>, <return> block labels.
    if block.label is not None and block.label.startswith('<'):
      node_attrs['style'] = 'bold'
      if not label.rstrip(LEFT_ALIGN):
        label = block.label + LEFT_ALIGN
      else:
        label = block.label + LEFT_ALIGN + label
    node_attrs['label'] = label
    node_attrs['fontname'] = 'Courier New'
    node_attrs['fontsize'] = 10.0

    node_id = id(block)
    g.add_node(node_id, **node_attrs)
    for next_node in block.next:
      next_node_id = id(next_node)
      if next_node in block.exits_from_middle:
        edge_attrs = {}
        edge_attrs['style'] = 'dashed'
        g.add_edge(node_id, next_node_id, **edge_attrs)
      if next_node in block.exits_from_end:
        edge_attrs = {}
        edge_attrs['style'] = 'solid'
        g.add_edge(node_id, next_node_id, **edge_attrs)

  if include_src is not None:
    node_id = id(include_src)
    node_attrs['label'] = include_src.replace('\n', LEFT_ALIGN)
    node_attrs['fontname'] = 'Courier New'
    node_attrs['fontsize'] = 10.0
    node_attrs['shape'] = 'box'
    g.add_node(node_id, **node_attrs)

  return g
