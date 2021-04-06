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

"""Creates ProgramGraphs from a program or function's AST.

A ProgramGraph represents a Python program or function. The nodes in a
ProgramGraph represent an Instruction (see instruction.py), an AST node, or a
piece of syntax from the program. The edges in a ProgramGraph represent the
relationships between these nodes.
"""

import codecs
import collections
import os

from absl import logging
import astunparse
from astunparse import unparser
import gast as ast
from python_graphs import control_flow
from python_graphs import data_flow
from python_graphs import instruction as instruction_module
from python_graphs import program_graph_dataclasses as pb
from python_graphs import program_utils
import six
from six.moves import builtins
from six.moves import filter

NEWLINE_TOKEN = '#NEWLINE#'
UNINDENT_TOKEN = '#UNINDENT#'
INDENT_TOKEN = '#INDENT#'


class ProgramGraph(object):
  """A ProgramGraph represents a Python program or function.

  Attributes:
    root_id: The id of the root ProgramGraphNode.
    nodes: Maps from node id to the ProgramGraphNode with that id.
    edges: A list of the edges (from_node.id, to_node.id, edge type) in the
      graph.
    child_map: Maps from node id to a list of that node's AST children node ids.
    parent_map: Maps from node id to that node's AST parent node id.
    neighbors_map: Maps from node id to a list of that node's neighboring edges.
    ast_id_to_program_graph_node: Maps from an AST node's object id to the
      corresponding AST program graph node, if it exists.
    root: The root ProgramGraphNode.
  """

  def __init__(self):
    """Constructs an empty ProgramGraph with no root."""
    self.root_id = None

    self.nodes = {}
    # TODO(charlessutton): Seems odd to have Edge proto objects as part of the
    # program graph object if node protos aren't. Consider a more consistent
    # treatment.
    self.edges = []

    self.ast_id_to_program_graph_node = {}
    self.child_map = collections.defaultdict(list)
    self.parent_map = collections.defaultdict(lambda: None)
    self.neighbors_map = collections.defaultdict(list)

  # Accessors
  @property
  def root(self):
    if self.root_id not in self.nodes:
      raise ValueError('Graph has no root node.')
    return self.nodes[self.root_id]

  def all_nodes(self):
    return self.nodes.values()

  def get_node(self, obj):
    """Returns the node in the program graph corresponding to an object.

    Arguments:
       obj: Can be an integer, AST node, ProgramGraphNode, or program graph node
         protobuf.

    Raises:
       ValueError: no node exists in the program graph matching obj.
    """
    if isinstance(obj, six.integer_types) and obj in self.nodes:
      return self.get_node_by_id(obj)
    elif isinstance(obj, ProgramGraphNode):
      # assert obj in self.nodes.values()
      return obj
    elif isinstance(obj, pb.Node):
      return self.get_node_by_id(obj.id)
    elif isinstance(obj, (ast.AST, list)):
      return self.get_node_by_ast_node(obj)
    else:
      raise ValueError('Unexpected value for obj.', obj)

  def get_node_by_id(self, obj):
    """Gets a ProgramGraph node for the given integer id."""
    return self.nodes[obj]

  def get_node_by_access(self, access):
    """Gets a ProgramGraph node for the given read or write."""
    if isinstance(access, ast.Name):
      return self.get_node(access)
    else:
      assert isinstance(access, tuple)
      if isinstance(access[1], ast.Name):
        return self.get_node(access[1])
      else:
        return self.get_node(access[2])
    raise ValueError('Could not find node for access.', access)

  def get_nodes_by_source(self, source):
    """Generates the nodes in the program graph containing the query source.

    Args:
      source: The query source.

    Returns:
      A generator of all nodes in the program graph with an Instruction with
      source that includes the query source.
    """
    module = ast.parse(source, mode='exec')  # TODO(dbieber): Factor out 4 lines
    # TODO(dbieber): Use statements beyond the first statement from source.
    node = module.body[0]
    # If the query source is an Expression, and the matching instruction matches
    # the value field of that Expression, then the matching instruction is
    # considered a match. This allows us to match subexpressions which appear in
    # ast.Expr nodes in the query but not in the parent.
    if isinstance(node, ast.Expr):
      node = node.value

    def matches_source(pg_node):
      if pg_node.has_instruction():
        return pg_node.instruction.contains_subprogram(node)
      else:
        return instruction_module.represent_same_program(pg_node.ast_node, node)

    return filter(matches_source, self.nodes.values())

  def get_node_by_source(self, node):
    # We use min since nodes can contain each other and we want the most
    # specific one.
    return min(
        self.get_nodes_by_source(node), key=lambda x: len(ast.dump(x.node)))

  def get_nodes_by_function_name(self, name):
    return filter(
        lambda n: n.has_instance_of(ast.FunctionDef) and n.node.name == name,
        self.nodes.values())

  def get_node_by_function_name(self, name):
    return next(self.get_nodes_by_function_name(name))

  def get_node_by_ast_node(self, ast_node):
    return self.ast_id_to_program_graph_node[id(ast_node)]

  def contains_ast_node(self, ast_node):
    return id(ast_node) in self.ast_id_to_program_graph_node

  def get_ast_nodes_of_type(self, ast_type):
    for node in six.itervalues(self.nodes):
      if node.node_type == pb.NodeType.AST_NODE and node.ast_type == ast_type:
        yield node

  # TODO(dbieber): Unify selectors across program_graph and control_flow.
  def get_nodes_by_source_and_identifier(self, source, name):
    for pg_node in self.get_nodes_by_source(source):
      for node in ast.walk(pg_node.node):
        if isinstance(node, ast.Name) and node.id == name:
          if self.contains_ast_node(node):
            yield self.get_node_by_ast_node(node)

  def get_node_by_source_and_identifier(self, source, name):
    return next(self.get_nodes_by_source_and_identifier(source, name))

  # Graph Construction Methods
  def add_node(self, node):
    """Adds a ProgramGraphNode to this graph.

    Args:
      node: The ProgramGraphNode that should be added.

    Returns:
      The node that was added.

    Raises:
      ValueError: the node has already been added to this graph.
    """
    assert isinstance(node, ProgramGraphNode), 'Not a ProgramGraphNode'
    if node.id in self.nodes:
      raise ValueError('Already contains node', self.nodes[node.id], node.id)
    if node.ast_node is not None:
      if self.contains_ast_node(node.ast_node):
        raise ValueError('Already contains ast node', node.ast_node)
      self.ast_id_to_program_graph_node[id(node.ast_node)] = node
    self.nodes[node.id] = node
    return node

  def add_node_from_instruction(self, instruction):
    """Adds a node to the program graph."""
    node = make_node_from_instruction(instruction)
    return self.add_node(node)

  def add_edge(self, edge):
    """Adds an edge between two nodes in the graph.

    Args:
      edge: The edge, a pb.Edge proto.
    """
    assert isinstance(edge, pb.Edge), 'Not a pb.Edge'
    self.edges.append(edge)

    n1 = self.get_node_by_id(edge.id1)
    n2 = self.get_node_by_id(edge.id2)
    if edge.type == pb.EdgeType.FIELD:  # An AST node.
      self.child_map[edge.id1].append(edge.id2)
      # TODO(charlessutton): Add the below sanity check back once Instruction
      # updates are complete.
      # pylint: disable=line-too-long
      # other_parent_id = self.parent_map[edge.id2]
      # if other_parent_id and other_parent_id != edge.id1:
      #   raise Exception('Node {} {} with two parents\n {} {}\n {} {}'
      #                   .format(edge.id2, dump_node(self.get_node(edge.id2)),
      #                           edge.id1, dump_node(self.get_node(edge.id1)),
      #                           other_parent_id, dump_node(self.get_node(other_parent_id))))
      # pylint: enable=line-too-long
      self.parent_map[n2.id] = edge.id1
    self.neighbors_map[n1.id].append((edge, edge.id2))
    self.neighbors_map[n2.id].append((edge, edge.id1))

  def remove_edge(self, edge):
    """Removes an edge from the graph.

    If there are multiple copies of the same edge, only one copy is removed.

    Args:
      edge: The edge, a pb.Edge proto.
    """
    self.edges.remove(edge)

    n1 = self.get_node_by_id(edge.id1)
    n2 = self.get_node_by_id(edge.id2)

    if edge.type == pb.EdgeType.FIELD:  # An AST node.
      self.child_map[edge.id1].remove(edge.id2)
      del self.parent_map[n2.id]

    self.neighbors_map[n1.id].remove((edge, edge.id2))
    self.neighbors_map[n2.id].remove((edge, edge.id1))

  def add_new_edge(self, n1, n2, edge_type=None, field_name=None):
    """Adds a new edge between two nodes in the graph.

    Both nodes must already be part of the graph.

    Args:
      n1: Specifies the from node of the edge. Can be any object type accepted
        by get_node.
      n2: Specifies the to node of the edge. Can be any object type accepted by
        get_node.
      edge_type: The type of edge. Can be any integer in the pb.Edge enum.
      field_name: For AST edges, a string describing the Python AST field

    Returns:
      The new edge.
    """
    n1 = self.get_node(n1)
    n2 = self.get_node(n2)
    new_edge = pb.Edge(
        id1=n1.id, id2=n2.id, type=edge_type, field_name=field_name)
    self.add_edge(new_edge)
    return new_edge

  # AST Methods
  # TODO(charlessutton): Consider whether AST manipulation should be moved
  # e.g., to a more general graph object.
  def to_ast(self, node=None):
    """Convert the program graph to a Python AST."""
    if node is None:
      node = self.root
    return self._build_ast(node=node, update_references=False)

  def reconstruct_ast(self):
    """Reconstruct all internal ProgramGraphNode.ast_node references.

    After calling this method, all nodes of type AST_NODE will have their
    `ast_node` property refer to subtrees of a reconstructed AST object, and
    self.ast_id_to_program_graph_node will contain only entries from this new
    AST.

    Note that only AST nodes reachable by fields from the root node will be
    converted; this should be all of them but this is not checked.
    """
    self.ast_id_to_program_graph_node.clear()
    self._build_ast(node=self.root, update_references=True)

  def _build_ast(self, node, update_references):
    """Helper method: builds an AST and optionally sets ast_node references.

    Args:
      node: Program graph node to build an AST for.
      update_references: Whether to modify this node and all of its children so
        that they point to the reconstructed AST node.

    Returns:
      AST node corresponding to the program graph node.
    """
    if node.node_type == pb.NodeType.AST_NODE:
      ast_node = getattr(ast, node.ast_type)()
      adjacent_edges = self.neighbors_map[node.id]
      for edge, other_node_id in adjacent_edges:
        if other_node_id == edge.id1:  # it's an incoming edge
          continue
        if edge.type == pb.EdgeType.FIELD:
          child_id = other_node_id
          child = self.get_node_by_id(child_id)
          setattr(
              ast_node, edge.field_name,
              self._build_ast(node=child, update_references=update_references))
      if update_references:
        node.ast_node = ast_node
        self.ast_id_to_program_graph_node[id(ast_node)] = node
      return ast_node
    elif node.node_type == pb.NodeType.AST_LIST:
      list_items = {}
      adjacent_edges = self.neighbors_map[node.id]
      for edge, other_node_id in adjacent_edges:
        if other_node_id == edge.id1:  # it's an incoming edge
          continue
        if edge.type == pb.EdgeType.FIELD:
          child_id = other_node_id
          child = self.get_node_by_id(child_id)
          unused_field_name, index = parse_list_field_name(edge.field_name)
          list_items[index] = self._build_ast(
              node=child, update_references=update_references)

      ast_list = []
      for index in six.moves.range(len(list_items)):
        ast_list.append(list_items[index])
      return ast_list
    elif node.node_type == pb.NodeType.AST_VALUE:
      return node.ast_value
    else:
      raise ValueError('This ProgramGraphNode does not correspond to a node in'
                       ' an AST.')

  def walk_ast_descendants(self, node=None):
    """Yields the nodes that correspond to the descendants of node in the AST.

    Args:
      node: the node in the program graph corresponding to the root of the AST
        subtree that should be walked. If None, defaults to the root of the
        program graph.

    Yields:
      All nodes corresponding to descendants of node in the AST.
    """
    if node is None:
      node = self.root
    frontier = [node]
    while frontier:
      current = frontier.pop()
      for child_id in reversed(self.child_map[current.id]):
        frontier.append(self.get_node_by_id(child_id))
      yield current

  def parent(self, node):
    """Returns the AST parent of an AST program graph node.

    Args:
      node: A ProgramGraphNode.

    Returns:
      The node's AST parent, which is also a ProgramGraphNode.
    """
    parent_id = self.parent_map[node.id]
    if parent_id is None:
      return None
    else:
      return self.get_node_by_id(parent_id)

  def children(self, node):
    """Yields the (direct) AST children of an AST program graph node.

    Args:
      node: A ProgramGraphNode.

    Yields:
      The AST children of node, which are ProgramGraphNode objects.
    """
    for child_id in self.child_map[node.id]:
      yield self.get_node_by_id(child_id)

  def neighbors(self, node, edge_type=None):
    """Returns the incoming and outgoing neighbors of a program graph node.

    Args:
      node: A ProgramGraphNode.
      edge_type: If provided, only edges of this type are considered.

    Returns:
      The incoming and outgoing neighbors of node, which are ProgramGraphNode
      objects but not necessarily AST nodes.
    """
    adj_edges = self.neighbors_map[node.id]
    if edge_type is None:
      ids = list(tup[1] for tup in adj_edges)
    else:
      ids = list(tup[1] for tup in adj_edges if tup[0].type == edge_type)
    return [self.get_node_by_id(id0) for id0 in ids]

  def incoming_neighbors(self, node, edge_type=None):
    """Returns the incoming neighbors of a program graph node.

    Args:
      node: A ProgramGraphNode.
      edge_type: If provided, only edges of this type are considered.

    Returns:
      The incoming neighbors of node, which are ProgramGraphNode objects but not
      necessarily AST nodes.
    """
    adj_edges = self.neighbors_map[node.id]
    result = []
    for edge, neighbor_id in adj_edges:
      if edge.id2 == node.id:
        if (edge_type is None) or (edge.type == edge_type):
          result.append(self.get_node_by_id(neighbor_id))
    return result

  def outgoing_neighbors(self, node, edge_type=None):
    """Returns the outgoing neighbors of a program graph node.

    Args:
      node: A ProgramGraphNode.
      edge_type: If provided, only edges of this type are considered.

    Returns:
      The outgoing neighbors of node, which are ProgramGraphNode objects but not
      necessarily AST nodes.
    """
    adj_edges = self.neighbors_map[node.id]
    result = []
    for edge, neighbor_id in adj_edges:
      if edge.id1 == node.id:
        if (edge_type is None) or (edge.type == edge_type):
          result.append(self.get_node_by_id(neighbor_id))
    return result

  def dump_tree(self, start_node=None):
    """Returns a string representation for debugging."""

    def dump_tree_recurse(node, indent, all_lines):
      """Create a string representation for a subtree."""
      indent_str = ' ' + ('--' * indent)
      node_str = dump_node(node)
      line = ' '.join([indent_str, node_str, '\n'])
      all_lines.append(line)
      # output long distance edges
      for edge, neighbor_id in self.neighbors_map[node.id]:
        if (not is_ast_edge(edge) and not is_syntax_edge(edge) and
            node.id == edge.id1):
          type_str = edge.type.name
          line = [indent_str, '--((', type_str, '))-->', str(neighbor_id), '\n']
          all_lines.append(' '.join(line))
      for child in self.children(node):
        dump_tree_recurse(child, indent + 1, all_lines)
      return all_lines

    if start_node is None:
      start_node = self.root
    return ''.join(dump_tree_recurse(start_node, 0, []))

  # TODO(charlessutton): Consider whether this belongs in ProgramGraph
  # or in make_synthesis_problems.
  def copy_with_placeholder(self, node):
    """Returns a new program graph in which the subtree of NODE is removed.

    In the new graph, the subtree headed by NODE is replaced by a single
    node of type PLACEHOLDER, which is connected to the AST parent of NODE
    by the same edge type as in the original graph.

    The new program graph will share structure (i.e. the ProgramGraphNode
    objects) with the original graph.

    Args:
      node: A node in this program graph

    Returns:
      A new ProgramGraph object with NODE replaced
    """
    descendant_ids = {n.id for n in self.walk_ast_descendants(node)}
    new_graph = ProgramGraph()
    new_graph.add_node(self.root)
    new_graph.root_id = self.root_id
    for edge in self.edges:
      v1 = self.nodes[edge.id1]
      v2 = self.nodes[edge.id2]
      # Omit edges that are adjacent to the subtree rooted at `node` UNLESS this
      # is the AST edge to the root of the subtree.
      # In that case, create an edge to a new placeholder node
      adj_bad_subtree = ((edge.id1 in descendant_ids) or
                         (edge.id2 in descendant_ids))
      if adj_bad_subtree:
        if edge.id2 == node.id and is_ast_edge(edge):
          placeholder = ProgramGraphNode()
          placeholder.node_type = pb.NodeType.PLACEHOLDER
          placeholder.id = node.id
          new_graph.add_node(placeholder)
          new_graph.add_new_edge(v1, placeholder, edge_type=edge.type)
      else:
        # nodes on the edge have not been added yet
        if edge.id1 not in new_graph.nodes:
          new_graph.add_node(v1)
        if edge.id2 not in new_graph.nodes:
          new_graph.add_node(v2)
        new_graph.add_new_edge(v1, v2, edge_type=edge.type)
    return new_graph

  def copy_subgraph(self, node):
    """Returns a new program graph containing only the subtree rooted at NODE.

    All edges that connect nodes in the subtree are included, both AST edges
    and other types of edges.

    Args:
      node: A node in this program graph

    Returns:
      A new ProgramGraph object whose root is NODE
    """
    descendant_ids = {n.id for n in self.walk_ast_descendants(node)}
    new_graph = ProgramGraph()
    new_graph.add_node(node)
    new_graph.root_id = node.id
    for edge in self.edges:
      v1 = self.nodes[edge.id1]
      v2 = self.nodes[edge.id2]
      # Omit edges that are adjacent to the subtree rooted at NODE
      # UNLESS this is the AST edge to the root of the subtree.
      # In that case, create an edge to a new placeholder node
      good_edge = ((edge.id1 in descendant_ids) and
                   (edge.id2 in descendant_ids))
      if good_edge:
        if edge.id1 not in new_graph.nodes:
          new_graph.add_node(v1)
        if edge.id2 not in new_graph.nodes:
          new_graph.add_node(v2)
        new_graph.add_new_edge(v1, v2, edge_type=edge.type)
    return new_graph


def is_ast_node(node):
  return node.node_type == pb.NodeType.AST_NODE


def is_ast_edge(edge):
  # TODO(charlessutton): Expand to enumerate edge types in gast.
  return edge.type == pb.EdgeType.FIELD


def is_syntax_edge(edge):
  return edge.type == pb.EdgeType.SYNTAX


def dump_node(node):
  type_str = '[' + node.node_type.name + ']'
  elements = [type_str, str(node.id), node.ast_type]
  if node.ast_value:
    elements.append(str(node.ast_value))
  if node.syntax:
    elements.append(str(node.syntax))
  return ' '.join(elements)


def get_program_graph(program):
  """Constructs a program graph to represent the given program."""
  program_node = program_utils.program_to_ast(program)  # An AST node.

  # TODO(dbieber): Refactor sections of graph building into separate functions.
  program_graph = ProgramGraph()

  # Perform control flow analysis.
  control_flow_graph = control_flow.get_control_flow_graph(program_node)

  # Add AST_NODE program graph nodes corresponding to Instructions in the
  # control flow graph.
  for control_flow_node in control_flow_graph.get_control_flow_nodes():
    program_graph.add_node_from_instruction(control_flow_node.instruction)

  # Add AST_NODE program graph nodes corresponding to AST nodes.
  for ast_node in ast.walk(program_node):
    if not program_graph.contains_ast_node(ast_node):
      pg_node = make_node_from_ast_node(ast_node)
      program_graph.add_node(pg_node)

  root = program_graph.get_node_by_ast_node(program_node)
  program_graph.root_id = root.id

  # Add AST edges (FIELD). Also add AST_LIST and AST_VALUE program graph nodes.
  for ast_node in ast.walk(program_node):
    for field_name, value in ast.iter_fields(ast_node):
      if isinstance(value, list):
        pg_node = make_node_for_ast_list()
        program_graph.add_node(pg_node)
        program_graph.add_new_edge(
            ast_node, pg_node, pb.EdgeType.FIELD, field_name)
        for index, item in enumerate(value):
          list_field_name = make_list_field_name(field_name, index)
          if isinstance(item, ast.AST):
            program_graph.add_new_edge(pg_node, item, pb.EdgeType.FIELD,
                                       list_field_name)
          else:
            item_node = make_node_from_ast_value(item)
            program_graph.add_node(item_node)
            program_graph.add_new_edge(pg_node, item_node, pb.EdgeType.FIELD,
                                       list_field_name)
      elif isinstance(value, ast.AST):
        program_graph.add_new_edge(
            ast_node, value, pb.EdgeType.FIELD, field_name)
      else:
        pg_node = make_node_from_ast_value(value)
        program_graph.add_node(pg_node)
        program_graph.add_new_edge(
            ast_node, pg_node, pb.EdgeType.FIELD, field_name)

  # Add SYNTAX_NODE nodes. Also add NEXT_SYNTAX and LAST_LEXICAL_USE edges.
  # Add these edges using a custom AST unparser to visit leaf nodes in preorder.
  SyntaxNodeUnparser(program_node, program_graph)

  # Perform data flow analysis.
  analysis = data_flow.LastAccessAnalysis()
  for node in control_flow_graph.get_enter_control_flow_nodes():
    analysis.visit(node)

  # Add control flow edges (CFG_NEXT).
  for control_flow_node in control_flow_graph.get_control_flow_nodes():
    instruction = control_flow_node.instruction
    for next_control_flow_node in control_flow_node.next:
      next_instruction = next_control_flow_node.instruction
      program_graph.add_new_edge(
          instruction.node, next_instruction.node,
          edge_type=pb.EdgeType.CFG_NEXT)

  # Add data flow edges (LAST_READ and LAST_WRITE).
  for control_flow_node in control_flow_graph.get_control_flow_nodes():
    # Start with the most recent accesses before this instruction.
    last_accesses = control_flow_node.get_label('last_access_in').copy()
    for access in control_flow_node.instruction.accesses:
      # Extract the node and identifiers for the current access.
      pg_node = program_graph.get_node_by_access(access)
      access_name = instruction_module.access_name(access)
      read_identifier = instruction_module.access_identifier(
          access_name, 'read')
      write_identifier = instruction_module.access_identifier(
          access_name, 'write')
      # Find previous reads.
      for read in last_accesses.get(read_identifier, []):
        read_pg_node = program_graph.get_node_by_access(read)
        program_graph.add_new_edge(
            pg_node, read_pg_node, edge_type=pb.EdgeType.LAST_READ)
      # Find previous writes.
      for write in last_accesses.get(write_identifier, []):
        write_pg_node = program_graph.get_node_by_access(write)
        program_graph.add_new_edge(
            pg_node, write_pg_node, edge_type=pb.EdgeType.LAST_WRITE)
      # Update the state to refer to this access as the most recent one.
      if instruction_module.access_is_read(access):
        last_accesses[read_identifier] = [access]
      elif instruction_module.access_is_write(access):
        last_accesses[write_identifier] = [access]

  # Add COMPUTED_FROM edges.
  for node in ast.walk(program_node):
    if isinstance(node, ast.Assign):
      for value_node in ast.walk(node.value):
        if isinstance(value_node, ast.Name):
          # TODO(dbieber): If possible, improve precision of these edges.
          for target in node.targets:
            program_graph.add_new_edge(
                target, value_node, edge_type=pb.EdgeType.COMPUTED_FROM)

  # Add CALLS, FORMAL_ARG_NAME and RETURNS_TO edges.
  for node in ast.walk(program_node):
    if isinstance(node, ast.Call):
      if isinstance(node.func, ast.Name):
        # TODO(dbieber): Use data flow analysis instead of all function defs.
        func_defs = list(program_graph.get_nodes_by_function_name(node.func.id))
        # For any possible last writes that are a function definition, add the
        # formal_arg_name and returns_to edges.
        if not func_defs:
          # TODO(dbieber): Add support for additional classes of functions,
          # such as attributes of known objects and builtins.
          if node.func.id in dir(builtins):
            message = 'Function is builtin.'
          else:
            message = 'Cannot statically determine the function being called.'
          logging.debug('%s (%s)', message, node.func.id)
        for func_def in func_defs:
          fn_node = func_def.node
          # Add calls edge from the call node to the function definition.
          program_graph.add_new_edge(node, fn_node, edge_type=pb.EdgeType.CALLS)
          # Add returns_to edges from the function's return statements to the
          # call node.
          for inner_node in ast.walk(func_def.node):
            # TODO(dbieber): Determine if the returns_to should instead go to
            # the next instruction after the Call node instead.
            if isinstance(inner_node, ast.Return):
              program_graph.add_new_edge(
                  inner_node, node, edge_type=pb.EdgeType.RETURNS_TO)

          # Add formal_arg_name edges from the args of the Call node to the
          # args in the FunctionDef.
          for index, arg in enumerate(node.args):
            formal_arg = None
            if index < len(fn_node.args.args):
              formal_arg = fn_node.args.args[index]
            elif fn_node.args.vararg:
              # Since args.vararg is a string, we use the arguments node.
              # TODO(dbieber): Use a node specifically for the vararg.
              formal_arg = fn_node.args
            if formal_arg is not None:
              # Note: formal_arg can be an AST node or a string.
              program_graph.add_new_edge(
                  arg, formal_arg, edge_type=pb.EdgeType.FORMAL_ARG_NAME)
            else:
              # TODO(dbieber): If formal_arg is None, then remove all
              # formal_arg_name edges for this FunctionDef.
              logging.debug('formal_arg is None')
          for keyword in node.keywords:
            name = keyword.arg
            formal_arg = None
            for arg in fn_node.args.args:
              if isinstance(arg, ast.Name) and arg.id == name:
                formal_arg = arg
                break
            else:
              if fn_node.args.kwarg:
                # Since args.kwarg is a string, we use the arguments node.
                # TODO(dbieber): Use a node specifically for the kwarg.
                formal_arg = fn_node.args
            if formal_arg is not None:
              program_graph.add_new_edge(
                  keyword.value, formal_arg,
                  edge_type=pb.EdgeType.FORMAL_ARG_NAME)
            else:
              # TODO(dbieber): If formal_arg is None, then remove all
              # formal_arg_name edges for this FunctionDef.
              logging.debug('formal_arg is None')
      else:
        # TODO(dbieber): Add a special case for Attributes.
        logging.debug(
            'Cannot statically determine the function being called. (%s)',
            astunparse.unparse(node.func).strip())

  return program_graph


class SyntaxNodeUnparser(unparser.Unparser):
  """An Unparser class helpful for creating Syntax Token nodes for fn graphs."""

  def __init__(self, ast_node, graph):
    self.graph = graph

    self.current_ast_node = None  # The AST node currently being unparsed.
    self.last_syntax_node = None
    self.last_lexical_uses = {}
    self.last_indent = 0

    with codecs.open(os.devnull, 'w', encoding='utf-8') as devnull:
      super(SyntaxNodeUnparser, self).__init__(ast_node, file=devnull)

  def dispatch(self, ast_node):
    """Dispatcher function, dispatching tree type T to method _T."""
    tmp_ast_node = self.current_ast_node
    self.current_ast_node = ast_node
    super(SyntaxNodeUnparser, self).dispatch(ast_node)
    self.current_ast_node = tmp_ast_node

  def fill(self, text=''):
    """Indent a piece of text, according to the current indentation level."""
    text_with_whitespace = NEWLINE_TOKEN
    if self.last_indent > self._indent:
      text_with_whitespace += UNINDENT_TOKEN * (self.last_indent - self._indent)
    elif self.last_indent < self._indent:
      text_with_whitespace += INDENT_TOKEN * (self._indent - self.last_indent)
    self.last_indent = self._indent
    text_with_whitespace += text
    self._add_syntax_node(text_with_whitespace)
    super(SyntaxNodeUnparser, self).fill(text)

  def write(self, text):
    """Append a piece of text to the current line."""
    if isinstance(text, ast.AST):  # text may be a Name, Tuple, or List node.
      return self.dispatch(text)
    self._add_syntax_node(text)
    super(SyntaxNodeUnparser, self).write(text)

  def _add_syntax_node(self, text):
    text = text.strip()
    if not text:
      return
    syntax_node = make_node_from_syntax(six.text_type(text))
    self.graph.add_node(syntax_node)
    self.graph.add_new_edge(
        self.current_ast_node, syntax_node, edge_type=pb.EdgeType.SYNTAX)
    if self.last_syntax_node:
      self.graph.add_new_edge(
          self.last_syntax_node, syntax_node, edge_type=pb.EdgeType.NEXT_SYNTAX)
    self.last_syntax_node = syntax_node

  def _Name(self, node):
    if node.id in self.last_lexical_uses:
      self.graph.add_new_edge(
          node,
          self.last_lexical_uses[node.id],
          edge_type=pb.EdgeType.LAST_LEXICAL_USE)
    self.last_lexical_uses[node.id] = node
    super(SyntaxNodeUnparser, self)._Name(node)


class ProgramGraphNode(object):
  """A single node in a Program Graph.

  Corresponds to either a SyntaxNode or an Instruction (as in a
  ControlFlowGraph).

  Attributes:
    node_type: One of the node types from pb.NodeType.
    id: A unique id for the node.
    instruction: If applicable, the corresponding Instruction.
    ast_node: If available, the AST node corresponding to the ProgramGraphNode.
    ast_type: If available, the type of the AST node, as a string.
    ast_value: If available, the primitive Python value corresponding to the
      node.
    syntax: For SYNTAX_NODEs, the syntax information stored in the node.
    node: If available, the AST node for this program graph node or its
      instruction.
  """

  def __init__(self):
    self.node_type = None
    self.id = None

    self.instruction = None
    self.ast_node = None
    self.ast_type = ''
    self.ast_value = ''
    self.syntax = ''

  def has_instruction(self):
    return self.instruction is not None

  def has_instance_of(self, t):
    """Whether the node's instruction is an instance of type `t`."""
    if self.instruction is None:
      return False
    return isinstance(self.instruction.node, t)

  @property
  def node(self):
    if self.ast_node is not None:
      return self.ast_node
    if self.instruction is None:
      return None
    return self.instruction.node

  def __repr__(self):
    return str(self.id) + ' ' + str(self.ast_type)


def make_node_from_syntax(text):
  node = ProgramGraphNode()
  node.node_type = pb.NodeType.SYNTAX_NODE
  node.id = program_utils.unique_id()
  node.syntax = text
  return node


def make_node_from_instruction(instruction):
  """Creates a ProgramGraphNode corresponding to an existing Instruction.

  Args:
    instruction: An Instruction object.

  Returns:
    A ProgramGraphNode corresponding to that instruction.
  """
  ast_node = instruction.node
  node = make_node_from_ast_node(ast_node)
  node.instruction = instruction
  return node


def make_node_from_ast_node(ast_node):
  """Creates a program graph node for the provided AST node.

  This is only called when the AST node doesn't already correspond to an
  Instruction in the program's control flow graph.

  Args:
    ast_node: An AST node from the program being analyzed.

  Returns:
    A node in the program graph corresponding to the AST node.
  """
  node = ProgramGraphNode()
  node.node_type = pb.NodeType.AST_NODE
  node.id = program_utils.unique_id()
  node.ast_node = ast_node
  node.ast_type = type(ast_node).__name__
  return node


def make_node_for_ast_list():
  node = ProgramGraphNode()
  node.node_type = pb.NodeType.AST_LIST
  node.id = program_utils.unique_id()
  return node


def make_node_from_ast_value(value):
  """Creates a ProgramGraphNode for the provided value.

  `value` is a primitive value appearing in a Python AST.

  For example, the number 1 in Python has AST Num(n=1). In this, the value '1'
  is a primitive appearing in the AST. It gets its own ProgramGraphNode with
  node_type AST_VALUE.

  Args:
    value: A primitive value appearing in an AST.

  Returns:
    A ProgramGraphNode corresponding to the provided value.
  """
  node = ProgramGraphNode()
  node.node_type = pb.NodeType.AST_VALUE
  node.id = program_utils.unique_id()
  node.ast_value = value
  return node


def make_list_field_name(field_name, index):
  return '{}:{}'.format(field_name, index)


def parse_list_field_name(list_field_name):
  field_name, index = list_field_name.split(':')
  index = int(index)
  return field_name, index
