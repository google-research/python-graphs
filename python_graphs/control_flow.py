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

"""Computes the control flow graph for a Python program from its AST."""

import itertools
import uuid

from absl import logging  # pylint: disable=unused-import
import gast as ast
from python_graphs import instruction as instruction_module
from python_graphs import program_utils
import six


def get_control_flow_graph(program):
  """Get a ControlFlowGraph for the provided AST node.

  Args:
    program: Either an AST node, source string, or a function.
  Returns:
    A ControlFlowGraph.
  """
  control_flow_visitor = ControlFlowVisitor()
  node = program_utils.program_to_ast(program)
  control_flow_visitor.run(node)
  return control_flow_visitor.graph


class ControlFlowGraph(object):
  """A control flow graph for a Python program.

  Attributes:
    blocks: All blocks contained in the control flow graph.
    nodes: All control flow nodes in the control flow graph.
    start_block: The entry point to the program.
  """

  def __init__(self):
    self.blocks = []
    self.nodes = []

    self.start_block = self.new_block(prunable=False)
    self.start_block.label = '<start>'

  def add_node(self, control_flow_node):
    self.nodes.append(control_flow_node)

  def new_block(self, node=None, label=None, prunable=True):
    block = BasicBlock(node=node, label=label, prunable=prunable)
    block.graph = self
    self.blocks.append(block)
    return block

  def move_block_to_rear(self, block):
    self.blocks.remove(block)
    self.blocks.append(block)

  def get_control_flow_nodes(self):
    return self.nodes

  def get_enter_blocks(self):
    """Returns entry blocks for all functions."""
    return six.moves.filter(
        lambda block: block.label.startswith('<entry:'), self.blocks)

  def get_enter_control_flow_nodes(self):
    """Yields all ControlFlowNodes without any prev nodes."""
    for block in self.blocks:
      if (block.label is not None
          and block.label.startswith('<entry')
          and not block.control_flow_nodes):
        # The entry block does't have nodes itself, but rather has a next block
        # with control flow nodes.
        next_block = next(iter(block.next))
        if next_block.control_flow_nodes:
          yield next_block.control_flow_nodes[0]
      elif block.control_flow_nodes:
        node = block.control_flow_nodes[0]
        if not node.prev:
          yield node

  def get_exit_blocks(self):
    """Yields all blocks without any next blocks."""
    for block in self.blocks:
      if not block.next:
        yield block

  def get_instructions(self):
    """Yields all instructions in the control flow graph."""
    for block in self.blocks:
      for node in block.control_flow_nodes:
        yield node.instruction

  def get_start_control_flow_node(self):
    if self.start_block.control_flow_nodes:
      return self.start_block.control_flow_nodes[0]
    if self.start_block.exits_from_end:
      assert len(self.start_block.exits_from_end) == 1
      first_block = next(iter(self.start_block.exits_from_end))
      if first_block.control_flow_nodes:
        return first_block.control_flow_nodes[0]
      else:
        return first_block.label

  def get_control_flow_nodes_by_ast_node(self, node):
    return six.moves.filter(
        lambda control_flow_node: control_flow_node.instruction.node == node,
        self.get_control_flow_nodes())

  def get_control_flow_node_by_ast_node(self, node):
    return next(self.get_control_flow_nodes_by_ast_node(node))

  def get_blocks_by_ast_node(self, node):
    for block in self.blocks:
      for control_flow_node in block.control_flow_nodes:
        if node == control_flow_node.instruction.node:
          yield block
          break

  def get_block_by_ast_node(self, node):
    return next(self.get_blocks_by_ast_node(node))

  def get_blocks_by_function_name(self, name):
    """Returns entry blocks for any functions named `name`."""
    return six.moves.filter(
        lambda block: block.label == '<entry:{name}>'.format(name=name),
        self.blocks)

  def get_block_by_function_name(self, name):
    return next(self.get_blocks_by_function_name(name))

  def get_control_flow_nodes_by_source(self, source):
    module = ast.parse(source, mode='exec')  # TODO(dbieber): Factor out 4 lines
    node = module.body[0]
    if isinstance(node, ast.Expr):
      node = node.value

    return six.moves.filter(
        lambda cfn: cfn.instruction.contains_subprogram(node),
        self.get_control_flow_nodes())

  def get_control_flow_node_by_source(self, source):
    return next(self.get_control_flow_nodes_by_source(source))

  def get_control_flow_nodes_by_source_and_identifier(self, source, name):
    for control_flow_node in self.get_control_flow_nodes_by_source(source):
      for node in ast.walk(control_flow_node.instruction.node):
        if isinstance(node, ast.Name) and node.id == name:
          for i2 in self.get_control_flow_nodes_by_ast_node(node):
            yield i2

  def get_control_flow_node_by_source_and_identifier(self, source, name):
    return next(
        self.get_control_flow_nodes_by_source_and_identifier(source, name))

  def get_blocks_by_source(self, source):
    """Yields blocks that contain instructions matching the query source."""
    module = ast.parse(source, mode='exec')
    node = module.body[0]
    if isinstance(node, ast.Expr):
      node = node.value

    for block in self.blocks:
      for control_flow_node in block.control_flow_nodes:
        if control_flow_node.instruction.contains_subprogram(node):
          yield block
          break

  def get_block_by_source(self, source):
    return next(self.get_blocks_by_source(source))

  def get_blocks_by_source_and_ast_node_type(self, source, node_type):
    """Blocks with an Instruction matching node_type and containing source."""
    module = ast.parse(source, mode='exec')
    node = module.body[0]
    if isinstance(node, ast.Expr):
      node = node.value

    for block in self.blocks:
      for instruction in block.instructions:
        if (isinstance(instruction.node, node_type)
            and instruction.contains_subprogram(node)):
          yield block
          break

  def get_block_by_source_and_ast_node_type(self, source, node_type):
    """A block with an Instruction matching node_type and containing source."""
    return next(self.get_blocks_by_source_and_ast_node_type(source, node_type))

  def get_block_by_ast_node_and_label(self, node, label):
    """Gets the block corresponding to `node` having label `label`."""
    for block in self.blocks:
      if block.node is node and block.label == label:
        return block

  def get_blocks_by_ast_node_type_and_label(self, node_type, label):
    """Gets the blocks with node type `node_type` having label `label`."""
    for block in self.blocks:
      if isinstance(block.node, node_type) and block.label == label:
        yield block

  def get_block_by_ast_node_type_and_label(self, node_type, label):
    """Gets a block with node type `node_type` having label `label`."""
    return next(self.get_blocks_by_ast_node_type_and_label(node_type, label))

  def prune(self):
    """Prunes all prunable blocks from the graph."""
    progress = True
    while progress:
      progress = False
      for block in iter(self.blocks):
        if block.can_prune():
          to_remove = block.prune()
          self.blocks.remove(to_remove)
          progress = True

  def compact(self):
    """Prunes unused blocks and merges blocks when possible."""
    self.prune()
    for block in iter(self.blocks):
      while block.can_merge():
        to_remove = block.merge()
        self.blocks.remove(to_remove)
    for block in self.blocks:
      block.compact()


class Frame(object):
  """A Frame indicates how statements affect control flow in parts of a program.

  Frames are introduced when the program enters a new loop, function definition,
  or try/except/finally block.

  A Frame indicates how an exit such as a continue, break, exception, or return
  affects control flow. For example, a continue statement inside of a loop sends
  control back to the loop's condition. In nested loops, a continue statement
  sends control back to the condition of the innermost loop containing the
  continue statement.

  Attributes:
    kind: One of LOOP, FUNCTION, TRY_EXCEPT, or TRY_FINALLY.
    blocks: A dictionary with the blocks relevant to the frame.
  """

  # Kinds:
  MODULE = 'module'
  LOOP = 'loop'
  FUNCTION = 'function'
  TRY_EXCEPT = 'try-except'
  TRY_FINALLY = 'try-finally'

  def __init__(self, kind, **blocks):
    self.kind = kind
    self.blocks = blocks


class BasicBlock(object):
  """A basic block in a control flow graph.

  All instructions (generally, AST nodes) in a basic block are either executed
  or none are (with the exception of blocks interrupted by exceptions). These
  instructions are executed in a straight-line manner.

  Attributes:
    graph: The control flow graph which this basic block is a part of.

    next: Indicates which basic blocks may be executed after this basic block.
    prev: Indicates which basic blocks may lead to the execution of this basic
      block in a Python program.
    control_flow_nodes: A list of the ControlFlowNodes contained in this basic
      block. Each ControlFlowNode corresponds to a single Instruction.
    control_flow_node_indexes: Maps from id(control_flow_node) to the
      ControlFlowNode's index in self.control_flow_nodes. Only available once
      the block is compacted.

    branches: A map from booleans to the basic block reachable by making the
      branch decision indicated by that boolean.
    exits_from_middle: These basic blocks may be exited to at any point during
      the execution of this basic block.
    exits_from_end: These basic blocks may only be exited to at the end of
      the execution of this basic block.
    node: The AST node this basic block is associated with.
    prunable: Whether this basic block may be pruned from the control flow graph
      if empty. Set to False for special blocks, such as enter and exit blocks.
    label: A label for the basic block.
    identities: A list of (node, label) pairs that refer to this basic block.
      This starts as (self.node, self.label), but old identities are preserved
      during merging and pruning. Allows lookup of blocks by node and label,
      e.g. for finding the after block of a particular if statement.
    labels: Labels, used for example by data flow analyses. Maps from label name
      to value.
  """

  def __init__(self, node=None, label=None, prunable=True):
    self.graph = None
    self.next = set()
    self.prev = set()
    self.control_flow_nodes = []
    self.control_flow_node_indexes = None

    self.branches = {}
    self.except_branches = {}
    self.reraise_branches = {}

    self.exits_from_middle = set()
    self.exits_from_end = set()
    self.node = node
    self.prunable = prunable
    self.label = label
    self.identities = [(node, label)]
    self.labels = {}

  def has_label(self, label):
    """Returns whether this BasicBlock has the specified label."""
    return label in self.labels

  def set_label(self, label, value):
    """Sets the value of a label on the BasicBlock."""
    self.labels[label] = value

  def get_label(self, label):
    """Gets the value of a label on the BasicBlock."""
    return self.labels[label]

  def is_empty(self):
    """Whether this block is empty."""
    return not self.control_flow_nodes

  def exits_to(self, block):
    """Whether this block exits to `block`."""
    return block in self.next

  def raises_to(self, block):
    """Whether this block exits to `block` in the case of an exception."""
    return block in self.next and block in self.exits_from_middle

  def add_exit(self, block, interrupting=False,
               branch=None, except_branch=None, reraise_branch=None):
    """Adds an exit from this block to `block`."""
    self.next.add(block)
    block.prev.add(self)

    if branch is not None:
      self.branches[branch] = block
    if except_branch is not None:
      self.except_branches[except_branch] = block
    if reraise_branch is not None:
      self.reraise_branches[reraise_branch] = block

    if interrupting:
      self.exits_from_middle.add(block)
    else:
      self.exits_from_end.add(block)

  def remove_exit(self, block):
    """Removes the exit from this block to `block`."""
    self.next.remove(block)
    block.prev.remove(self)

    if block in self.exits_from_middle:
      self.exits_from_middle.remove(block)
    if block in self.exits_from_end:
      self.exits_from_end.remove(block)
    for branch_decision, branch_exit in self.branches.copy().items():
      if branch_exit is block:
        del self.branches[branch_decision]
    for branch_decision, branch_exit in self.except_branches.copy().items():
      if branch_exit is block:
        del self.except_branches[branch_decision]
    for branch_decision, branch_exit in self.reraise_branches.copy().items():
      if branch_exit is block:
        del self.reraise_branches[branch_decision]

  def can_prune(self):
    return self.is_empty() and self.prunable

  def prune(self):
    """Prunes the empty block from its control flow graph.

    A block is prunable if it has no control flow nodes and has not been marked
    as unprunable (e.g. because it's the exit block, or a return block, etc).

    Returns:
      The block removed by the prune operation. That is, self.
    """
    assert self.can_prune()
    prevs = self.prev.copy()
    nexts = self.next.copy()
    for prev_block in prevs:
      exits_from_middle = prev_block.exits_from_middle.copy()
      exits_from_end = prev_block.exits_from_end.copy()
      branches = prev_block.branches.copy()
      except_branches = prev_block.except_branches.copy()
      reraise_branches = prev_block.reraise_branches.copy()
      for next_block in nexts:
        if self in exits_from_middle:
          prev_block.add_exit(next_block, interrupting=True)
        if self in exits_from_end:
          prev_block.add_exit(next_block, interrupting=False)

        for branch_decision, branch_exit in branches.items():
          if branch_exit is self:
            prev_block.branches[branch_decision] = next_block
        for branch_decision, branch_exit in except_branches.items():
          if branch_exit is self:
            prev_block.except_branches[branch_decision] = next_block
        for branch_decision, branch_exit in reraise_branches.items():
          if branch_exit is self:
            prev_block.reraise_branches[branch_decision] = next_block

    for prev_block in prevs:
      prev_block.remove_exit(self)
    for next_block in nexts:
      self.remove_exit(next_block)
      next_block.identities = next_block.identities + self.identities
    return self

  def can_merge(self):
    if len(self.exits_from_end) != 1:
      return False
    next_block = next(iter(self.exits_from_end))
    if not next_block.prunable:
      return False
    if self.exits_from_middle != next_block.exits_from_middle:
      return False
    if len(next_block.prev) == 1:
      return True

  def merge(self):
    """Merge this block with its one successor.

    Returns:
      The successor block removed by the merge operation.
    """
    assert self.can_merge()
    next_block = next(iter(self.exits_from_end))

    exits_from_middle = next_block.exits_from_middle.copy()
    exits_from_end = next_block.exits_from_end.copy()

    for branch_decision, branch_exit in next_block.branches.items():
      self.branches[branch_decision] = branch_exit
    for branch_decision, branch_exit in next_block.except_branches.items():
      self.except_branches[branch_decision] = branch_exit
    for branch_decision, branch_exit in next_block.reraise_branches.items():
      self.reraise_branches[branch_decision] = branch_exit

    self.remove_exit(next_block)
    for block in next_block.next.copy():
      next_block.remove_exit(block)
      if block in exits_from_middle:
        self.add_exit(block, interrupting=True)
      if block in exits_from_end:
        self.add_exit(block, interrupting=False)
    for control_flow_node in next_block.control_flow_nodes:
      control_flow_node.block = self
      self.control_flow_nodes.append(control_flow_node)
    self.prunable = self.prunable and next_block.prunable
    self.label = self.label or next_block.label
    self.identities = self.identities + next_block.identities
    # Note: self.exits_from_middle is unchanged.
    return next_block

  def add_instruction(self, instruction):
    assert isinstance(instruction, instruction_module.Instruction)
    control_flow_node = ControlFlowNode(graph=self.graph,
                                        block=self,
                                        instruction=instruction)
    self.graph.add_node(control_flow_node)
    self.control_flow_nodes.append(control_flow_node)

  def compact(self):
    self.control_flow_node_indexes = {}
    for index, control_flow_node in enumerate(self.control_flow_nodes):
      self.control_flow_node_indexes[control_flow_node.uuid] = index

  def index_of(self, control_flow_node):
    """Returns the index of the Instruction in this BasicBlock."""
    return self.control_flow_node_indexes[control_flow_node.uuid]


class ControlFlowNode(object):
  """A node in a control flow graph.

  Corresponds to a single Instruction contained in a single BasicBlock.

  Attributes:
    graph: The ControlFlowGraph which this node is a part of.
    block: The BasicBlock in which this node's instruction resides.
    instruction: The Instruction corresponding to this node.
    labels: Metadata attached to this node, for example for use by data flow
      analyses.
    uuid: A unique identifier for the ControlFlowNode.
  """

  def __init__(self, graph, block, instruction):
    self.graph = graph
    self.block = block
    self.instruction = instruction
    self.labels = {}
    self.uuid = uuid.uuid4()

  @property
  def next(self):
    """Returns the set of possible next instructions.

    This allows for taking exits from the middle (exceptions).
    """
    if self.block is None:
      return None
    index_in_block = self.block.index_of(self)
    if len(self.block.control_flow_nodes) > index_in_block + 1:
      return {self.block.control_flow_nodes[index_in_block + 1]}
    control_flow_nodes = set()
    for next_block in self.block.next:
      if next_block.control_flow_nodes:
        control_flow_nodes.add(next_block.control_flow_nodes[0])
      else:
        # If next_block is empty, it isn't the case that some downstream block
        # is nonempty. This is guaranteed by the pruning phase of control flow
        # graph construction.
        assert not next_block.next
    return control_flow_nodes

  @property
  def next_from_end(self):
    """Returns the set of possible next instructions.

    This does not allow for taking exits from the middle (exceptions).
    """
    if self.block is None:
      return None
    index_in_block = self.block.index_of(self)
    if len(self.block.control_flow_nodes) > index_in_block + 1:
      return {self.block.control_flow_nodes[index_in_block + 1]}
    control_flow_nodes = set()
    for next_block in self.block.exits_from_end:
      if next_block.control_flow_nodes:
        control_flow_nodes.add(next_block.control_flow_nodes[0])
      else:
        # If next_block is empty, it isn't the case that some downstream block
        # is nonempty. This is guaranteed by the pruning phase of control flow
        # graph construction.
        assert not next_block.next
        control_flow_nodes.add(next_block.label)
    return control_flow_nodes

  @property
  def prev(self):
    """Returns the set of possible previous instructions."""
    if self.block is None:
      return None
    index_in_block = self.block.index_of(self)
    if index_in_block - 1 >= 0:
      return {self.block.control_flow_nodes[index_in_block - 1]}
    control_flow_nodes = set()
    for prev_block in self.block.prev:
      if prev_block.control_flow_nodes:
        control_flow_nodes.add(prev_block.control_flow_nodes[-1])
      else:
        # If prev_block is empty, it isn't the case that some upstream block
        # is nonempty. This is guaranteed by the pruning phase of control flow
        # graph construction.
        assert not prev_block.prev
    return control_flow_nodes

  @property
  def branches(self):
    """Returns the branch options available at the end of this node.

    Returns:
      A dictionary with possible keys True and False, and values given by the
      node that is reached by taking the True/False branch. An empty dictionary
      indicates that there are no branches to take, and so self.next gives the
      next node (in a set of size 1). A value of None indicates that taking that
      branch leads to the exit, since there are no exit ControlFlowNodes in a
      ControlFlowGraph.
    """
    return self.get_branches(
        include_except_branches=False,
        include_reraise_branches=False)

  def get_branches(self, include_except_branches=False, include_reraise_branches=False):
    """Returns the branch options available at the end of this node.

    Returns:
      A dictionary with possible keys True and False, and values given by the
      node that is reached by taking the True/False branch. An empty dictionary
      indicates that there are no branches to take, and so self.next gives the
      next node (in a set of size 1). A value of '<exit>' or '<raise>' indicates that
      taking that branch leads to the exit or raise block, since there are no exit
      ControlFlowNodes in a ControlFlowGraph.
    """
    if self.block is None:
      return {}  # We're not in a block. No branch decision.
    index_in_block = self.block.index_of(self)
    if len(self.block.control_flow_nodes) > index_in_block + 1:
      return {}  # We're not yet at the end of the block. No branch decision.

    branches = {}  # We're at the end of the block.

    all_branches = [self.block.branches.items()]
    if include_except_branches:
      all_branches.append(self.block.except_branches.items())
    if include_reraise_branches:
      all_branches.append(self.block.reraise_branches.items())

    for key, next_block in itertools.chain(*all_branches):
      if next_block.control_flow_nodes:
        branches[key] = next_block.control_flow_nodes[0]
      else:
        # If next_block is empty, it isn't the case that some downstream block
        # is nonempty. This is guaranteed by the pruning phase of control flow
        # graph construction.
        assert not next_block.next
        branches[key] = next_block.label  # Indicates exit or raise; there is no node to return.
    return branches

  def has_label(self, label):
    """Returns whether this Instruction has the specified label."""
    return label in self.labels

  def set_label(self, label, value):
    """Sets the value of a label on the Instruction."""
    self.labels[label] = value

  def get_label(self, label):
    """Gets the value of a label on the Instruction."""
    return self.labels[label]


# pylint: disable=invalid-name,g-doc-return-or-yield,g-doc-args
class ControlFlowVisitor(object):
  """A visitor for determining the control flow of a Python program from an AST.

  The main function of interest here is `visit`, which causes the visitor to
  construct the control flow graph for the node passed to visit.

  Basic control flow:
  The state of the Visitor consists of a sequence of frames, and a current
  basic block. When an AST node is visited by `visit`, it is added to the
  current basic block. When a node can indicate a possible change in control,
  new basic blocks are created and exits between the basic blocks are added
  as appropriate.

  For example, an If statement introduces two possibilities for control flow.
  Consider the program:

  if a > b:
    c = 1
  else:
    c = 2
  return c

  There are four basic blocks in this program: let's call them `compare`,
  `c = 1`, `c = 2`, and `return`. The exits between the blocks are:
  `compare` -> `c = 1`, `compare` -> `c = 2`, `c = 1` -> `return`, and
  `c = 2` -> `return`.

  Frames:
  There are four kinds of frames: function frames, loop frames, try-except, and
  try-finally frames. All AST nodes in a function definition are in that
  function's function frame. All AST nodes in the body of a loop are in that
  loop's loop frame. And all AST nodes in the try and except blocks of a
  try/except/finally are in that try's try-finally frame.

  A function frame contains information about where control should flow to in
  the case of a return statement or an uncaught exception.

  A loop frame contains information about where control should pass to in the
  case of a continue or break statement.

  A try-except frame contains information about where control should flow to in
  the case of an exception.

  A try-finally frame contains information about where control should flow to
  in the case of an exit (such as a finally block that must run before a return,
  continue, or break statement can be executed).

  Attributes:
    graph: The control flow graph being generated by the visitor.
    frames: The current frames. Each frame in this list contains all frames that
      come after it in the list.
  """

  def __init__(self):
    self.graph = ControlFlowGraph()
    self.frames = []

  def run(self, node):
    start_block = self.graph.start_block
    end_block = self.visit(node, start_block)
    self.graph.compact()

  def visit(self, node, current_block):
    """Visit node, either an AST node or a list.

    Args:
      node: The AST node being visited. Not necessarily an instance of ast.AST;
        node may also be a list, primitive, or Instruction.
      current_block: The basic block whose execution necessarily precedes the
        execution of `node`.
    Returns:
      The final basic block for the node.
    """
    assert isinstance(node, ast.AST)

    if isinstance(node, instruction_module.INSTRUCTION_AST_NODES):
      self.add_new_instruction(current_block, node)

    method_name = 'visit_' + node.__class__.__name__
    method = getattr(self, method_name, None)
    if method is not None:
      current_block = method(node, current_block)
    return current_block

  def visit_list(self, items, current_block):
    """Visit each of the items in a list from the AST."""
    for item in items:
      current_block = self.visit(item, current_block)
    return current_block

  def add_new_instruction(self, block, node, accesses=None, source=None):
    assert isinstance(node, ast.AST)
    instruction = instruction_module.Instruction(
        node, accesses=accesses, source=source)
    self.add_instruction(block, instruction)

  def add_instruction(self, block, instruction):
    assert isinstance(instruction, instruction_module.Instruction)
    block.add_instruction(instruction)

    # Any instruction may raise an exception.
    if not block.exits_from_middle:
      self.raise_through_frames(block, interrupting=True)

  def raise_through_frames(self, block, interrupting=True, except_branch=None):
    """Adds exits for the control flow of a raised exception.

    `interrupting` means the exit can occur at any point (exit_from_middle).
    `not interrupting` means the exit can only occur at the end of the block.

    The reason to raise_through_frames with interrupting=False is for an
    exception that already has been partially raised, but has passed control to
    a finally block, and is now being raised at the end of that finally block.

    Args:
      block: The block where the exception's control flow begins.
      interrupting: Whether the exception can be raised from any point in block.
        If False, the exception is only raised from the end of block.
      except_branch: False indicates the node raising is doing so the because an exception
        header did not match the raised error. None indicates otherwise.
    """
    frames = self.get_current_exception_handling_frames()

    if frames is None:
      return

    # reraise_branch indicates whether the a raise is a reraise of an earlier exception.
    # This is True after raising through a finally block, and None otherwise.
    reraise_branch = None

    for frame in frames:
      if frame.kind == Frame.TRY_FINALLY:
        # Exit to finally and have finally exit to whatever's next...
        final_block = frame.blocks['final_block']
        block.add_exit(final_block, interrupting=interrupting, except_branch=except_branch, reraise_branch=reraise_branch)
        block = frame.blocks['final_block_end']
        interrupting = False
        # "True" indicates the path taken after finally if an error has been raised.
        except_branch = None
        reraise_branch = True
      elif frame.kind == Frame.TRY_EXCEPT:
        handler_block = frame.blocks['handler_block']
        block.add_exit(handler_block, interrupting=interrupting, except_branch=except_branch, reraise_branch=reraise_branch)
        # This will be the last frame in frames.
      elif frame.kind == Frame.FUNCTION:
        raise_block = frame.blocks['raise_block']
        block.add_exit(raise_block, interrupting=interrupting, except_branch=except_branch, reraise_branch=reraise_branch)
        # This will be the last frame in frames.
      elif frame.kind == Frame.MODULE:
        raise_block = frame.blocks['raise_block']
        block.add_exit(raise_block, interrupting=interrupting, except_branch=except_branch, reraise_branch=reraise_branch)
        # This will be the last frame in frames.

  def new_block(self, node=None, label=None, prunable=True):
    """Create a new block."""
    return self.graph.new_block(node=node, label=label, prunable=prunable)

  def enter_module_frame(self, exit_block, raise_block):
    # The entire module is in the interior of the frame.
    # The exit block and raise block are the exits from the frame.
    self.frames.append(Frame(Frame.MODULE,
                             exit_block=exit_block,
                             raise_block=raise_block))

  def enter_loop_frame(self, continue_block, break_block):
    # The loop body is the interior of the frame.
    # The continue block (loop condition) and break block (loop's after block)
    # are the exits from the frame.
    self.frames.append(Frame(Frame.LOOP,
                             continue_block=continue_block,
                             break_block=break_block))

  def enter_function_frame(self, return_block, raise_block):
    # The function body is the interior of the frame.
    # The return block and raise block are the exits from the frame.
    self.frames.append(Frame(Frame.FUNCTION,
                             return_block=return_block,
                             raise_block=raise_block))

  def enter_try_except_frame(self, handler_block):
    # The try block is the interior of the frame.
    # handler_block is where the frame exits to on an exception.
    self.frames.append(Frame(Frame.TRY_EXCEPT,
                             handler_block=handler_block))

  def enter_try_finally_frame(self, final_block, final_block_end):
    # The try block and handler blocks are the interior of the frame.
    # The finally block is the exit from the frame.
    self.frames.append(Frame(Frame.TRY_FINALLY,
                             final_block=final_block,
                             final_block_end=final_block_end))

  def exit_frame(self):
    """Exits the innermost current frame.

    Note: Each enter_* function must be matched to exactly one exit_frame call
    in reverse order.

    Returns:
      The frame being exited.
    """
    return self.frames.pop()

  def get_current_loop_frame(self):
    """Gets the current loop frame and contained current try-finally frames.

    In order to exit the current loop frame, we must first enter the finally
    blocks of all current contained try-finally frames.

    Returns:
      A list of frames, all of which are try-finally frames except for the last,
      which is the current loop frame. Each of the returned try-finally
      frames is contained within the current loop frame.
    """
    frames = []
    for frame in reversed(self.frames):
      if frame.kind == Frame.TRY_FINALLY:
        frames.append(frame)
      if frame.kind == Frame.LOOP:
        frames.append(frame)
        return frames
    # There are no loop frames.
    return None

  def get_current_function_frame(self):
    """Gets the current function frame and contained current try-finally frames.

    In order to exit the current function frame, we must first enter the finally
    blocks of all current contained try-finally frames.

    Returns:
      A list of frames, all of which are try-finally frames except for the last,
      which is the current function frame. Each of the returned try-finally
      frames is contained within the current function frame.
    """
    frames = []
    for frame in reversed(self.frames):
      if frame.kind == Frame.TRY_FINALLY:
        frames.append(frame)
      if frame.kind == Frame.FUNCTION:
        frames.append(frame)
        return frames
    # There are no function frames.
    return None

  def get_current_exception_handling_frames(self):
    """Get all exception handling frames containing the current block.

    Returns:
      A list of frames, all of which are exception handling frames containing
      the current block. Any instruction contained in a try-except frame may
      exit to the frame's exception handling block, with the caveat that an
      instruction cannot exit through a TRY_FINALLY frame without passing first
      through the frame's finally block. (The instruction will exit to the
      finally block, and the finally block in turn will exit to the exception
      handler.) A function frame's raise block serves to catch exceptions as
      well.
    """
    frames = []
    # Traverse frames from innermost to outermost until a frame that fully
    # catches the exception is found.
    for frame in reversed(self.frames):
      if frame.kind == Frame.TRY_FINALLY:
        frames.append(frame)
      if frame.kind == Frame.TRY_EXCEPT:
        # A try-except frame catches any exception, even if the frame's except
        # statements do not match the exception. In this case, the final except
        # will reraise the exception to higher frames.
        frames.append(frame)
        return frames
      if frame.kind == Frame.FUNCTION:
        # A function frame's raise_block catches any exception that reaches it.
        frames.append(frame)
        return frames
      if frame.kind == Frame.MODULE:
        # A module frame's raise_block catches any exception that reaches it.
        frames.append(frame)
        return frames
    # There is no frame to fully catch the exception.
    raise ValueError('No frame exists to catch the exception.')

  def visit_Module(self, node, current_block):
    exit_block = self.new_block(node=node, label='<exit>', prunable=False)
    raise_block = self.new_block(node=node, label='<raise>', prunable=False)
    self.enter_module_frame(exit_block, raise_block)
    end_block = self.visit_list(node.body, current_block)
    end_block.add_exit(exit_block)
    self.exit_frame()
    # Move exit and raise blocks to the end of the block list.
    self.graph.move_block_to_rear(exit_block)
    self.graph.move_block_to_rear(raise_block)
    return end_block

  def visit_ClassDef(self, node, current_block):
    """Visit a ClassDef node of the AST.

    Blocks:
      current_block: The block in which the class is defined.
    """
    # TODO(dbieber): Make sure all statements are handled, such as base classes.
    # http://greentreesnakes.readthedocs.io/en/latest/nodes.html#ClassDef
    # The body is executed before the decorators.
    current_block = self.visit_list(node.body, current_block)
    for decorator in node.decorator_list:
      self.add_new_instruction(current_block, decorator)
    assert isinstance(node.name, six.string_types)
    self.add_new_instruction(
        current_block,
        node,
        accesses=instruction_module.create_writes(node.name, node),
        source=instruction_module.CLASS)
    return current_block

  def visit_FunctionDef(self, node, current_block):
    """Visit a FunctionDef node of the AST.

    Blocks:
      current_block: The block in which the function is defined.
    """
    # First defaults are computed, then decorators are run, then the functiondef
    # is assigned to the function name.
    current_block = self.handle_argument_defaults(node.args, current_block)
    for decorator in node.decorator_list:
      self.add_new_instruction(current_block, decorator)
    assert isinstance(node.name, six.string_types)
    self.add_new_instruction(
        current_block,
        node,
        accesses=instruction_module.create_writes(node.name, node),
        source=instruction_module.FUNCTION)
    self.handle_function_definition(node, node.name, node.args, node.body)
    return current_block

  def visit_Lambda(self, node, current_block):
    """Visit a Lambda node of the AST.

    Blocks:
      current_block: The block in which the lambda is defined.
    """
    current_block = self.handle_argument_defaults(node.args, current_block)
    self.handle_function_definition(node, 'lambda', node.args, node.body)
    return current_block

  def handle_function_definition(self, node, name, args, body):
    """A helper fn for Lambda and FunctionDef.

    Note that this function doesn't require a block as input, since it doesn't
    modify the blocks where the function definition resides.

    Blocks:
      entry_block: The block where control flow starts when the function is
         called.
      return_block: The block the function returns to.
      raise_block: The block the function raises uncaught exceptions to.
      fn_block: The first used block of the FunctionDef.

    Args:
      node: The AST node of the function definition, either a FunctionDef or
        Lambda node.
      name: The function's name, a string.
      args: The function's args, an ast.arguments node.
      body: The function's body, a list of AST nodes.
    """
    return_block = self.new_block(node=node, label='<return>', prunable=False)
    raise_block = self.new_block(node=node, label='<raise>', prunable=False)
    self.enter_function_frame(return_block, raise_block)

    entry_block = self.new_block(node=node, label='<entry:{}>'.format(name),
                                 prunable=False)
    fn_block = self.new_block(node=node, label='fn_block')
    entry_block.add_exit(fn_block)
    fn_block = self.handle_argument_writes(args, fn_block)
    fn_block = self.visit_list(body, fn_block)
    fn_block.add_exit(return_block)
    self.exit_frame()
    self.graph.move_block_to_rear(return_block)
    self.graph.move_block_to_rear(raise_block)

  def handle_argument_defaults(self, node, current_block):
    """Add Instructions for all of a FunctionDef's default values.

    Note that these instructions are in the block containing the function, not
    in the function definition itself.
    """
    for default in node.defaults:
      self.add_new_instruction(current_block, default)
    for default in node.kw_defaults:
      if default is None:
        continue
      self.add_new_instruction(current_block, default)
    return current_block

  def handle_argument_writes(self, node, current_block):
    """Add Instructions for all of a FunctionDef's arguments.

    These instructions are part of a function's body.
    """
    accesses = []
    if node.args:
      for arg in node.args:
        accesses.extend(instruction_module.create_writes(arg, node))
    if node.vararg:
      accesses.extend(instruction_module.create_writes(node.vararg, node))
    if node.kwonlyargs:
      for arg in node.kwonlyargs:
        accesses.extend(instruction_module.create_writes(arg, node))
    if node.kwarg:
      accesses.extend(instruction_module.create_writes(node.kwarg, node))

    if accesses:
      self.add_new_instruction(
          current_block,
          node,
          accesses=accesses,
          source=instruction_module.ARGS)
    return current_block

  def visit_If(self, node, current_block):
    """Visit an If node of the AST.

    Blocks:
      current_block: This is where the if statement resides. The if statement's
        test is added here.
      after_block: The block to which control is passed after the if statement
        is completed.
      true_block: The true branch of the if statements.
      false_block: The false branch of the if statements.
    """
    self.add_new_instruction(current_block, node.test)
    after_block = self.new_block(node=node, label='after_block')
    true_block = self.new_block(node=node, label='true_block')
    current_block.add_exit(true_block, branch=True)
    true_block = self.visit_list(node.body, true_block)
    true_block.add_exit(after_block)
    if node.orelse:
      false_block = self.new_block(node=node, label='false_block')
      current_block.add_exit(false_block, branch=False)
      false_block = self.visit_list(node.orelse, false_block)
      false_block.add_exit(after_block)
    else:
      current_block.add_exit(after_block, branch=False)
    return after_block

  def visit_While(self, node, current_block):
    """Visit a While node of the AST.

    Blocks:
      current_block: This is where the while statement resides.
    """
    test_instruction = instruction_module.Instruction(node.test)
    return self.handle_Loop(node, test_instruction, current_block)

  def visit_For(self, node, current_block):
    """Visit a For node of the AST.

    Blocks:
      current_block: This is where the for statement resides.
    """
    self.add_new_instruction(current_block, node.iter)
    # node.target is a Name, Tuple, or List node.
    # We wrap it in an Instruction so it knows where its write is coming from.
    target = instruction_module.Instruction(
        node.target,
        accesses=instruction_module.create_writes(node.target, node),
        source=instruction_module.ITERATOR)
    return self.handle_Loop(node, target, current_block)

  def handle_Loop(self, node, loop_instruction, current_block):
    """A helper fn for For and While.

    Args:
      node: The AST node representing the loop.
      loop_instruction: The Instruction in the loop header, such as a test or an
        assignment from an iterator.
      current_block: The BasicBlock containing the loop.

    Blocks:
      current_block: This is where the loop resides.
      test_block: Contains the part of the loop header that is repeated. For a
        While, this is the loop condition. For a For, this is assignment to the
        target variable.
      test_block_end: The last block in the test (often the same as test_block.)
      body_block: The body of the loop.
      else_block: Executed if the loop terminates naturally.
      after_block: Follows the completion of the loop.
    """
    # We do not add an instruction for the ast.For or ast.While node.
    test_block = self.new_block(node=node, label='test_block')
    current_block.add_exit(test_block)
    self.add_instruction(test_block, loop_instruction)
    body_block = self.new_block(node=node, label='body_block')
    after_block = self.new_block(node=node, label='after_block')

    test_block.add_exit(body_block, branch=True)
    # In the loop, continue goes to test_block and break goes to after_block.
    self.enter_loop_frame(test_block, after_block)
    body_block = self.visit_list(node.body, body_block)
    body_block.add_exit(test_block)
    self.exit_frame()

    # If a loop exits via its test (rather than via a break) and it has
    # an orelse, then it enters the orelse.
    if node.orelse:
      else_block = self.new_block(node=node, label='else_block')
      test_block.add_exit(else_block, branch=False)
      else_block = self.visit_list(node.orelse, else_block)
      else_block.add_exit(after_block)
    else:
      test_block.add_exit(after_block, branch=False)

    self.graph.move_block_to_rear(after_block)
    return after_block

  def visit_Try(self, node, current_block):
    """Visit a Try node of the AST.

    Blocks:
      current_block: This is where the try statement resides.
      after_block: The block to which control flows after the conclusion of the
        full try statement (including e.g. the else and finally sections, if
        present).
      handler_blocks: A list of blocks corresponding to the except statements.
      bare_handler_block: The handler block corresponding to a bare except
        statement. One of handler_blocks or None.
      handler_body_blocks: A list of blocks corresponding to the bodies of the
        except sections.
      final_block: The block corresponding to the finally section.
      final_block_end: The last block corresponding to the finally section.
      try_block: The block corresponding to the try section.
      try_block_end: The last block corresponding to the try section.
      else_block: The block corresponding to the else section.
    """
    # We do not add an instruction for the ast.Try node.
    after_block = self.new_block()
    handler_blocks = [self.new_block() for _ in node.handlers]
    handler_body_blocks = [self.new_block() for _ in node.handlers]

    # If there is a bare except clause, determine its handler block.
    # Only the last except is permitted to be a bare except.
    if node.handlers and node.handlers[-1].type is None:
      bare_handler_block = handler_blocks[-1]
    else:
      bare_handler_block = None

    if node.finalbody:
      final_block = self.new_block(node=node, label='final_block')
      final_block_end = self.visit_list(node.finalbody, final_block)
      # "False" indicates the path taken after finally if no error has been raised.
      final_block_end.add_exit(after_block, reraise_branch=False)
      self.enter_try_finally_frame(final_block, final_block_end)
    else:
      final_block = after_block

    if node.handlers:
      self.enter_try_except_frame(handler_blocks[0])

    # The exits from the try_block may happen at any point since any instruction
    # can throw an exception.
    try_block = self.new_block(node=node, label='try_block')
    current_block.add_exit(try_block)
    try_block_end = self.visit_list(node.body, try_block)
    if node.orelse:  # The try body can exit to the else block.
      else_block = self.new_block(node=node, label='else_block')
      try_block_end.add_exit(else_block)
    else:  # If there is no else block, the try body can exit to final/after.
      try_block_end.add_exit(final_block)

    if node.handlers:
      self.exit_frame()  # Exit the try-except frame.

    previous_handler_block_end = None
    for handler, handler_block, handler_body_block in zip(node.handlers,
                                                          handler_blocks,
                                                          handler_body_blocks):
      previous_handler_block_end = self.handle_ExceptHandler(
          handler, handler_block, handler_body_block, final_block,
          previous_handler_block_end=previous_handler_block_end)

    if bare_handler_block is None and previous_handler_block_end is not None:
      # If no exceptions match, then raise up through the frames.
      # (A bare-except will always match.)
      # Here "False" indicates the final exception header did not match the raised error.
      self.raise_through_frames(
          previous_handler_block_end, interrupting=False, except_branch=False)

    if node.orelse:
      else_block = self.visit_list(node.orelse, else_block)
      else_block.add_exit(final_block)  # orelse exits to final/after

    if node.finalbody:
      self.exit_frame()  # Exit the try-finally frame.

    self.graph.move_block_to_rear(after_block)
    return after_block

  def handle_ExceptHandler(self, handler, handler_block, handler_body_block,
                           final_block, previous_handler_block_end=None):
    """Create the blocks appropriate for an exception handler.

    Args:
      handler: The AST ExceptHandler node.
      handler_block: The block corresponding the ExceptHandler header.
      handler_body_block: The block corresponding to the ExceptHandler body.
      final_block: Where the handler body should exit to when it executes
        successfully.
      previous_handler_block_end: The last block corresponding to the previous
        ExceptHandler header, if there is one, or None otherwise. The previous
        handler's header should exit to this handler's header if the exception
        doesn't match the previous handler's header.

    Returns:
      The last (usually the only) block in the handler's header.

    Note that rather than having a visit_ExceptHandler function, we instead
    use the following logic. This is because except statements don't follow
    the visitor pattern exactly. Specifically, a handler may exit to either
    its body or to the next handler, but under the visitor pattern the
    handler would not know the block belonging to the next handler.
    """
    if handler.type is not None:
      self.add_new_instruction(handler_block, handler.type)
    # An ExceptHandler header can only have a single Instruction, so there is
    # only one handler_block BasicBlock.
    # Here "True" indicates the exception header matches the raised error.
    handler_block.add_exit(handler_body_block, except_branch=True)

    if previous_handler_block_end is not None:
      # Here "False" indicates the previous exception header did not match the
      # raised error.
      previous_handler_block_end.add_exit(handler_block, except_branch=False)
    previous_handler_block_end = handler_block

    if handler.name is not None:
      # handler.name is a Name, Tuple, or List AST node.
      self.add_new_instruction(
          handler_body_block,
          handler.name,
          accesses=instruction_module.create_writes(handler.name, handler),
          source=instruction_module.EXCEPTION)
    handler_body_block = self.visit_list(handler.body, handler_body_block)
    handler_body_block.add_exit(final_block)  # handler exits to final/after
    return previous_handler_block_end

  def visit_Return(self, node, current_block):
    """Visit a Return node of the AST.

    Blocks:
      current_block: This is where the return statement resides.
      return_block: The containing function's return block. All successful exits
        from the function lead here.

    Raises:
      RuntimeError: If a return AST node is visited while not in a function
        frame.
    """
    # The Return statement is an Instruction. Don't visit the node's children.
    frames = self.get_current_function_frame()
    if frames is None:
      raise RuntimeError('return occurs outside of a function frame.')
    try_finally_frames = frames[:-1]
    function_frame = frames[-1]

    return_block = function_frame.blocks['return_block']
    return self.handle_ExitStatement(node,
                                     return_block,
                                     try_finally_frames,
                                     current_block)

  def visit_Yield(self, node, current_block):
    """Visit a Yield node of the AST.

    The current implementation of yields allows control to flow directly through
    a yield statement. TODO(dbieber): Introduce a <yield> node in between
    yielding and resuming execution.
    TODO(dbieber): Yield nodes aren't even visited since they are contained in
    Expr nodes. Determine if Yield can occur outside of an Expr. Check for
    Yield when visiting Expr.
    """
    logging.warn('yield visited: %s', ast.dump(node))
    # The Yield statement is an Instruction. Don't visit children.
    return current_block

  def visit_Continue(self, node, current_block):
    """Visit a Continue node of the AST.

    Blocks:
      current_block: This is where the continue statement resides.
      continue_block: The block of the containing loop's header. For a For,
        this is the target variable assignment. For a While, this is the loop
        condition.

    Raises:
      RuntimeError: If a continue AST node is visited while not in a loop frame.
    """
    frames = self.get_current_loop_frame()
    if frames is None:
      raise RuntimeError('continue occurs outside of a loop frame.')

    try_finally_frames = frames[:-1]
    loop_frame = frames[-1]

    continue_block = loop_frame.blocks['continue_block']
    return self.handle_ExitStatement(node,
                                     continue_block,
                                     try_finally_frames,
                                     current_block)

  def visit_Break(self, node, current_block):
    """Visit a Break node of the AST.

    Blocks:
      current_block: This is where the break statement resides.
      break_block: The block that the containing loop exits to.

    Raises:
      RuntimeError: If a break AST node is visited while not in a loop frame.
    """
    frames = self.get_current_loop_frame()
    if frames is None:
      raise RuntimeError('break occurs outside of a loop frame.')

    try_finally_frames = frames[:-1]
    loop_frame = frames[-1]

    break_block = loop_frame.blocks['break_block']
    return self.handle_ExitStatement(node,
                                     break_block,
                                     try_finally_frames,
                                     current_block)

  def visit_Raise(self, node, current_block):
    """Visit a Raise node of the AST.

    Blocks:
      current_block: This is where the raise statement resides.
      after_block: An unreachable block for code that follows the raise
        statement.
    """
    self.raise_through_frames(current_block, interrupting=False)
    # The Raise statement is an Instruction. Don't visit children.

    # Note there is no exit to the after_block. It is unreachable.
    after_block = self.new_block(node=node, label='after_block')

    return after_block

  def handle_ExitStatement(self, node, next_block, try_finally_frames,
                           current_block):
    """A helper fn for Return, Continue, and Break.

    An exit statement is a statement such as return, continue, break, or raise.
    Such a statement causes control to leave through a frame's exit. Any
    instructions immediately following an exit statement will be unreachable.

    Args:
      node: The AST node of the exit statement.
      next_block: The block the exit statement exits to.
      try_finally_frames: A possibly empty list of try-finally frames whose
        finally blocks must be executed before control can pass to next_block.
      current_block: The block the exit statement resides in.

    Blocks:
      current_block: This is where the exit statement resides.
      next_block: The block the exit statement exits to (after first passing
        through all the finally blocks.)
      final_block: The start of a finally section that control must pass
        through on the way to next_block.
      final_block_end: The end of a finally section that control must pass
        through on the way to next_block.
      after_block: An unreachable block for code that follows the raise
        statement.
    """
    for try_finally_frame in try_finally_frames:
      final_block = try_finally_frame.blocks['final_block']
      current_block.add_exit(final_block)
      current_block = try_finally_frame.blocks['final_block_end']

    current_block.add_exit(next_block)

    # Note there is no exit to the after_block. It is unreachable.
    after_block = self.new_block(node=node, label='after_block')
    return after_block
