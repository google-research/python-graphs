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

"""Data flow analysis of Python programs."""

import collections

from absl import logging  # pylint: disable=unused-import
import gast as ast

from python_graphs import control_flow
from python_graphs import instruction as instruction_module


READ = instruction_module.READ
WRITE = instruction_module.WRITE


class Analysis(object):
  """Base class for a data flow analysis.

  Attributes:
    label: The name of the analysis.
    forward: (bool) True for forward analyses, False for backward analyses.
    in_label: The name of the analysis, suffixed with _in.
    out_label: The name of the analysis, suffixed with _out.
    before_label: Either the in_label or out_label depending on the direction of
      the analysis. Marks the before_value on a node during an analysis.
    after_label: Either the in_label or out_label depending on the direction of
      the analysis. Marks the after_value on a node during an analysis.
  """

  def __init__(self, label, forward):
    self.label = label
    self.forward = forward

    self.in_label = label + '_in'
    self.out_label = label + '_out'

    self.before_label = self.in_label if forward else self.out_label
    self.after_label = self.out_label if forward else self.in_label

  def aggregate_previous_after_values(self, previous_after_values):
    """Computes the before value for a node from the previous after values.

    This is the 'meet' or 'join' function of the analysis.
    TODO(dbieber): Update terminology to match standard textbook notation.

    Args:
      previous_after_values: The after values of all before nodes.
    Returns:
      The before value for the current node.
    """
    raise NotImplementedError

  def compute_after_value(self, node, before_value):
    """Computes the after value for a node from the node and the before value.

    This is the 'transfer' function of the analysis.
    TODO(dbieber): Update terminology to match standard textbook notation.

    Args:
      node: The node or block for which to compute the after value.
      before_value: The before value of the node.
    Returns:
      The computed after value for the node.
    """
    raise NotImplementedError

  def visit(self, node):
    """Visit the nodes of the control flow graph, performing the analysis.

    Terminology:
      in_value: The value of the analysis at the start of a node.
      out_value: The value of the analysis at the end of a node.
      before_value: in_value in a forward analysis; out_value in a backward
        analysis.
      after_value: out_value in a forward analysis; in_value in a backward
        analysis.

    Args:
      node: A graph element that supports the .next / .prev API, such as a
        ControlFlowNode from a ControlFlowGraph or a BasicBlock from a
        ControlFlowGraph.
    """
    to_visit = collections.deque([node])
    while to_visit:
      node = to_visit.popleft()

      before_nodes = node.prev if self.forward else node.next
      after_nodes = node.next if self.forward else node.prev
      previous_after_values = [
          before_node.get_label(self.after_label)
          for before_node in before_nodes
          if before_node.has_label(self.after_label)]

      if node.has_label(self.after_label):
        initial_after_value_hash = hash(node.get_label(self.after_label))
      else:
        initial_after_value_hash = None
      before_value = self.aggregate_previous_after_values(previous_after_values)
      node.set_label(self.before_label, before_value)
      after_value = self.compute_after_value(node, before_value)
      node.set_label(self.after_label, after_value)
      if hash(after_value) != initial_after_value_hash:
        for after_node in after_nodes:
          to_visit.append(after_node)


def get_while_loop_variables(node, graph=None):
  """Gets the set of loop variables used for while loop rewriting.

  This is the set of variables used for rewriting a while loop into its
  functional form.

  Args:
    node: An ast.While AST node.
    graph: (Optional) The ControlFlowGraph of the function or program containing
      the while loop. If not present, the control flow graph for the while loop
      will be computed.
  Returns:
    The set of variable identifiers that are live at the start of the loop's
    test and at the start of the loop's body.
  """
  graph = graph or control_flow.get_control_flow_graph(node)
  test_block = graph.get_block_by_ast_node(node.test)

  for block in graph.get_exit_blocks():
    analysis = LivenessAnalysis()
    analysis.visit(block)
  # TODO(dbieber): Move this logic into the Analysis class to avoid the use of
  # magic strings.
  live_variables = test_block.get_label('liveness_in')
  written_variables = {
      write.id
      for write in instruction_module.get_writes_from_ast_node(node)
      if isinstance(write, ast.Name)
  }
  return live_variables & written_variables


class LivenessAnalysis(Analysis):
  """Liveness analysis by basic block.

  In the liveness analysis, the in_value of a block is the set of variables
  that are live at the start of a block. "Live" means that the current value of
  the variable may be used later in the execution. The out_value of a block is
  the set of variable identifiers that are live at the end of the block.

  Since this is a backward analysis, the "before_value" is the out_value and the
  "after_value" is the in_value.
  """

  def __init__(self):
    super(LivenessAnalysis, self).__init__(label='liveness', forward=False)

  def aggregate_previous_after_values(self, previous_after_values):
    """Computes the out_value (before_value) of a block.

    Args:
      previous_after_values: A list of the sets of live variables at the start
        of each of the blocks following the current block.
    Returns:
      The set of live variables at the end of the current block. This is the
      union of live variable sets at the start of each subsequent block.
    """
    result = set()
    for before_value in previous_after_values:
      result |= before_value
    return frozenset(result)

  def compute_after_value(self, block, before_value):
    """Computes the liveness analysis gen and kill sets for a basic block.

    The gen set is the set of variables read by the block before they are
    written to.
    The kill set is the set of variables written to by the basic block.

    Args:
      block: The BasicBlock to analyze.
      before_value: The out_value for block (the set of variables live at the
        end of the block.)
    Returns:
      The in_value for block (the set of variables live at the start of the
      block).
    """
    gen = set()
    kill = set()
    for control_flow_node in block.control_flow_nodes:
      instruction = control_flow_node.instruction
      for read in instruction.get_read_names():
        if read not in kill:
          gen.add(read)
      kill.update(instruction.get_write_names())
    return frozenset((before_value - kill) | gen)


class FrozenDict(dict):

  def __hash__(self):
    return hash(tuple(sorted(self.items())))


class LastAccessAnalysis(Analysis):
  """Computes for each variable its possible last reads and last writes."""

  def __init__(self):
    super(LastAccessAnalysis, self).__init__(label='last_access', forward=True)

  def aggregate_previous_after_values(self, previous_after_values):
    result = collections.defaultdict(frozenset)
    for previous_after_value in previous_after_values:
      for key, value in previous_after_value.items():
        result[key] |= value
    return FrozenDict(result)

  def compute_after_value(self, node, before_value):
    result = before_value.copy()
    for access in node.instruction.accesses:
      kind_and_name = instruction_module.access_kind_and_name(access)
      result[kind_and_name] = frozenset([access])
    return FrozenDict(result)
