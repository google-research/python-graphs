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

"""An Instruction represents an executable unit of a Python program.

Almost all simple statements correspond to Instructions, except for statements
likes pass, continue, and break, whose effects are already represented in the
structure of the control-flow graph.

In addition to simple statements, assignments that take place outside of simple
statements such as implicitly in a function or class definition also correspond
to Instructions.

The complete set of places where Instructions occur in source are listed here:

1. <Instruction>  (Any node in INSTRUCTION_AST_NODES used as a statement.)
2. if <Instruction>: ... (elif is the same.)
3+4. for <Instruction> in <Instruction>: ...
5. while <Instruction>: ...
6. try: ... except <Instruction>: ...
7. TODO(dbieber): Test for "with <Instruction>:"...

In the code:

@decorator
def fn(args=defaults):
  body

Outside of the function definition, we get the following instructions:
8.  Each decorator is an Instruction.
9.  Each default is an Instruction.
10. The assignment of the function def to the function name is an Instruction.
Inside the function definition, we get the following instructions:
11. An Instruction for the assignment of values to the arguments.
(1, again) And then the body can consist of multiple Instructions too.

Likewise in the code:

@decorator
class C(object):
  body

The following are Instructions:
(8, again) Each decorator is an Instruction
12. The assignment of the class to the variable C is an Instruction.
(1, again) And then the body can consist of multiple Instructions too.
13. TODO(dbieber): The base class (object) is an Instruction too.
"""

import gast as ast
import six

# Types of accesses:
READ = 'read'
WRITE = 'write'

# Context lists
WRITE_CONTEXTS = (ast.Store, ast.Del, ast.Param, ast.AugStore)
READ_CONTEXTS = (ast.Load, ast.AugLoad)

# Sources of implicit writes:
CLASS = 'class'
FUNCTION = 'function'
ARGS = 'args'
KWARG = 'kwarg'
KWONLYARGS = 'kwonlyargs'
VARARG = 'vararg'
ITERATOR = 'iter'
EXCEPTION = 'exception'

INSTRUCTION_AST_NODES = (
    ast.Expr,  # expression_stmt
    ast.Assert,  # assert_stmt
    ast.Assign,  # assignment_stmt
    ast.AugAssign,  # augmented_assignment_stmt
    ast.Delete,  # del_stmt
    ast.Print,  # print_stmt
    ast.Return,  # return_stmt
    # ast.Yield,  # yield_stmt. ast.Yield nodes are contained in ast.Expr nodes.
    ast.Raise,  # raise_stmt
    ast.Import,  # import_stmt
    ast.ImportFrom,
    ast.Global,  # global_stmt
    ast.Exec,  # exec_stmt
)

# https://docs.python.org/2/reference/simple_stmts.html
SIMPLE_STATEMENT_AST_NODES = INSTRUCTION_AST_NODES + (
    ast.Pass,  # pass_stmt
    ast.Break,  # break_stmt
    ast.Continue,  # continue_stmt
)


def _canonicalize(node):
  if isinstance(node, list) and len(node) == 1:
    return _canonicalize(node[0])
  if isinstance(node, ast.Module):
    return _canonicalize(node.body)
  if isinstance(node, ast.Expr):
    return _canonicalize(node.value)
  return node


def represent_same_program(node1, node2):
  """Whether AST nodes node1 and node2 represent the same program syntactically.

  Two programs are the same syntactically is they have equivalent ASTs, up to
  some small changes. The context field of Name nodes can change without the
  syntax represented by the AST changing. This allows for example for the short
  program 'x' (a read) to match with a subprogram 'x' of 'x = 3' (in which x is
  a write), since these two programs are the same syntactically ('x' and 'x').

  Except for the context field of Name nodes, the two nodes are recursively
  checked for exact equality.

  Args:
    node1: An AST node. This can be an ast.AST object, a primitive, or a list of
      AST nodes (primitives or ast.AST objects).
    node2: An AST node. This can be an ast.AST object, a primitive, or a list of
      AST nodes (primitives or ast.AST objects).

  Returns:
    Whether the two nodes represent equivalent programs.
  """
  node1 = _canonicalize(node1)
  node2 = _canonicalize(node2)

  if type(node1) != type(node2):  # pylint: disable=unidiomatic-typecheck
    return False
  if not isinstance(node1, ast.AST):
    return node1 == node2

  fields1 = list(ast.iter_fields(node1))
  fields2 = list(ast.iter_fields(node2))
  if len(fields1) != len(fields2):
    return False

  for (field1, value1), (field2, value2) in zip(fields1, fields2):
    if field1 == 'ctx':
      continue
    if field1 != field2 or type(value1) is not type(value2):
      return False
    if isinstance(value1, list):
      for item1, item2 in zip(value1, value2):
        if not represent_same_program(item1, item2):
          return False
    elif not represent_same_program(value1, value2):
      return False

  return True


class AccessVisitor(ast.NodeVisitor):
  """Visitor that computes an ordered list of accesses.

  Accesses are ordered based on a depth-first traversal of the AST, using the
  order of fields defined in `gast`, except for Assign nodes, for which the RHS
  is ordered before the LHS.

  This may differ from Python execution semantics in two ways:

  - Both branches sides of short-circuit `and`/`or` expressions or conditional
    `X if Y else Z` expressions are considered to be evaluated, even if one of
    them is actually skipped at runtime.
  - For AST nodes whose field order doesn't match the Python interpreter's
    evaluation order, the field order is used instead. Most AST nodes match
    execution order, but some differ (e.g. for dictionary literals, the
    interpreter alternates evaluating keys and values, but the field order has
    all keys and then all values). Assignments are a special case; the
    AccessVisitor evaluates the RHS first even though the LHS occurs first in
    the expression.

  Attributes:
    accesses: List of accesses encountered by the visitor.
  """

  # TODO(dbieber): Include accesses of ast.Subscript and ast.Attribute targets.

  def __init__(self):
    self.accesses = []

  def visit_Name(self, node):
    """Visit a Name, adding it to the list of accesses."""
    self.accesses.append(node)

  def visit_Assign(self, node):
    """Visit an Assign, ordering RHS accesses before LHS accesses."""
    self.visit(node.value)
    for target in node.targets:
      self.visit(target)

  def visit_AugAssign(self, node):
    """Visit an AugAssign, which contains both a read and a write."""
    # An AugAssign is a read as well as a write, even with the ctx of a write.
    self.visit(node.value)
    # Add a read access if we are assigning to a name.
    if isinstance(node.target, ast.Name):
      # TODO(dbieber): Use a proper type instead of a tuple for accesses.
      self.accesses.append(('read', node.target, node))
    # Add the write access as normal.
    self.visit(node.target)


def get_accesses_from_ast_node(node):
  """Get all accesses for an AST node, in depth-first AST field order."""
  visitor = AccessVisitor()
  visitor.visit(node)
  return visitor.accesses


def get_reads_from_ast_node(ast_node):
  """Get all reads for an AST node, in depth-first AST field order.

  Args:
    ast_node: The AST node of interest.

  Returns:
    A list of writes performed by that AST node.
  """
  return [
      access for access in get_accesses_from_ast_node(ast_node)
      if access_is_read(access)
  ]


def get_writes_from_ast_node(ast_node):
  """Get all writes for an AST node, in depth-first AST field order.

  Args:
    ast_node: The AST node of interest.

  Returns:
    A list of writes performed by that AST node.
  """
  return [
      access for access in get_accesses_from_ast_node(ast_node)
      if access_is_write(access)
  ]


def create_writes(node, parent=None):
  # TODO(dbieber): Use a proper type instead of a tuple for accesses.
  if isinstance(node, ast.AST):
    return [
        ('write', n, parent) for n in ast.walk(node) if isinstance(n, ast.Name)
    ]
  else:
    return [('write', node, parent)]


def access_is_read(access):
  if isinstance(access, ast.AST):
    assert isinstance(access, ast.Name), access
    return isinstance(access.ctx, READ_CONTEXTS)
  else:
    return access[0] == 'read'


def access_is_write(access):
  if isinstance(access, ast.AST):
    assert isinstance(access, ast.Name), access
    return isinstance(access.ctx, WRITE_CONTEXTS)
  else:
    return access[0] == 'write'


def access_name(access):
  if isinstance(access, ast.AST):
    return access.id
  elif isinstance(access, tuple):
    if isinstance(access[1], six.string_types):
      return access[1]
    elif isinstance(access[1], ast.Name):
      return access[1].id
  raise ValueError('Unexpected access type.', access)


def access_kind(access):
  if access_is_read(access):
    return 'read'
  elif access_is_write(access):
    return 'write'


def access_kind_and_name(access):
  return '{}-{}'.format(access_kind(access), access_name(access))


def access_identifier(name, kind):
  return '{}-{}'.format(kind, name)


class Instruction(object):
  # pyformat:disable
  """Represents an executable unit of a Python program.

  An Instruction is a part of an AST corresponding to a simple statement or
  assignment, not corresponding to control flow. The part of the AST is not
  necessarily an AST node. It may be an AST node, or it may instead be a string
  (such as a variable name).

  Instructions play an important part in control flow graphs. An Instruction
  is the smallest unit of a control flow graph (wrapped in a ControlFlowNode).
  A control flow graph consists of basic blocks which represent a sequence of
  Instructions that are executed in a straight-line manner, or not at all.

  Conceptually an Instruction is immutable. This means that while Python does
  permit the mutation of an Instruction, in practice an Instruction object
  should not be modified once it is created.

  Note that an Instruction may be interrupted by an exception mid-execution.
  This is captured in control flow graphs via interrupting exits from basic
  blocks to either exception handlers or special 'raises' blocks.

  In addition to pure simple statements, an Instruction can represent a number
  of different parts of code. These are all listed explicitly in the module
  docstring.

  In the common case, the accesses made by an Instruction are given by the Name
  AST nodes contained in the Instruction's AST node. In some cases, when the
  instruction.source field is not None, the accesses made by an Instruction are
  not simply the Name AST nodes of the Instruction's node. For example, in a
  function definition, the only access is the assignment of the function def to
  the variable with the function's name; the Name nodes contained in the
  function definition are not part of the function definition Instruction, and
  instead are part of other Instructions that make up the function. The set of
  accesses made by an Instruction is computed when the Instruction is created
  and available via the accesses attribute of the Instruction.

  Attributes:
    node: The AST node corresponding to the instruction.
    accesses: (optional) An ordered list of all reads and writes made by this
      instruction. Each item in `accesses` is one of either:
        - A 3-tuple with fields (kind, node, parent). kind is either 'read' or
          'write'. node is either a string or Name AST node. parent is an AST
          node where node occurs.
        - A Name AST node
        # TODO(dbieber): Use a single type for all accesses.
    source: (optional) The source of the writes. For example in the for loop
      `for x in items: pass` there is a instruction for the Name node "x". Its
        source is ITERATOR, indicating that this instruction corresponds to x
        being assigned a value from an iterator. When source is not None, the
        Python code corresponding to the instruction does not coincide with the
        Python code corresponding to the instruction's node.
  """
  # pyformat:enable

  def __init__(self, node, accesses=None, source=None):
    if not isinstance(node, ast.AST):
      raise TypeError('node must be an instance of ast.AST.', node)
    self.node = node
    if accesses is None:
      accesses = get_accesses_from_ast_node(node)
    self.accesses = accesses
    self.source = source

  def contains_subprogram(self, node):
    """Whether this Instruction contains the given AST as a subprogram.

    Computes whether `node` is a subtree of this Instruction's AST.
    If the Instruction represents an implied write, then the node must match
    against the Instruction's writes.

    Args:
      node: The node to check the instruction against for a match.

    Returns:
      (bool) Whether or not this Instruction contains the node, syntactically.
    """
    if self.source is not None:
      # Only exact matches are permissible if source is not None.
      return represent_same_program(node, self.node)
    for subtree in ast.walk(self.node):
      if represent_same_program(node, subtree):
        return True
    return False

  def get_reads(self):
    return {access for access in self.accesses if access_is_read(access)}

  def get_read_names(self):
    return {access_name(access) for access in self.get_reads()}

  def get_writes(self):
    return {access for access in self.accesses if access_is_write(access)}

  def get_write_names(self):
    return {access_name(access) for access in self.get_writes()}
