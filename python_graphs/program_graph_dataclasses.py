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

"""The dataclasses for representing a Program Graph."""

import enum
from typing import List, Optional, Text
import dataclasses


class NodeType(enum.Enum):
  UNSPECIFIED = 0
  AST_NODE = 1
  AST_LIST = 2
  AST_VALUE = 3
  SYNTAX_NODE = 4
  PLACEHOLDER = 5


@dataclasses.dataclass
class Node:
  """Represents a node in a program graph."""
  id: int
  type: NodeType

  # If an AST node, a string that identifies what type of AST node,
  # e.g. "Num" or "Expr". These are defined by the underlying AST for the
  # language.
  ast_type: Optional[Text] = ""

  # Primitive valued AST node, such as:
  # - the name of an identifier for a Name node
  # - the number attached to a Num node
  # The corresponding ast_type value is the Python type of ast_value, not the
  # type of the parent AST node.
  ast_value_repr: Optional[Text] = ""

  # For syntax nodes, the syntax attached to the node.
  syntax: Optional[Text] = ""


class EdgeType(enum.Enum):
  """The different kinds of edges that can appear in a program graph."""
  UNSPECIFIED = 0
  CFG_NEXT = 1
  LAST_READ = 2
  LAST_WRITE = 3
  COMPUTED_FROM = 4
  RETURNS_TO = 5
  FORMAL_ARG_NAME = 6
  FIELD = 7
  SYNTAX = 8
  NEXT_SYNTAX = 9
  LAST_LEXICAL_USE = 10
  CALLS = 11


@dataclasses.dataclass
class Edge:
  id1: int
  id2: int
  type: EdgeType
  field_name: Optional[Text] = None  # For FIELD edges, the field name.
  has_back_edge: bool = False


@dataclasses.dataclass
class Graph:
  nodes: List[Node]
  edges: List[Edge]
  root_id: int
