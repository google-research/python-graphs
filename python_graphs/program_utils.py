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

"""Program utility functions."""

import inspect
import textwrap
import uuid

import gast as ast
import six


def getsource(obj):
  """Gets the source for the given object.

  Args:
    obj: A module, class, method, function, traceback, frame, or code object.
  Returns:
    The source of the object, if available.
  """
  if inspect.ismethod(obj):
    func = obj.__func__
  else:
    func = obj
  source = inspect.getsource(func)
  return textwrap.dedent(source)


def program_to_ast(program):
  """Convert a program to its AST.

  Args:
    program: Either an AST node, source string, or a function.
  Returns:
    The root AST node of the AST representing the program.
  """
  if isinstance(program, ast.AST):
    return program
  if isinstance(program, six.string_types):
    source = program
  else:
    source = getsource(program)
  module_node = ast.parse(source, mode='exec')
  return module_node


def unique_id():
  """Returns a unique id that is suitable for identifying graph nodes."""
  return uuid.uuid4().int & ((1 << 64) - 1)

