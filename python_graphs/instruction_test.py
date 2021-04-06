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

"""Tests for instruction module."""

from absl.testing import absltest
import gast as ast
from python_graphs import instruction as instruction_module


def create_instruction(source):
  node = ast.parse(source)
  node = instruction_module._canonicalize(node)
  return instruction_module.Instruction(node)


class InstructionTest(absltest.TestCase):

  def test_instruction(self):
    self.assertIsNotNone(instruction_module.Instruction)

  def test_represent_same_program_basic_positive_case(self):
    program1 = ast.parse('x + 1')
    program2 = ast.parse('x + 1')
    self.assertTrue(
        instruction_module.represent_same_program(program1, program2))

  def test_represent_same_program_basic_negative_case(self):
    program1 = ast.parse('x + 1')
    program2 = ast.parse('x + 2')
    self.assertFalse(
        instruction_module.represent_same_program(program1, program2))

  def test_represent_same_program_different_contexts(self):
    full_program1 = ast.parse('y = x + 1')  # y is a write
    program1 = full_program1.body[0].targets[0]  # 'y'
    program2 = ast.parse('y')  # y is a read
    self.assertTrue(
        instruction_module.represent_same_program(program1, program2))

  def test_get_accesses(self):
    instruction = create_instruction('x + 1')
    self.assertEqual(instruction.get_read_names(), {'x'})
    self.assertEqual(instruction.get_write_names(), set())

    instruction = create_instruction('return x + y + z')
    self.assertEqual(instruction.get_read_names(), {'x', 'y', 'z'})
    self.assertEqual(instruction.get_write_names(), set())

    instruction = create_instruction('fn(a, b, c)')
    self.assertEqual(instruction.get_read_names(), {'a', 'b', 'c', 'fn'})
    self.assertEqual(instruction.get_write_names(), set())

    instruction = create_instruction('c = fn(a, b, c)')
    self.assertEqual(instruction.get_read_names(), {'a', 'b', 'c', 'fn'})
    self.assertEqual(instruction.get_write_names(), {'c'})

  def test_get_accesses_augassign(self):
    instruction = create_instruction('x += 1')
    self.assertEqual(instruction.get_read_names(), {'x'})
    self.assertEqual(instruction.get_write_names(), {'x'})

    instruction = create_instruction('x *= y')
    self.assertEqual(instruction.get_read_names(), {'x', 'y'})
    self.assertEqual(instruction.get_write_names(), {'x'})

  def test_get_accesses_augassign_subscript(self):
    instruction = create_instruction('x[0] *= y')
    # This is not currently considered a write of x. It is a read of x.
    self.assertEqual(instruction.get_read_names(), {'x', 'y'})
    self.assertEqual(instruction.get_write_names(), set())

  def test_get_accesses_augassign_attribute(self):
    instruction = create_instruction('x.attribute *= y')
    # This is not currently considered a write of x. It is a read of x.
    self.assertEqual(instruction.get_read_names(), {'x', 'y'})
    self.assertEqual(instruction.get_write_names(), set())

  def test_get_accesses_subscript(self):
    instruction = create_instruction('x[0] = y')
    # This is not currently considered a write of x. It is a read of x.
    self.assertEqual(instruction.get_read_names(), {'x', 'y'})
    self.assertEqual(instruction.get_write_names(), set())

  def test_get_accesses_attribute(self):
    instruction = create_instruction('x.attribute = y')
    # This is not currently considered a write of x. It is a read of x.
    self.assertEqual(instruction.get_read_names(), {'x', 'y'})
    self.assertEqual(instruction.get_write_names(), set())

  def test_access_ordering(self):
    instruction = create_instruction('c = fn(a, b + c, d / a)')
    access_names_and_kinds = [(instruction_module.access_name(access),
                               instruction_module.access_kind(access))
                              for access in instruction.accesses]
    self.assertEqual(access_names_and_kinds, [('fn', 'read'), ('a', 'read'),
                                              ('b', 'read'), ('c', 'read'),
                                              ('d', 'read'), ('a', 'read'),
                                              ('c', 'write')])

    instruction = create_instruction('c += fn(a, b + c, d / a)')
    access_names_and_kinds = [(instruction_module.access_name(access),
                               instruction_module.access_kind(access))
                              for access in instruction.accesses]
    self.assertEqual(access_names_and_kinds, [('fn', 'read'), ('a', 'read'),
                                              ('b', 'read'), ('c', 'read'),
                                              ('d', 'read'), ('a', 'read'),
                                              ('c', 'read'), ('c', 'write')])


if __name__ == '__main__':
  absltest.main()
