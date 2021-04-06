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

"""Computes the cyclomatic complexity of a program or control flow graph."""


def cyclomatic_complexity(control_flow_graph):
  """Computes the cyclomatic complexity of a function from its cfg."""
  enter_block = next(control_flow_graph.get_enter_blocks())

  new_blocks = []
  seen_block_ids = set()
  new_blocks.append(enter_block)
  seen_block_ids.add(id(enter_block))
  num_edges = 0

  while new_blocks:
    block = new_blocks.pop()
    for next_block in block.exits_from_end:
      num_edges += 1
      if id(next_block) not in seen_block_ids:
        new_blocks.append(next_block)
        seen_block_ids.add(id(next_block))
  num_nodes = len(seen_block_ids)

  p = 1  # num_connected_components
  e = num_edges
  n = num_nodes
  return e - n + 2 * p


def cyclomatic_complexity2(control_flow_graph):
  """Computes the cyclomatic complexity of a program from its cfg."""
  # Assumes a single connected component.
  p = 1  # num_connected_components
  e = sum(len(block.exits_from_end) for block in control_flow_graph.blocks)
  n = len(control_flow_graph.blocks)
  return e - n + 2 * p
