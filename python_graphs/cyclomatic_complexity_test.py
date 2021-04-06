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

"""Tests for cyclomatic_complexity.py."""

from absl.testing import absltest
from absl.testing import parameterized

from python_graphs import control_flow
from python_graphs import control_flow_test_components as tc
from python_graphs import cyclomatic_complexity


class CyclomaticComplexityTest(parameterized.TestCase):

  @parameterized.parameters(
      (tc.straight_line_code, 1),
      (tc.simple_if_statement, 2),
      (tc.simple_for_loop, 2),
  )
  def test_cyclomatic_complexity(self, component, target_value):
    graph = control_flow.get_control_flow_graph(component)
    value = cyclomatic_complexity.cyclomatic_complexity(graph)
    self.assertEqual(value, target_value)

if __name__ == '__main__':
  absltest.main()
