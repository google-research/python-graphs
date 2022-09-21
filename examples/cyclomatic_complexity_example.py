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

"""Example computing the cyclomatic complexity of various Python functions.

For each of the functions in control_flow_test_components.py, this computes and
prints the function's cyclomatic complexity.

Usage:
python -m examples.cyclomatic_complexity_example
"""

import inspect

from absl import app

from python_graphs import control_flow
from python_graphs import control_flow_test_components as tc
from python_graphs import cyclomatic_complexity


def main(argv) -> None:
  del argv  # Unused

  # For each function in control_flow_test_components.py, compute its cyclomatic
  # complexity and print the result.
  for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
    print(f'{name}: ', end='')
    graph = control_flow.get_control_flow_graph(fn)
    value = cyclomatic_complexity.cyclomatic_complexity(graph)
    print(value)


if __name__ == '__main__':
  app.run(main)
