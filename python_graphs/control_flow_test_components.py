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

"""Test components for testing control flow.

Many of these components would produce RuntimeErrors if run. Their purpose is
for the testing of the control_flow module.
"""


# pylint: disable=missing-docstring
# pylint: disable=pointless-statement,undefined-variable
# pylint: disable=unused-variable,unused-argument
# pylint: disable=bare-except,lost-exception,unreachable
# pylint: disable=keyword-arg-before-vararg
def straight_line_code():
  x = 1
  y = x + 2
  z = y * 3
  return z


def simple_if_statement():
  x = 1
  y = 2
  if x > y:
    y = 3
  return y


def simple_for_loop():
  x = 1
  for y in range(x + 2):
    z = y + 3
  return z


def tuple_in_for_loop():
  a, b = 0, 1
  for a, b in [(1, 2), (2, 3)]:
    if a > b:
      break
  return b - a


def simple_while_loop():
  x = 1
  while x < 2:
    x += 3
  return x


def break_in_while_loop():
  x = 1
  while x < 2:
    x += 3
    break
  return x


def nested_while_loops():
  x = 1
  while x < 2:
    y = 3
    while y < 4:
      y += 5
    x += 6
  return x


def multiple_excepts():
  try:
    x = 1
  except ValueError:
    x = 2
    x = 3
  except RuntimeError:
    x = 4
  except:
    x = 5
  return x


def try_finally():
  header0
  try:
    try0
    try1
  except Exception0 as value0:
    exception0_stmt0
  finally:
    finally_stmt0
    finally_stmt1
  after0


def exception_handling():
  try:
    before_stmt0
    before_stmt1
    try:
      try_block
    except error_type as value:
      except_block1
    after_stmt0
    after_stmt1
  except:
    except_block2_stmt0
    except_block2_stmt1
  finally:
    final_block_stmt0
    final_block_stmt1
  end_block_stmt0
  end_block_stmt1


def fn_with_args(a, b=10, *varargs, **kwargs):
  body_stmt0
  body_stmt1
  return


def fn1(a, b):
  return a + b


def fn2(a, b):
  c = a
  if a > b:
    c -= b
  return c


def fn3(a, b):
  c = a
  if a > b:
    c -= b
    c += 1
    c += 2
    c += 3
  else:
    c += b
  return c


def fn4(i):
  count = 0
  for i in range(i):
    count += 1
  return count


def fn5(i):
  count = 0
  for _ in range(i):
    if count > 5:
      break
    count += 1
  return count


def fn6():
  count = 0
  while count < 10:
    count += 1
  return count


def fn7():
  try:
    raise ValueError('This will be caught.')
  except ValueError as e:
    del e
  return


def try_with_else():
  try:
    raise ValueError('This will be caught.')
  except ValueError as e:
    del e
  else:
    return 1
  return 2


def for_with_else():
  for target in iterator:
    for_stmt0
    if condition:
      break
    for_stmt1
  else:
    else_stmt0
    else_stmt1
  after_stmt0


def fn8(a):
  a += 1


def nested_loops(a):
  """A test function illustrating nested loops."""
  for i in range(a):
    while True:
      break
      unreachable = 10
    for j in range(i):
      for k in range(j):
        if j * k > 10:
          continue
          unreachable = 5
      if i + j == 10:
        return True
  return False


def try_with_loop():
  before_stmt0
  try:
    for target in iterator:
      for_body0
      for_body1
  except:
    except_body0
    except_body1
  after_stmt0


def break_in_finally():
  header0
  for target0 in iter0:
    try:
      try0
      try1
    except Exception0 as value0:
      exception0_stmt0
    except Exception1 as value1:
      exception1_stmt0
      exception1_stmt1
    finally:
      finally_stmt0
      finally_stmt1
      # This breaks out of the for-loop.
      break
  after0


def break_in_try():
  count = 0
  for _ in range(10):
    try:
      count += 1
      # This breaks out of the for-loop through the finally block.
      break
    except ValueError:
      pass
    finally:
      count += 2
  return count


def nested_try_excepts():
  try:
    try:
      x = 0
      x += 1
      try:
        x = 2 + 2
      except ValueError(1+1) as e:
        x = 3 - 3
      finally:
        x = 4
    except RuntimeError:
      x = 5 * 5
    finally:
      x = 6 ** 6
  except:
    x = 7 / 7
  return x


def multi_op_expression():
  return 1 + 2 * 3


def create_lambda():
  before_stmt0
  fn = lambda args: output
  after_stmt0


def generator():
  for target in iterator:
    yield yield_statement
    after_stmt0


def fn_with_inner_fn():
  def inner_fn():
    x = 10
    while True:
      pass


class ExampleClass(object):

  def method0(self, arg):
    method_stmt0
    method_stmt1
