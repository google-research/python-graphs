# python_graphs

This package is for computing graph representations of Python programs for
machine learning applications. It includes the following modules:

*   `control_flow` For computing control flow graphs statically from Python
    programs.
*   `data_flow` For computing data flow analyses of Python programs.
*   `program_graph` For computing graphs statically to represent arbitrary
    Python programs or functions.
*   `cyclomatic_complexity` For computing the cyclomatic complexity of a Python function.


## Installation

To install python_graphs with pip, run: `pip install python_graphs`.

To install python_graphs from source, run: `python setup.py develop`.

## Common Tasks

**Generate a control flow graph from a function `fn`:**

```python
from python_graphs import control_flow
graph = control_flow.get_control_flow_graph(fn)
```

**Generate a program graph from a function `fn`:**

```python
from python_graphs import program_graph
graph = program_graph.get_program_graph(fn)
```

**Compute the cyclomatic complexity of a function `fn`:**

```python
from python_graphs import control_flow
from python_graphs import cyclomatic_complexity
graph = control_flow.get_control_flow_graph(fn)
value = cyclomatic_complexity.cyclomatic_complexity(graph)
```

---

This is not an officially supported Google product.
