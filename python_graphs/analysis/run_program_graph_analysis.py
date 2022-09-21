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

"""Runs the program graph analysis for datasets of programs.

Analyzes each dataset of programs, producing plots for properties such as the
AST height.
"""

import inspect
import math

from absl import app
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
from python_graphs import control_flow_test_components as cftc
from python_graphs import program_graph
from python_graphs import program_graph_test_components as pgtc
from python_graphs.analysis import program_graph_analysis
import six
from six.moves import range



TARGET_NUM_BINS = 15  # A reasonable constant number of histogram bins.
MAX_NUM_BINS = 20   # The maximum number of bins reasonable on a histogram.


def test_components():
  """Generates functions from two sets of test components.

  Yields:
    All functions in the program graph and control flow test components files.
  """
  for unused_name, fn in inspect.getmembers(pgtc, predicate=inspect.isfunction):
    yield fn

  for unused_name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
    yield fn




def get_graph_generator(function_generator):
  """Generates ProgramGraph objects from functions.

  Args:
    function_generator: A function generator.

  Yields:
    ProgramGraph objects for the functions.
  """
  for index, function in enumerate(function_generator):
    try:
      graph = program_graph.get_program_graph(function)
      yield graph
    except SyntaxError:
      # get_program_graph can fail for programs with different string encodings.
      logging.info('SyntaxError in get_program_graph for function index %d. '
                   'First 100 chars of function source:\n%s',
                   index, function[:100])
    except RuntimeError:
      # get_program_graph can fail for programs that are only return statements.
      logging.info('RuntimeError in get_program_graph for function index %d. '
                   'First 100 chars of function source:\n%s',
                   index, function[:100])


def get_percentiles(data, percentiles, integer_valued=True):
  """Returns a dict of percentiles of the data.

  Args:
    data: An unsorted list of datapoints.
    percentiles: A list of ints or floats in the range [0, 100] representing the
      percentiles to compute.
    integer_valued: Whether or not the values are all integers. If so,
      interpolate to the nearest datapoint (instead of computing a fractional
      value between the two nearest datapoints).

  Returns:
    A dict mapping each element of percentiles to the computed result.
  """
  # Ensure integer datapoints for cleaner binning if necessary.
  interpolation = 'nearest' if integer_valued else 'linear'
  results = np.percentile(data, percentiles, interpolation=interpolation)
  return {percentiles[i]: results[i] for i in range(len(percentiles))}


def analyze_graph(graph, identifier):
  """Performs various analyses on a graph.

  Args:
    graph: A ProgramGraph to analyze.
    identifier: A unique identifier for this graph (for later aggregation).

  Returns:
    A pair (identifier, result_dict), where result_dict contains the results of
    analyses run on the graph.
  """
  num_nodes = program_graph_analysis.num_nodes(graph)
  num_edges = program_graph_analysis.num_edges(graph)
  ast_height = program_graph_analysis.graph_ast_height(graph)

  degree_percentiles = [10, 25, 50, 75, 90]
  degrees = get_percentiles(program_graph_analysis.degrees(graph),
                            degree_percentiles)
  in_degrees = get_percentiles(program_graph_analysis.in_degrees(graph),
                               degree_percentiles)
  out_degrees = get_percentiles(program_graph_analysis.out_degrees(graph),
                                degree_percentiles)

  diameter = program_graph_analysis.diameter(graph)
  max_betweenness = program_graph_analysis.max_betweenness(graph)

  # TODO(kshi): Turn this into a protobuf and fix everywhere else in this file.
  # Eventually this should be parallelized (currently takes ~6 hours to run).
  result_dict = {
      'num_nodes': num_nodes,
      'num_edges': num_edges,
      'ast_height': ast_height,
      'degrees': degrees,
      'in_degrees': in_degrees,
      'out_degrees': out_degrees,
      'diameter': diameter,
      'max_betweenness': max_betweenness,
  }

  return (identifier, result_dict)


def create_bins(values, integer_valued=True, log_x=False):
  """Creates appropriate histogram bins.

  Args:
    values: The values to be plotted in a histogram.
    integer_valued: Whether the values are all integers.
    log_x: Whether to plot the x-axis using a log scale.

  Returns:
    An object (sequence, integer, or 'auto') that can be used as the 'bins'
      keyword argument to plt.hist(). If there are no values to plot, or all of
      the values are identical, then 'auto' is returned.
  """
  if not values:
    return 'auto'  # No data to plot; let pyplot handle this case.
  min_value = min(values)
  max_value = max(values)
  if min_value == max_value:
    return 'auto'  # All values are identical; let pyplot handle this case.

  if log_x:
    return np.logspace(np.log10(min_value), np.log10(max_value + 1),
                       num=(TARGET_NUM_BINS + 1))
  elif integer_valued:
    # The minimum integer width resulting in at most MAX_NUM_BINS bins.
    bin_width = math.ceil((max_value - min_value + 1) / MAX_NUM_BINS)
    # Place bin boundaries between integers.
    return np.arange(min_value - 0.5, max_value + bin_width + 0.5, bin_width)
  else:
    return TARGET_NUM_BINS


def create_histogram(values, title, percentiles=False, integer_valued=True,
                     log_x=False, log_y=False):
  """Returns a histogram of integer values computed from a dataset.

  Args:
    values: A list of integer values to plot, or if percentiles is True, then
      each value is a dict mapping some chosen percentiles in [0, 100] to the
      corresponding data value.
    title: The figure title.
    percentiles: Whether to plot multiple histograms for percentiles.
    integer_valued: Whether the values are all integers, which affects how the
      data is partitioned into bins.
    log_x: Whether to plot the x-axis using a log scale.
    log_y: Whether to plot the y-axis using a log scale.

  Returns:
    A histogram figure.
  """
  figure = plt.figure()

  if percentiles:
    for percentile in sorted(values[0].keys()):
      new_values = [percentile_dict[percentile]
                    for percentile_dict in values]
      bins = create_bins(new_values, integer_valued=integer_valued, log_x=log_x)
      plt.hist(new_values, bins=bins, alpha=0.5, label='{}%'.format(percentile))
    plt.legend(loc='upper right')
  else:
    bins = create_bins(values, integer_valued=integer_valued, log_x=log_x)
    plt.hist(values, bins=bins)

  if log_x:
    plt.xscale('log', nonposx='clip')
  if log_y:
    plt.yscale('log', nonposy='clip')
  plt.title(title)
  return figure


def save_histogram(all_results, result_key, dataset_name, path_root,
                   percentiles=False, integer_valued=True,
                   log_x=False, log_y=False):
  """Saves a histogram image to disk.

  Args:
    all_results: A list of dicts containing all analysis results for each graph.
    result_key: The key in the result dicts specifying what data to plot.
    dataset_name: The name of the dataset, which appears in the figure title and
      the image filename.
    path_root: The directory to save the histogram image in.
    percentiles: Whether the data has multiple percentiles to plot.
    integer_valued: Whether the values are all integers, which affects how the
      data is partitioned into bins.
    log_x: Whether to plot the x-axis using a log scale.
    log_y: Whether to plot the y-axis using a log scale.
  """
  values = [result[result_key] for result in all_results]
  title = '{} distribution for {}'.format(result_key, dataset_name)
  figure = create_histogram(values, title, percentiles=percentiles,
                            integer_valued=integer_valued,
                            log_x=log_x, log_y=log_y)
  path = '{}/{}-{}.png'.format(path_root, result_key, dataset_name)
  figure.savefig(path)
  logging.info('Saved image %s', path)


def main(argv):
  del argv  # Unused.

  dataset_pairs = [
      (test_components(), 'test_components'),
  ]
  path_root = '/tmp/program_graph_analysis'

  for function_generator, dataset_name in dataset_pairs:
    logging.info('Analyzing graphs in dataset %s...', dataset_name)
    graph_generator = get_graph_generator(function_generator)
    all_results = []
    for index, graph in enumerate(graph_generator):
      identifier = '{}-{}'.format(dataset_name, index)
      # Discard the identifiers (not needed until this is parallelized).
      all_results.append(analyze_graph(graph, identifier)[1])

    if all_results:
      logging.info('Creating plots for dataset %s...', dataset_name)
      for result_key in ['num_nodes', 'num_edges']:
        save_histogram(all_results, result_key, dataset_name, path_root,
                       percentiles=False, integer_valued=True, log_x=True)
      for result_key in ['ast_height', 'diameter']:
        save_histogram(all_results, result_key, dataset_name, path_root,
                       percentiles=False, integer_valued=True)
      for result_key in ['max_betweenness']:
        save_histogram(all_results, result_key, dataset_name, path_root,
                       percentiles=False, integer_valued=False)
      for result_key in ['degrees', 'in_degrees', 'out_degrees']:
        save_histogram(all_results, result_key, dataset_name, path_root,
                       percentiles=True, integer_valued=True)
    else:
      logging.warn('Dataset %s is empty.', dataset_name)


if __name__ == '__main__':
  app.run(main)
