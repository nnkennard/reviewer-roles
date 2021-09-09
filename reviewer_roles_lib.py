import collections
import re



from math import pi
from matplotlib import pyplot as plt

NodeRule = collections.namedtuple("NodeRule",
                                  "node feature threshold samples".split())
TerminalNode = collections.namedtuple("TerminalNode",
                                      "node label samples mistakes".split())


class MyTreeNode(object):

  def __init__(self, label, left_child=None, right_child=None):
    self.label = label
    self.left_child = left_child
    self.right_child = right_child

  def __str__(self):
    return " ".join(
        [str(self.label),
         str(self.left_child),
         str(self.right_child)])


def tree_dfs_helper(i, tree_nodes, node_rule_map, terminal_node_map,
                    path_so_far):
  for (direction, child) in [("l", tree_nodes[i].left_child),
                             ("r", tree_nodes[i].right_child)]:
    if child is None:
      print(terminal_node_map[str(i)].label, end=": ")
      for i in range(0, len(path_so_far) - 1, 2):
        node = path_so_far[i]
        child_dir = path_so_far[i + 1]
        rule_node = node_rule_map[str(node)]
        if child_dir == "l":
          print(rule_node.feature, "<=", rule_node.threshold, end="; ")
        else:
          print(rule_node.feature, ">", rule_node.threshold, end="; ")
      print()
      return
    tree_dfs_helper(child, tree_nodes, node_rule_map, terminal_node_map,
                    path_so_far + [direction, child])


def tree_parser(tree_filename, feature_vectorizer):
  feature_list = feature_vectorizer.get_feature_names()
  with open(tree_filename, 'r') as f:
    tree_lines = f.readlines()
  label_lines = [line.strip() for line in tree_lines[1:] if '->' not in line]
  edge_lines = [line for line in tree_lines if '->' in line]
  node_rule_map = {}
  terminal_node_map = {}
  labels = re.findall(
      "n_.*?;",
      " ".join(label_lines),
  )
  for label in labels:
    if '<=' in label:
      match = re.match(
          "n_([0-9]+) \[label=\"([0-9]+)....([0-9]+\.?[0-9]*).samples..([0-9]+)",
          label)
      node, feature_index, thresh, samples = match.groups()
      node_rule_map[node] = NodeRule(node, feature_list[int(feature_index)],
                                     float(thresh), int(samples))
    else:
      match = re.match(
          "n_([0-9]+) \[label=\"([0-9]+).samples..([0-9]+).mistakes..([0-9]+)",
          label)
      node, label, samples, mistakes = match.groups()
      terminal_node_map[node] = TerminalNode(node, label, int(samples),
                                             int(mistakes))

  tree_nodes = [MyTreeNode(str(i), None, None) for i in range(len(labels))]
  for edge_line in edge_lines:
    match = re.match("n_([0-9]+)....n_([0-9]+)\;", edge_line)
    parent, child = match.groups()
    child_index = int(child)
    parent_node = tree_nodes[int(parent)]
    if parent_node.left_child is None:
      parent_node.left_child = child_index
    else:
      assert parent_node.right_child is None
      parent_node.right_child = child_index

  tree_dfs_helper(0, tree_nodes, node_rule_map, terminal_node_map, [0])


def get_angles_from_categories(categories):
  N = len(categories)
  angles = [n / float(N) * 2 * pi for n in range(N)]
  angles += angles[:1]
  return angles


def initialize_spiders(
    categories,
    final_category_names,
    titles,
    figsize=(1000, 1100),
    dpi=96,
):
  plt.figure(figsize=figsize, dpi=dpi)
  fig, axes = plt.subplots(2,
                           2,
                           subplot_kw=dict(projection="polar"),
                           figsize=(figsize[0] / dpi, figsize[1] / dpi))
  axes = axes.flatten()

  angles = get_angles_from_categories(categories)

  for index in range(4):

    ax = axes[index]

    # Set first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set x ticks
    ax.set_xticks(angles[:-1])
    ax.set_title(titles[index])
    ax.set_xticklabels(final_category_names,
                       fontdict={
                           "color": 'grey',
                           "size": 10
                       })

    # Set y ticks
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.50, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"],
                       fontdict={
                           "color": 'grey',
                           "size": 9
                       })
    ax.set_ylim(0, 1)

  return axes


def make_spider(ax, row_dict, color, categories):

  values = [row_dict[k] for k in categories] + [row_dict[categories[0]]]
  angles = get_angles_from_categories(categories)
  ax.plot(angles,
                       values,
                       color=color,
                       linewidth=2,
                       linestyle='solid')
  ax.fill(angles, values, color=color, alpha=0.4)
