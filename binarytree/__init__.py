__all__ = ["Node", "tree", "bst", "heap", "build", "get_parent", "__version__"]

import heapq
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from graphviz import Digraph, nohtml
from pkg_resources import get_distribution

from copy import deepcopy

from binarytree.exceptions import (
    NodeIndexError,
    NodeModifyError,
    NodeNotFoundError,
    NodeReferenceError,
    NodeTypeError,
    NodeValueError,
    TreeHeightError,
)

__version__ = get_distribution("binarytree").version

LEFT_FIELD = "left"
RIGHT_FIELD = "right"
VAL_FIELD = "val"
VALUE_FIELD = "value"

NodeValue = Union[float, int, str]
NodeProperty = Union[float, int, bool]
NodeValueTup = (float, int, str)


class Node:
    """Represents a binary tree node.

    This class provides methods and properties for managing the current node,
    and the binary tree in which the node is the root. When a docstring in
    this class mentions "binary tree", it is referring to the current node and
    its descendants.

    :param value: Node value (must be a number).
    :type value: int | float
    :param left: Left child node (default: None).
    :type left: binarytree.Node
    :param right: Right child node (default: None).
    :type right: binarytree.Node
    :raise binarytree.exceptions.NodeTypeError: If left or right child node is
        not an instance of :class:`binarytree.Node`.
    :raise binarytree.exceptions.NodeValueError: If node value is not a number
        (e.g. int, float).
    """

    def __init__(
        self,
        value: NodeValue,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
    ) -> None:
        self.value = self.val = value
        self.left = left
        self.right = right

        if not isinstance(value, NodeValueTup):
            raise NodeValueError("node value must be a float or int")

        if left is not None and not isinstance(left, Node):
            raise NodeTypeError("left child must be a Node instance")

        if right is not None and not isinstance(right, Node):
            raise NodeTypeError("right child must be a Node instance")

    def __repr__(self) -> str:
        """Return the string representation of the current node.

        :return: String representation.
        :rtype: str

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> Node(1)
            Node(1)
        """
        return "Node({})".format(self.val)

    def __str__(self) -> str:
        """Return the pretty-print string for the binary tree.

        :return: Pretty-print string.
        :rtype: str

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> print(root)
            <BLANKLINE>
              __1
             /   \\
            2     3
             \\
              4
            <BLANKLINE>

        .. note::
            To include level-order_ indexes in the output string, use
            :func:`binarytree.Node.pprint` instead.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
        """
        lines = _build_tree_string(self, 0, False, "-")[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __setattr__(self, attr: str, obj: Any) -> None:
        """Modified version of ``__setattr__`` with extra sanity checking.

        Class attributes **left**, **right** and **value** are validated.

        :param attr: Name of the class attribute.
        :type attr: str
        :param obj: Object to set.
        :type obj: object
        :raise binarytree.exceptions.NodeTypeError: If left or right child is
            not an instance of :class:`binarytree.Node`.
        :raise binarytree.exceptions.NodeValueError: If node value is not a
            number (e.g. int, float).

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> node = Node(1)
            >>> node.left = 'invalid'  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeTypeError: Left child must be a Node instance

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> node = Node(1)
            >>> node.val = 'invalid'  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeValueError: node value must be a float or int
        """
        if attr == LEFT_FIELD:
            if obj is not None and not isinstance(obj, Node):
                raise NodeTypeError("left child must be a Node instance")
        elif attr == RIGHT_FIELD:
            if obj is not None and not isinstance(obj, Node):
                raise NodeTypeError("right child must be a Node instance")
        elif attr == VALUE_FIELD:
            if not isinstance(obj, NodeValueTup):
                raise NodeValueError("node value must be a float or int")
            object.__setattr__(self, VAL_FIELD, obj)
        elif attr == VAL_FIELD:
            if not isinstance(obj, NodeValueTup):
                raise NodeValueError("node value must be a float or int")
            object.__setattr__(self, VALUE_FIELD, obj)

        object.__setattr__(self, attr, obj)

    def __iter__(self) -> Iterator["Node"]:
        """Iterate through the nodes in the binary tree in level-order_.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :return: Node iterator.
        :rtype: (binarytree.Node)

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> list(root)
            [Node(1), Node(2), Node(3), Node(4), Node(5)]
        """
        current_level = [self]

        while len(current_level) > 0:
            next_level = []
            for node in current_level:
                yield node
                if node.left is not None:
                    next_level.append(node.left)
                if node.right is not None:
                    next_level.append(node.right)
            current_level = next_level

    def __len__(self) -> int:
        """Return the total number of nodes in the binary tree.

        :return: Total number of nodes.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>>
            >>> len(root)
            3

        .. note::
            This method is equivalent to :attr:`binarytree.Node.size`.
        """
        return self.properties["size"]

    def __getitem__(self, index: int) -> "Node":
        """Return the node (or subtree) at the given level-order_ index.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :param index: Level-order index of the node.
        :type index: int
        :return: Node (or subtree) at the given index.
        :rtype: binarytree.Node
        :raise binarytree.exceptions.NodeIndexError: If node index is invalid.
        :raise binarytree.exceptions.NodeNotFoundError: If the node is missing.

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[0]
            Node(1)
            >>> root[1]
            Node(2)
            >>> root[2]
            Node(3)
            >>> root[3]  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeNotFoundError: node missing at index 3
        """
        if not isinstance(index, int) or index < 0:
            raise NodeIndexError("node index must be a non-negative int")

        current_level: List[Optional[Node]] = [self]
        current_index = 0
        has_more_nodes = True

        while has_more_nodes:
            has_more_nodes = False
            next_level: List[Optional[Node]] = []

            for node in current_level:
                if current_index == index:
                    if node is None:
                        break
                    else:
                        return node
                current_index += 1

                if node is None:
                    next_level.append(None)
                    next_level.append(None)
                    continue
                next_level.append(node.left)
                next_level.append(node.right)
                if node.left is not None or node.right is not None:
                    has_more_nodes = True

            current_level = next_level

        raise NodeNotFoundError("node missing at index {}".format(index))

    def __setitem__(self, index: int, node: "Node") -> None:
        """Insert a node (or subtree) at the given level-order_ index.

        * An exception is raised if the parent node is missing.
        * Any existing node or subtree is overwritten.
        * Root node (current node) cannot be replaced.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :param index: Level-order index of the node.
        :type index: int
        :param node: Node to insert.
        :type node: binarytree.Node
        :raise binarytree.exceptions.NodeTypeError: If new node is not an
            instance of :class:`binarytree.Node`.
        :raise binarytree.exceptions.NodeNotFoundError: If parent is missing.
        :raise binarytree.exceptions.NodeModifyError: If user attempts to
            overwrite the root node (current node).

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[0] = Node(4)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeModifyError: cannot modify the root node

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[11] = Node(4)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeNotFoundError: parent node missing at index 5

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[1] = Node(4)
            >>>
            >>> root.left
            Node(4)
        """
        if index == 0:
            raise NodeModifyError("cannot modify the root node")

        parent_index = (index - 1) // 2
        try:
            parent = self.__getitem__(parent_index)
        except NodeNotFoundError:
            raise NodeNotFoundError(
                "parent node missing at index {}".format(parent_index)
            )

        setattr(parent, LEFT_FIELD if index % 2 else RIGHT_FIELD, node)

    def __delitem__(self, index: int) -> None:
        """Remove the node (or subtree) at the given level-order_ index.

        * An exception is raised if the target node is missing.
        * The descendants of the target node (if any) are also removed.
        * Root node (current node) cannot be deleted.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :param index: Level-order index of the node.
        :type index: int
        :raise binarytree.exceptions.NodeNotFoundError: If the target node or
            its parent is missing.
        :raise binarytree.exceptions.NodeModifyError: If user attempts to
            delete the root node (current node).

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)          # index: 0, value: 1
            >>> root.left = Node(2)     # index: 1, value: 2
            >>> root.right = Node(3)    # index: 2, value: 3
            >>>
            >>> del root[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeModifyError: cannot delete the root node

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)          # index: 0, value: 1
            >>> root.left = Node(2)     # index: 1, value: 2
            >>> root.right = Node(3)    # index: 2, value: 3
            >>>
            >>> del root[2]
            >>>
            >>> root[2]  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeNotFoundError: node missing at index 2
        """
        if index == 0:
            raise NodeModifyError("cannot delete the root node")

        parent_index = (index - 1) // 2
        try:
            parent = self.__getitem__(parent_index)
        except NodeNotFoundError:
            raise NodeNotFoundError("no node to delete at index {}".format(index))

        child_attr = LEFT_FIELD if index % 2 == 1 else RIGHT_FIELD
        if getattr(parent, child_attr) is None:
            raise NodeNotFoundError("no node to delete at index {}".format(index))

        setattr(parent, child_attr, None)
    
    def __deepcopy__(self, *args, **kwargs):
        return Node(deepcopy(self.value), deepcopy(self.left), deepcopy(self.right))

    def _repr_svg_(self) -> str:
        """Display the binary tree using Graphviz (used for `Jupyter notebooks`_).

        .. _Jupyter notebooks: https://jupyter.org
        """
        # noinspection PyProtectedMember
        return self.graphviz()._repr_svg_()

    def graphviz(self, anno:"annotator"=None, *args: Any, **kwargs: Any) -> Digraph:
        """ Return a graphviz.Digraph_ object representing the binary tree.
        
            :param anno: Annotator object to change representation

        This method's positional and keyword arguments are passed directly into the
        the Digraph's **__init__** method.

        :return: graphviz.Digraph_ object representing the binary tree.

        .. code-block:: python

            >>> from binarytree import tree
            >>>
            >>> t = tree()
            >>>
            >>> graph = t.graphviz()    # Generate a graphviz object
            >>> graph.body              # Get the DOT body
            >>> graph.render()          # Render the graph

        .. _graphviz.Digraph: https://graphviz.readthedocs.io/en/stable/api.html#digraph
        """
        if "node_attr" not in kwargs:
            kwargs["node_attr"] = {
                "shape": "record",
                "style": "filled, rounded",
                "color": "lightgray",
                "fillcolor": "lightgray",
                "fontcolor": "black",
            }
        
        anno = annotator() if anno is None else anno

        digraph = Digraph(*args, **kwargs)

        for node in self:
            node_id = str(id(node))

            if anno.node2ncolor(node) != "":
                digraph.node(node_id, nohtml(f"<l>|<v> {node.value}{anno.node2str(node)}|<r>"), fillcolor=anno.node2ncolor(node))
            else:
                digraph.node(node_id, nohtml(f"<l>|<v> {node.value}{anno.node2str(node)}|<r>"))

            if node.left is not None:

                if anno.edge2ecolor(node, node.left) != "":
                    digraph.edge(f"{node_id}:l", f"{id(node.left)}:v", color=f"{anno.edge2ecolor(node, node.left)}")
                else:
                    digraph.edge(f"{node_id}:l", f"{id(node.left)}:v")

            if node.right is not None:
                if anno.edge2ecolor(node, node.right) != "":
                    digraph.edge(f"{node_id}:r", f"{id(node.right)}:v", color=f"{anno.edge2ecolor(node, node.right)}")
                else:
                    digraph.edge(f"{node_id}:r", f"{id(node.right)}:v")

        return digraph

    def pprint(self, index: bool = False, delimiter: str = "-") -> None:
        """Pretty-print the binary tree.

        :param index: If set to True (default: False), display level-order_
            indexes using the format: ``{index}{delimiter}{value}``.
        :type index: bool
        :param delimiter: Delimiter character between the node index and
            the node value (default: '-').
        :type delimiter: str

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)              # index: 0, value: 1
            >>> root.left = Node(2)         # index: 1, value: 2
            >>> root.right = Node(3)        # index: 2, value: 3
            >>> root.left.right = Node(4)   # index: 4, value: 4
            >>>
            >>> root.pprint()
            <BLANKLINE>
              __1
             /   \\
            2     3
             \\
              4
            <BLANKLINE>
            >>> root.pprint(index=True)     # Format: {index}-{value}
            <BLANKLINE>
               _____0-1_
              /         \\
            1-2_        2-3
                \\
                4-4
            <BLANKLINE>

        .. note::
            If you do not need level-order_ indexes in the output string, use
            :func:`binarytree.Node.__str__` instead.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
        """
        lines = _build_tree_string(self, 0, index, delimiter)[0]
        print("\n" + "\n".join((line.rstrip() for line in lines)))

    def validate(self) -> None:
        """Check if the binary tree is malformed.

        :raise binarytree.exceptions.NodeReferenceError: If there is a
            cyclic reference to a node in the binary tree.
        :raise binarytree.exceptions.NodeTypeError: If a node is not an
            instance of :class:`binarytree.Node`.
        :raise binarytree.exceptions.NodeValueError: If a node value is not a
            number (e.g. int, float).

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = root  # Cyclic reference to root
            >>>
            >>> root.validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            NodeReferenceError: cyclic node reference at index 0
        """
        has_more_nodes = True
        visited = set()
        to_visit: List[Optional[Node]] = [self]
        index = 0

        while has_more_nodes:
            has_more_nodes = False
            next_level: List[Optional[Node]] = []

            for node in to_visit:
                if node is None:
                    next_level.append(None)
                    next_level.append(None)
                else:
                    if node in visited:
                        raise NodeReferenceError(
                            f"cyclic reference at Node({node.val}) "
                            + f"(level-order index {index})"
                        )
                    if not isinstance(node, Node):
                        raise NodeTypeError(
                            "invalid node instance at index {}".format(index)
                        )
                    if not isinstance(node.val, NodeValueTup):
                        raise NodeValueError(
                            "invalid node value at index {}".format(index)
                        )
                    if not isinstance(node.value, NodeValueTup):
                        raise NodeValueError(
                            "invalid node value at index {}".format(index)
                        )
                    if node.left is not None or node.right is not None:
                        has_more_nodes = True

                    visited.add(node)
                    next_level.append(node.left)
                    next_level.append(node.right)

                index += 1

            to_visit = next_level

    @property
    def values(self) -> List[Optional[NodeValue]]:
        """Return the `list representation`_ of the binary tree.

        .. _list representation:
            https://en.wikipedia.org/wiki/Binary_tree#Arrays

        :return: List representation of the binary tree, which is a list of
            node values in breadth-first order starting from the root (current
            node). If a node is at index i, its left child is always at 2i + 1,
            right child at 2i + 2, and parent at index floor((i - 1) / 2). None
            indicates absence of a node at that index. See example below for an
            illustration.
        :rtype: [float | int | None]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> root.values
            [1, 2, 3, None, 4]
        """
        current_level: List[Optional[Node]] = [self]
        has_more_nodes = True
        values: List[Optional[NodeValue]] = []

        while has_more_nodes:
            has_more_nodes = False
            next_level: List[Optional[Node]] = []

            for node in current_level:
                if node is None:
                    values.append(None)
                    next_level.append(None)
                    next_level.append(None)
                    continue

                if node.left is not None or node.right is not None:
                    has_more_nodes = True

                values.append(node.val)
                next_level.append(node.left)
                next_level.append(node.right)

            current_level = next_level

        # Get rid of trailing None values
        while values and values[-1] is None:
            values.pop()

        return values

    @property
    def leaves(self) -> List["Node"]:
        """Return the leaf nodes of the binary tree.

        A leaf node is any node that does not have child nodes.

        :return: List of leaf nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> print(root)
            <BLANKLINE>
              __1
             /   \\
            2     3
             \\
              4
            <BLANKLINE>
            >>> root.leaves
            [Node(3), Node(4)]
        """
        current_level = [self]
        leaves = []

        while len(current_level) > 0:
            next_level = []
            for node in current_level:
                if node.left is None and node.right is None:
                    leaves.append(node)
                    continue
                if node.left is not None:
                    next_level.append(node.left)
                if node.right is not None:
                    next_level.append(node.right)
            current_level = next_level
        return leaves

    @property
    def levels(self) -> List[List["Node"]]:
        """Return the nodes in the binary tree level by level.

        :return: Lists of nodes level by level.
        :rtype: [[binarytree.Node]]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> print(root)
            <BLANKLINE>
              __1
             /   \\
            2     3
             \\
              4
            <BLANKLINE>
            >>>
            >>> root.levels
            [[Node(1)], [Node(2), Node(3)], [Node(4)]]
        """
        current_level = [self]
        levels = []

        while len(current_level) > 0:
            next_level = []
            for node in current_level:
                if node.left is not None:
                    next_level.append(node.left)
                if node.right is not None:
                    next_level.append(node.right)
            levels.append(current_level)
            current_level = next_level
        return levels

    @property
    def height(self) -> int:
        """Return the height of the binary tree.

        Height of a binary tree is the number of edges on the longest path
        between the root node and a leaf node. Binary tree with just a single
        node has a height of 0.

        :return: Height of the binary tree.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.left.left = Node(3)
            >>>
            >>> print(root)
            <BLANKLINE>
                1
               /
              2
             /
            3
            <BLANKLINE>
            >>> root.height
            2

        .. note::
            A binary tree with only a root node has a height of 0.
        """
        return _get_tree_properties(self)["height"]

    @property
    def size(self) -> int:
        """Return the total number of nodes in the binary tree.

        :return: Total number of nodes.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> root.size
            4

        .. note::
            This method is equivalent to :func:`binarytree.Node.__len__`.
        """
        return _get_tree_properties(self)["size"]

    @property
    def leaf_count(self) -> int:
        """Return the total number of leaf nodes in the binary tree.

        A leaf node is a node with no child nodes.

        :return: Total number of leaf nodes.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> root.leaf_count
            2
        """
        return _get_tree_properties(self)["leaf_count"]

    @property
    def is_balanced(self) -> bool:
        """Check if the binary tree is height-balanced.

        A binary tree is height-balanced if it meets the following criteria:

        * Left subtree is height-balanced.
        * Right subtree is height-balanced.
        * The difference between heights of left and right subtrees is no more
          than 1.
        * An empty binary tree is always height-balanced.

        :return: True if the binary tree is balanced, False otherwise.
        :rtype: bool

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.left.left = Node(3)
            >>>
            >>> print(root)
            <BLANKLINE>
                1
               /
              2
             /
            3
            <BLANKLINE>
            >>> root.is_balanced
            False
        """
        return _is_balanced(self) >= 0

    @property
    def is_bst(self) -> bool:
        """Check if the binary tree is a BST_ (binary search tree).

        :return: True if the binary tree is a BST_, False otherwise.
        :rtype: bool

        .. _BST: https://en.wikipedia.org/wiki/Binary_search_tree

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(2)
            >>> root.left = Node(1)
            >>> root.right = Node(3)
            >>>
            >>> print(root)
            <BLANKLINE>
              2
             / \\
            1   3
            <BLANKLINE>
            >>> root.is_bst
            True
        """
        return _is_bst(self)

    @property
    def is_symmetric(self) -> bool:
        """Check if the binary tree is symmetric.

        A binary tree is symmetric if it meets the following criteria:

        * Left subtree is a mirror of the right subtree about the root node.

        :return: True if the binary tree is a symmetric, False otherwise.
        :rtype: bool

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(2)
            >>> root.left.left = Node(3)
            >>> root.left.right = Node(4)
            >>> root.right.left = Node(4)
            >>> root.right.right = Node(3)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1__
               /     \\
              2       2
             / \\     / \\
            3   4   4   3
            <BLANKLINE>
            >>> root.is_symmetric
            True
        """
        return _is_symmetric(self)

    @property
    def is_max_heap(self) -> bool:
        """Check if the binary tree is a `max heap`_.

        :return: True if the binary tree is a `max heap`_, False otherwise.
        :rtype: bool

        .. _max heap: https://en.wikipedia.org/wiki/Min-max_heap

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(3)
            >>> root.left = Node(1)
            >>> root.right = Node(2)
            >>>
            >>> print(root)
            <BLANKLINE>
              3
             / \\
            1   2
            <BLANKLINE>
            >>> root.is_max_heap
            True
        """
        return _get_tree_properties(self)["is_max_heap"]

    @property
    def is_min_heap(self) -> bool:
        """Check if the binary tree is a `min heap`_.

        :return: True if the binary tree is a `min heap`_, False otherwise.
        :rtype: bool

        .. _min heap: https://en.wikipedia.org/wiki/Min-max_heap

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>>
            >>> print(root)
            <BLANKLINE>
              1
             / \\
            2   3
            <BLANKLINE>
            >>> root.is_min_heap
            True
        """
        return _get_tree_properties(self)["is_min_heap"]

    @property
    def is_perfect(self) -> bool:
        """Check if the binary tree is perfect.

        A binary tree is perfect if all its levels are completely filled. See
        example below for an illustration.

        :return: True if the binary tree is perfect, False otherwise.
        :rtype: bool

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>> root.right.left = Node(6)
            >>> root.right.right = Node(7)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1__
               /     \\
              2       3
             / \\     / \\
            4   5   6   7
            <BLANKLINE>
            >>> root.is_perfect
            True
        """
        return _get_tree_properties(self)["is_perfect"]

    @property
    def is_strict(self) -> bool:
        """Check if the binary tree is strict.

        A binary tree is strict if all its non-leaf nodes have both the left
        and right child nodes.

        :return: True if the binary tree is strict, False otherwise.
        :rtype: bool

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.is_strict
            True
        """
        return _get_tree_properties(self)["is_strict"]

    @property
    def is_complete(self) -> bool:
        """Check if the binary tree is complete.

        A binary tree is complete if it meets the following criteria:

        * All levels except possibly the last are completely filled.
        * Last level is left-justified.

        :return: True if the binary tree is complete, False otherwise.
        :rtype: bool

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.is_complete
            True
        """
        return _get_tree_properties(self)["is_complete"]

    @property
    def min_node_value(self) -> NodeValue:
        """Return the minimum node value of the binary tree.

        :return: Minimum node value.
        :rtype: float | int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>>
            >>> root.min_node_value
            1
        """
        return _get_tree_properties(self)["min_node_value"]

    @property
    def max_node_value(self) -> NodeValue:
        """Return the maximum node value of the binary tree.

        :return: Maximum node value.
        :rtype: float | int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>>
            >>> root.max_node_value
            3
        """
        return _get_tree_properties(self)["max_node_value"]

    @property
    def max_leaf_depth(self) -> int:
        """Return the maximum leaf node depth of the binary tree.

        :return: Maximum leaf node depth.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.right.left = Node(4)
            >>> root.right.left.left = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
              1____
             /     \\
            2       3
                   /
                  4
                 /
                5
            <BLANKLINE>
            >>> root.max_leaf_depth
            3
        """
        return _get_tree_properties(self)["max_leaf_depth"]

    @property
    def min_leaf_depth(self) -> int:
        """Return the minimum leaf node depth of the binary tree.

        :return: Minimum leaf node depth.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.right.left = Node(4)
            >>> root.right.left.left = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
              1____
             /     \\
            2       3
                   /
                  4
                 /
                5
            <BLANKLINE>
            >>> root.min_leaf_depth
            1
        """
        return _get_tree_properties(self)["min_leaf_depth"]

    @property
    def properties(self) -> Dict[str, Any]:
        """Return various properties of the binary tree.

        :return: Binary tree properties.
        :rtype: dict

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>> props = root.properties
            >>>
            >>> props['height']         # equivalent to root.height
            2
            >>> props['size']           # equivalent to root.size
            5
            >>> props['max_leaf_depth'] # equivalent to root.max_leaf_depth
            2
            >>> props['min_leaf_depth'] # equivalent to root.min_leaf_depth
            1
            >>> props['max_node_value'] # equivalent to root.max_node_value
            5
            >>> props['min_node_value'] # equivalent to root.min_node_value
            1
            >>> props['leaf_count']     # equivalent to root.leaf_count
            3
            >>> props['is_balanced']    # equivalent to root.is_balanced
            True
            >>> props['is_bst']         # equivalent to root.is_bst
            False
            >>> props['is_complete']    # equivalent to root.is_complete
            True
            >>> props['is_symmetric']   # equivalent to root.is_symmetric
            False
            >>> props['is_max_heap']    # equivalent to root.is_max_heap
            False
            >>> props['is_min_heap']    # equivalent to root.is_min_heap
            True
            >>> props['is_perfect']     # equivalent to root.is_perfect
            False
            >>> props['is_strict']      # equivalent to root.is_strict
            True
        """
        properties = _get_tree_properties(self)
        properties.update(
            {
                "is_bst": _is_bst(self),
                "is_balanced": _is_balanced(self) >= 0,
                "is_symmetric": _is_symmetric(self),
            }
        )
        return properties

    @property
    def inorder(self) -> List["Node"]:
        """Return the nodes in the binary tree using in-order_ traversal.

        An in-order_ traversal visits left subtree, root, then right subtree.

        .. _in-order: https://en.wikipedia.org/wiki/Tree_traversal

        :return: List of nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.inorder
            [Node(4), Node(2), Node(5), Node(1), Node(3)]
        """
        result: List[Node] = []
        stack: List[Node] = []
        node: Optional[Node] = self

        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            if stack:
                node = stack.pop()
                result.append(node)
                node = node.right

        return result

    @property
    def preorder(self) -> List["Node"]:
        """Return the nodes in the binary tree using pre-order_ traversal.

        A pre-order_ traversal visits root, left subtree, then right subtree.

        .. _pre-order: https://en.wikipedia.org/wiki/Tree_traversal

        :return: List of nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.preorder
            [Node(1), Node(2), Node(4), Node(5), Node(3)]
        """
        result: List[Node] = []
        stack: List[Optional[Node]] = [self]

        while stack:
            node = stack.pop()
            if node:
                result.append(node)
                stack.append(node.right)
                stack.append(node.left)

        return result

    @property
    def postorder(self) -> List["Node"]:
        """Return the nodes in the binary tree using post-order_ traversal.

        A post-order_ traversal visits left subtree, right subtree, then root.

        .. _post-order: https://en.wikipedia.org/wiki/Tree_traversal

        :return: List of nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.postorder
            [Node(4), Node(5), Node(2), Node(3), Node(1)]
        """
        result: List[Node] = []
        stack: List[Optional[Node]] = [self]

        while stack:
            node = stack.pop()
            if node:
                result.append(node)
                stack.append(node.left)
                stack.append(node.right)

        return result[::-1]

    @property
    def levelorder(self) -> List["Node"]:
        """Return the nodes in the binary tree using level-order_ traversal.

        A level-order_ traversal visits nodes left to right, level by level.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :return: List of nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.levelorder
            [Node(1), Node(2), Node(3), Node(4), Node(5)]
        """
        current_level = [self]
        result = []

        while len(current_level) > 0:
            next_level = []
            for node in current_level:
                result.append(node)
                if node.left is not None:
                    next_level.append(node.left)
                if node.right is not None:
                    next_level.append(node.right)
            current_level = next_level

        return result


def _is_balanced(root: Optional[Node]) -> int:
    """Return the tree height + 1 if balanced, -1 otherwise.

    :param root: Root node of the binary tree.
    :type root: binarytree.Node
    :return: Height if the binary tree is balanced, -1 otherwise.
    :rtype: int
    """
    if root is None:
        return 0
    left = _is_balanced(root.left)
    if left < 0:
        return -1
    right = _is_balanced(root.right)
    if right < 0:
        return -1
    return -1 if abs(left - right) > 1 else max(left, right) + 1


def _is_bst(root: Optional[Node]) -> bool:
    """Check if the binary tree is a BST (binary search tree).

    :param root: Root node of the binary tree.
    :type root: binarytree.Node
    :return: True if the binary tree is a BST, False otherwise.
    :rtype: bool
    """
    stack: List[Node] = []
    cur = root
    pre = None

    while stack or cur is not None:
        if cur is not None:
            stack.append(cur)
            cur = cur.left
        else:
            node = stack.pop()
            if pre is not None and node.val <= pre.val:
                return False
            pre = node
            cur = node.right
    return True


def _is_symmetric(root: Optional[Node]) -> bool:
    """Check if the binary tree is symmetric.

    :param root: Root node of the binary tree.
    :type root: binarytree.Node
    :return: True if the binary tree is symmetric, False otherwise.
    :rtype: bool
    """

    def symmetric_helper(left_subtree, right_subtree):
        if left_subtree is None and right_subtree is None:
            return True
        if left_subtree is None or right_subtree is None:
            return False
        return (
            left_subtree.val == right_subtree.val
            and symmetric_helper(left_subtree.left, right_subtree.right)
            and symmetric_helper(left_subtree.right, right_subtree.left)
        )

    return symmetric_helper(root, root)


def _validate_tree_height(height: int):
    """Check if the height of the binary tree is valid.

    :param height: Height of the binary tree (must be 0 - 9 inclusive).
    :type height: int
    :raise binarytree.exceptions.TreeHeightError: If height is invalid.
    """
    if not (type(height) == int and 0 <= height <= 9):
        raise TreeHeightError("height must be an int between 0 - 9")


def _generate_perfect_bst(height: int) -> Optional[Node]:
    """Generate a perfect BST (binary search tree) and return its root.

    :param height: Height of the BST.
    :type height: int
    :return: Root node of the BST.
    :rtype: binarytree.Node | None
    """
    max_node_count = 2 ** (height + 1) - 1
    node_values = list(range(max_node_count))
    return _build_bst_from_sorted_values(node_values)


def _build_bst_from_sorted_values(sorted_values: List[int]) -> Optional[Node]:
    """Recursively build a perfect BST from odd number of sorted values.

    :param sorted_values: Odd number of sorted values.
    :type sorted_values: [int | float]
    :return: Root node of the BST.
    :rtype: binarytree.Node | None
    """
    if len(sorted_values) == 0:
        return None
    mid_index = len(sorted_values) // 2
    root = Node(sorted_values[mid_index])
    root.left = _build_bst_from_sorted_values(sorted_values[:mid_index])
    root.right = _build_bst_from_sorted_values(sorted_values[mid_index + 1 :])
    return root


def _generate_random_leaf_count(height: int) -> int:
    """Return a random leaf count for building binary trees.

    :param height: Height of the binary tree.
    :type height: int
    :return: Random leaf count.
    :rtype: int
    """
    max_leaf_count = 2 ** height
    half_leaf_count = max_leaf_count // 2

    # A very naive way of mimicking normal distribution
    roll_1 = random.randint(0, half_leaf_count)
    roll_2 = random.randint(0, max_leaf_count - half_leaf_count)
    return roll_1 + roll_2 or half_leaf_count


def _generate_random_node_values(height: int) -> List[int]:
    """Return random node values for building binary trees.

    :param height: Height of the binary tree.
    :type height: int
    :return: Randomly generated node values.
    :rtype: [int]
    """
    max_node_count = 2 ** (height + 1) - 1
    node_values = list(range(max_node_count))
    random.shuffle(node_values)
    return node_values


def _build_tree_string(
    root: Optional[Node], curr_index: int, index: bool = False, delimiter: str = "-"
) -> Tuple[List[str], int, int, int]:
    """Recursively walk down the binary tree and build a pretty-print string.

    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.

    :param root: Root node of the binary tree.
    :type root: binarytree.Node
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)

    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.val)
    else:
        node_repr = str(root.val)

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = _build_tree_string(
        root.left, 2 * curr_index + 1, index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = _build_tree_string(
        root.right, 2 * curr_index + 2, index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


def _get_tree_properties(root: Node) -> Dict[str, Any]:
    """Inspect the binary tree and return its properties (e.g. height).

    :param root: Root node of the binary tree.
    :type root: binarytree.Node
    :return: Binary tree properties.
    :rtype: dict
    """
    is_descending = True
    is_ascending = True
    min_node_value = root.val
    max_node_value = root.val
    size = 0
    leaf_count = 0
    min_leaf_depth = 0
    max_leaf_depth = -1
    is_strict = True
    is_complete = True
    current_level = [root]
    non_full_node_seen = False

    while len(current_level) > 0:
        max_leaf_depth += 1
        next_level = []

        for node in current_level:
            size += 1
            val = node.val
            min_node_value = min(val, min_node_value)
            max_node_value = max(val, max_node_value)

            # Node is a leaf.
            if node.left is None and node.right is None:
                if min_leaf_depth == 0:
                    min_leaf_depth = max_leaf_depth
                leaf_count += 1

            if node.left is not None:
                if node.left.val > val:
                    is_descending = False
                elif node.left.val < val:
                    is_ascending = False
                next_level.append(node.left)
                is_complete = not non_full_node_seen
            else:
                non_full_node_seen = True

            if node.right is not None:
                if node.right.val > val:
                    is_descending = False
                elif node.right.val < val:
                    is_ascending = False
                next_level.append(node.right)
                is_complete = not non_full_node_seen
            else:
                non_full_node_seen = True

            # If we see a node with only one child, it is not strict
            is_strict &= (node.left is None) == (node.right is None)

        current_level = next_level

    return {
        "height": max_leaf_depth,
        "size": size,
        "is_max_heap": is_complete and is_descending,
        "is_min_heap": is_complete and is_ascending,
        "is_perfect": leaf_count == 2 ** max_leaf_depth,
        "is_strict": is_strict,
        "is_complete": is_complete,
        "leaf_count": leaf_count,
        "min_node_value": min_node_value,
        "max_node_value": max_node_value,
        "min_leaf_depth": min_leaf_depth,
        "max_leaf_depth": max_leaf_depth,
    }


def get_parent(root: Node, child: Node) -> Optional[Node]:
    """Search the binary tree and return the parent of given child.

    :param root: Root node of the binary tree.
    :type: binarytree.Node
    :param child: Child node.
    :rtype: binarytree.Node
    :return: Parent node, or None if missing.
    :rtype: binarytree.Node | None

    **Example**:

    .. doctest::

        >>> from binarytree import Node, get_parent
        >>>
        >>> root = Node(1)
        >>> root.left = Node(2)
        >>> root.right = Node(3)
        >>> root.left.right = Node(4)
        >>>
        >>> print(root)
        <BLANKLINE>
          __1
         /   \\
        2     3
         \\
          4
        <BLANKLINE>
        >>> print(get_parent(root, root.left.right))
        <BLANKLINE>
        2
         \\
          4
        <BLANKLINE>
    """
    if child is None:
        return None

    stack: List[Optional[Node]] = [root]

    while stack:
        node = stack.pop()
        if node:
            if node.left is child or node.right is child:
                return node
            else:
                stack.append(node.left)
                stack.append(node.right)
    return None


def build(values: List) -> Optional[Node]:
    """Build a tree from `list representation`_ and return its root node.

    .. _list representation:
        https://en.wikipedia.org/wiki/Binary_tree#Arrays

    :param values: List representation of the binary tree, which is a list of
        node values in breadth-first order starting from the root (current
        node). If a node is at index i, its left child is always at 2i + 1,
        right child at 2i + 2, and parent at floor((i - 1) / 2). None indicates
        absence of a node at that index. See example below for an illustration.
    :type values: [int | float | None]
    :return: Root node of the binary tree.
    :rtype: binarytree.Node | None
    :raise binarytree.exceptions.NodeNotFoundError: If the list representation
        is malformed (e.g. a parent node is missing).

    **Example**:

    .. doctest::

        >>> from binarytree import build
        >>>
        >>> root = build([1, 2, 3, None, 4])
        >>>
        >>> print(root)
        <BLANKLINE>
          __1
         /   \\
        2     3
         \\
          4
        <BLANKLINE>

    .. doctest::

        >>> from binarytree import build
        >>>
        >>> root = build([None, 2, 3])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
         ...
        NodeNotFoundError: parent node missing at index 0
    """
    nodes = [None if v is None else Node(v) for v in values]

    for index in range(1, len(nodes)):
        node = nodes[index]
        if node is not None:
            parent_index = (index - 1) // 2
            parent = nodes[parent_index]
            if parent is None:
                raise NodeNotFoundError(
                    "parent node missing at index {}".format(parent_index)
                )
            setattr(parent, LEFT_FIELD if index % 2 else RIGHT_FIELD, node)

    return nodes[0] if nodes else None


def tree(height: int = 3, is_perfect: bool = False) -> Optional[Node]:
    """Generate a random binary tree and return its root node.

    :param height: Height of the tree (default: 3, range: 0 - 9 inclusive).
    :type height: int
    :param is_perfect: If set to True (default: False), a perfect binary tree
        with all levels filled is returned. If set to False, a perfect binary
        tree may still be generated by chance.
    :type is_perfect: bool
    :return: Root node of the binary tree.
    :rtype: binarytree.Node
    :raise binarytree.exceptions.TreeHeightError: If height is invalid.

    **Example**:

    .. doctest::

        >>> from binarytree import tree
        >>>
        >>> root = tree()
        >>>
        >>> root.height
        3

    .. doctest::

        >>> from binarytree import tree
        >>>
        >>> root = tree(height=5, is_perfect=True)
        >>>
        >>> root.height
        5
        >>> root.is_perfect
        True

    .. doctest::

        >>> from binarytree import tree
        >>>
        >>> root = tree(height=20)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
         ...
        TreeHeightError: height must be an int between 0 - 9
    """
    if (height == -1):
        return None
    
    _validate_tree_height(height)
    values = _generate_random_node_values(height)
    if is_perfect:
        return build(values)

    leaf_count = _generate_random_leaf_count(height)
    root_node = Node(values.pop(0))
    leaves = set()

    for value in values:
        node = root_node
        depth = 0
        inserted = False

        while depth < height and not inserted:
            attr = random.choice((LEFT_FIELD, RIGHT_FIELD))
            if getattr(node, attr) is None:
                setattr(node, attr, Node(value))
                inserted = True
            node = getattr(node, attr)
            depth += 1

        if inserted and depth == height:
            leaves.add(node)
        if len(leaves) == leaf_count:
            break

    return root_node


def bst(height: int = 3, is_perfect: bool = False) -> Optional[Node]:
    """Generate a random BST (binary search tree) and return its root node.

    :param height: Height of the BST (default: 3, range: 0 - 9 inclusive).
    :type height: int
    :param is_perfect: If set to True (default: False), a perfect BST with all
        levels filled is returned. If set to False, a perfect BST may still be
        generated by chance.
    :type is_perfect: bool
    :return: Root node of the BST.
    :rtype: binarytree.Node
    :raise binarytree.exceptions.TreeHeightError: If height is invalid.

    **Example**:

    .. doctest::

        >>> from binarytree import bst
        >>>
        >>> root = bst()
        >>>
        >>> root.height
        3
        >>> root.is_bst
        True

    .. doctest::

        >>> from binarytree import bst
        >>>
        >>> root = bst(10)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
         ...
        TreeHeightError: height must be an int between 0 - 9
    """
    _validate_tree_height(height)
    if is_perfect:
        return _generate_perfect_bst(height)

    values = _generate_random_node_values(height)
    leaf_count = _generate_random_leaf_count(height)

    root_node = Node(values.pop(0))
    leaves = set()

    for value in values:
        node = root_node
        depth = 0
        inserted = False

        while depth < height and not inserted:
            attr = LEFT_FIELD if node.val > value else RIGHT_FIELD
            if getattr(node, attr) is None:
                setattr(node, attr, Node(value))
                inserted = True
            node = getattr(node, attr)
            depth += 1

        if inserted and depth == height:
            leaves.add(node)
        if len(leaves) == leaf_count:
            break

    return root_node


def heap(
    height: int = 3, is_max: bool = True, is_perfect: bool = False
) -> Optional[Node]:
    """Generate a random heap and return its root node.

    :param height: Height of the heap (default: 3, range: 0 - 9 inclusive).
    :type height: int
    :param is_max: If set to True (default: True), generate a max heap. If set
        to False, generate a min heap. A binary tree with only the root node is
        considered both a min and max heap.
    :type is_max: bool
    :param is_perfect: If set to True (default: False), a perfect heap with all
        levels filled is returned. If set to False, a perfect heap may still be
        generated by chance.
    :type is_perfect: bool
    :return: Root node of the heap.
    :rtype: binarytree.Node
    :raise binarytree.exceptions.TreeHeightError: If height is invalid.

    **Example**:

    .. doctest::

        >>> from binarytree import heap
        >>>
        >>> root = heap()
        >>>
        >>> root.height
        3
        >>> root.is_max_heap
        True

    .. doctest::

        >>> from binarytree import heap
        >>>
        >>> root = heap(4, is_max=False)
        >>>
        >>> root.height
        4
        >>> root.is_min_heap
        True

    .. doctest::

        >>> from binarytree import heap
        >>>
        >>> root = heap(5, is_max=False, is_perfect=True)
        >>>
        >>> root.height
        5
        >>> root.is_min_heap
        True
        >>> root.is_perfect
        True

    .. doctest::

        >>> from binarytree import heap
        >>>
        >>> root = heap(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
         ...
        TreeHeightError: height must be an int between 0 - 9
    """
    _validate_tree_height(height)
    values = _generate_random_node_values(height)

    if not is_perfect:
        # Randomly cut some of the leaf nodes away
        random_cut = random.randint(2 ** height, len(values))
        values = values[:random_cut]

    if is_max:
        negated = [-v for v in values]
        heapq.heapify(negated)
        return build([-v for v in negated])
    else:
        heapq.heapify(values)
        return build(values)

def randomize_tree_val(bintree:Node):
    """ Replaces all values in the tree with random int values in the
        range of [-1000, 1000]
    """
    
    n_nodes = _get_tree_properties(bintree)["size"]
    new_vals = random.choices(list(range(-1000,1000)), k=n_nodes)
    for n,v in zip(bintree, new_vals):
        n.value = v    
    
class annotator:
    def __init__(self, str_dict:dict=None, n_dict:dict=None, edge_dict:dict = None):
        """ Annotator class the color and add information to a binary tree
        
            :param str_dict: Dict with key: node values; value: str with the additional info added to the node
            :type str_dict: dict
            :param n_dict: Dict with key: node values; value: str with the desired color
            :type n_dict: dict
            :param edge_dict: Dict with key: tuple of node values, (src.value, dst.value); value: str with the desired color of the edge
            :type edge_dict: dict        
        """
        self.str_dict = str_dict if str_dict is not None else {}
        self.n_dict = n_dict if n_dict is not None else {}
        self.edge_dict = edge_dict if edge_dict is not None else {}


    def node2str(self, node:"Node"):
        """ Get the str for the given node, empty if not set
            
            :param node: The node
        """
        try:
            return self.str_dict[node.value]
        except KeyError:
            return ""

    def node2ncolor(self, node:"Node"):
        """ Get the color str for the given node, empty if not set
            
            :param node: The node
        """
        try:
            return self.n_dict[node.value]
        except KeyError:
            return ""

    def edge2ecolor(self, src:"Node", dst:"Node"):
        """ Get the color str for the edge from src to dst, empty if not set
            
            :param node: The src and dst node
        """
        try:
            return self.edge_dict[(src.value, dst.value)]
        except KeyError:
            return ""

    def color_path(self, nodes:List["Node"], color="red"):
        """ Color a path in the bt
       
            :param nodes: List of nodes on the path. nodes[i+1] must be a child of nodes[i]
            :type nodes: List[Node]
            :param color: The color
        """
        for src, dst in zip(nodes[:-1], nodes[1:]):
            self.n_dict[src.value] = color
            self.edge_dict[(src.value, dst.value)] = color
        self.n_dict[nodes[-1].value] = color

    def uncolor_path(self, nodes:List["Node"]):
        """ Uncolor a path in the bt
       
            :param nodes: List of nodes on the path. nodes[i+1] must be a child of nodes[i]
            :type nodes: List[Node]
        """
        return self.color_path(nodes, "")

    def color_node(self, node:"Node", color="red"):
        """ Color a single node
       
            :param node: The node
            :param color: The color
        """
        if node is None:
            return
        self.n_dict[node.value] = color

    def uncolor_node(self, node:"Node"):
        """ Uncolor a single node
       
            :param node: The node
        """
        return self.color_node(node, "")

    def annotate_node(self, node:"Node", txt:str):
        """ Annotate a single node
       
            :param node: The node
            :param txt: Annotation text
        """
        if node is None:
            return
        self.str_dict[node.value] = txt

    def unannotate_node(self, node:"Node"):
        """ Erase annotation of a single node
       
            :param node: The node
        """
        return self.annotate_node(node, "")

    def verify(self, bt:"Node"):
        """ Verifies that all annotated nodes and edges exist in the bt or the corresponding value is empty str
            
            :param bt: Binary tree to consistent with annotation
        """
        st_set = set()
        e_set = set()

        for n in bt:
            if n.value in st_set:
                print("Value has double occur")
                return False
            st_set.add(n.value)
            for nprime in [n.left, n.right]:
                if nprime is not None:
                    e_set.add((n.value, nprime.value))

        if not all(map(lambda v: v in st_set or self.str_dict[v] == "", self.str_dict.keys())):
            print("States annotated but not existing for string annotation")
            return False
        if not all(map(lambda v: v in st_set or self.n_dict[v] == "", self.n_dict.keys())):
            print("States annotated but not existing for color annotation")
            return False
        if not all(map(lambda v: v in st_set or self.edge_dict[v] == "", self.edge_dict.keys())):
            print("Edges annotated but not existing for color annotation")
            return False

        return True

def dict2prefixtree(enc_dict:dict):
    """ Create a binary tree that corresponds to the prefix encoding given as dict
        Note: In order to be optimal, the tree should be locally complement, we do not enforce this here
        
        :param enc_dict: The dictionnary (char/float/int):"occurence string"
        :type enc_dict: dict
    """
    node_dict = {}
    for (v,c) in enc_dict.items():
        if c in node_dict.keys():
            raise RuntimeError(f"code {c} used twice")
        n = Node(v)
        found = False
        while not found:
            d = c[-1]
            c = c[:-1]
            nprime = node_dict.get(c)
            found = True
            if nprime is None:
                nprime = Node("")
                node_dict[c] = nprime
                found = False or (c == "")
            
            if d == '0':
                nprime.left = n
            else:
                nprime.right = n
            n = nprime
    return node_dict[""]

def is_prefixtree(root:Node):
    """ Checks if the given tree is a prefix tree: That is locally complete and only leaves have non-empty values
    """
    if root is None:
        return True
    
    if not ((root.left is None and root.right is None)
           or (root.left is not None and root.right is not None)):
        return False
    if root.right is None and root.value == "":
        return False
    if root.right is not None and root.value != "":
        return False
    return is_prefixtree(root.left) and is_prefixtree(root.right)

def prefix_decode(prefixtree:Node, code:str):
    """ Uses a the given prefixtree to decode a string
    """
    
    def is_leaf(n:Node):
        return (n.left is None) and (n.right is None)
    
    res = ""
    
    i = 0
    lc = len(code)
    while i != lc:
        n = prefixtree
        while not is_leaf(n):
            if code[i] == "0":
                n = n.left
            else:
                n = n.right
            i += 1
        res += n.value
    return res
            
def generate_random_encoding_dict():
    root = tree(random.randint(1,5))
    enc_dict = dict()
    
    def is_leaf(n:Node):
        return (n.left is None) and (n.right is None)
    
    def rec_(cn:Node, occ:str):
        if cn is None:
            return
        if is_leaf(cn):
            enc_dict[chr(cn.value+33)] = occ
            return
        rec_(cn.left, occ+"0")
        rec_(cn.right, occ+"1")
    
    rec_(root, "")
    return enc_dict
    
def general_recursion(n:Node, f_pre:"fun", f_in:"fun", f_post:"fun", f_ret:"fun", *args):
    if n is None:
        return f_ret(n, *args)
    
    s_pre = f_pre(n, *args)
    ret_left = general_recursion(n.left, f_pre, f_in, f_post, f_ret, *args, *s_pre)
    s_in = f_in(n, *args, *s_pre, *ret_left)
    ret_right = general_recursion(n.right, f_pre, f_in, f_post, f_ret, *args, *s_pre, *ret_left, *s_in)
    s_post = f_post(n, *args, *s_pre, *ret_left, *s_in, *ret_right)
    
    return f_ret(n, *args, *s_pre, *ret_left, *s_in, *ret_right, *s_post)

def bst2list(bst:Node):
    L = []
    def app(n:Node):
        L.append(n.value)
        return []
    f_pre = f_post = f_ret = lambda *args: []
    f_in = lambda n, *args: app(n)
    
    general_recursion(bst, f_pre, f_in, f_post, f_ret)
    return L

def get_complete_tree(ndes:int):
    """ Generates a complete tree of the specified size
    """
    assert ndes > 0
    from collections import deque
    
    treesize = 1
    root = Node(treesize)
    
    todo = deque()
    todo.append((0, root))
    todo.append((1, root))
    while treesize < ndes:
        side, pnode = todo.popleft()
        treesize += 1
        if side:
            nnode = pnode.right = Node(treesize)
        else:
            nnode = pnode.left = Node(treesize)
        todo.append((0, nnode))
        todo.append((1, nnode))
    return root
        

def fillbstfromsorted(bst:Node, L:List["value"]):
    Lp = list(reversed(deepcopy(L)))
    def take(n:Node):
        n.value = Lp.pop()
        return []
    
    f_pre = f_post = f_ret = lambda *args: []
    f_in = lambda n, *args: take(n)
    
    general_recursion(bst, f_pre, f_in, f_post, f_ret)
    
def get_bst(height):
    """ Generate a random binary search tree with integers as labels
    """
    
    bst = tree(height)
    L = bst.size*[0]
    for i in range(1, len(L)):
        L[i] = L[i-1] + random.randint(1,20)
    m = int(sum(L)/len(L))
    for i in range(len(L)):
        L[i] -= m
    fillbstfromsorted(bst, L)
    return bst

def bst_search(bst:Node, x:"value"):
    """ Searches for a node with value x.
        If it is found, a pair [True, node] is returned with node.value == x
        If it is not found, a pair [False, node] is returned, with node being the last leaf checked
    """
    if bst is None:
        return [False, None]
    nlast = bst
    n = bst
    while True:
        nlast = n
        if n.value == x:
            return [True, n]
        n = n.left if x < n.value else n.right
        if n is None:
            return [False, nlast]

def bst_insert_leaf(bst:Node, x:"value"):
    """ Inserts a new node with value x into the bst as a leaf.
        Throws an exception if x already exists.
    """
    
    f,p = bst_search(bst, x)
    if f:
        raise RuntimeError(f"Label {x} already exists in the tree")
    if x < p.value:
        p.left = Node(x)
    else:
        p.right = Node(x)
    return bst

def bst_insert_root(bst:Node, x:"value"):
    """ Inserts a new node with value x into the bst as root.
        Throws an exception if x already exists.
        Returns the new root
    """
    
    class rel_place:
        """ class describing a place in a binary tree
            with respect to its parent
        """
        def __init__(self, n, s):
            self.n = n
            self.s = s
        
        def get(self):
            if self.s:
                return self.n.right
            else:
                return self.n.left
        def assign(self, other):
            if self.s:
                self.n.right = other
            else:
                self.n.left = other
        
    def insert_impl_(b,l,r):
        if b is None:
            return
        bv = b.value
        if x < bv:
            bl = b.left
            b.left = None
            r.assign(b)
            return insert_impl_(bl, l, rel_place(r.get(),0))
        elif x > bv:
            br = b.right
            b.right = None
            l.assign(b)
            return insert_impl_(br, rel_place(l.get(), 1), r)
        else:
            raise RuntimeError(f"Label {x} already exists in the tree")
    root = Node(x)
    insert_impl_(bst, proxy(root, 0), proxy(root, 1))
    return root

def rand_node(node:Node):
    """ Randomly chose one of the nodes in the tree
    """
    l = [n for n in node]
    return random.choice(l)

def bst_labels_in_range(bst:Node, a:"value", b:"value"):
    """ Return the number of labels in the bst that lie in the interval
        [a, b[
    """
    
    def rec_impl_(bst):
        if bst is None:
            return 0
        
        n_elems = (a <= bst.value and bst.value < b)
        
        if a <= bst.value:
            n_elems += rec_impl_(bst.left)
        if bst.value < b:
            n_elems += rec_impl_(bst.right)
        return n_elems
        
    return rec_impl_(bst)

def bst_delete_node(bst:Node, x:"val"):
    """ Delete a node with a given label value from the bst.
        Returns the new root and whether a node was acutally deleted 
        or not. In other words, if there existed a node with label
        x before the deletetion in the bst
    """
    if bst is None:
        return [False, None] #Nothing deleted, empty tree returned
    
    # Otherwise we have to search for the node while keeping track of its 
    # parent. Then in the second step we can actually delete it
    
    # Step 0 Search the node to be deleted while tracking the parent
    p = None #parent
    d = None # direction
    n = bst #node
    while n is not None:
        if n.value == x:
            break
        p = n
        d = 0 if n.value < x else 1
        n = n.right if d else n.left
    
    if n is None:
        return [False, bst]
    
    # Step 1: Actual deletion, here 3 cases have to be taken into account
    # The node being a leaf, has one child or two children
    new_root = -1
    if n.left is not None and n.right is not None:
        # Two children
        # In this case: search the inorder predescessor and swap them
        inopp = n #parent if inorder pred
        inop = n.left
        while inop.right is not None:
            inopp = inop
            inop = inop.right
        # Exchange values
        bst.value = inop.value
        # Cut it from the tree
        inopp.right = inop.left
        new_root = bst
    elif n.left is not None or n.right is not None:
        # Only one child
        # We can cut the node from the tree by replacing it with
        # its only in child except if it is the root
        if p is None:
            new_root = n.left if n.left is not None else n.right
        else:
            new_root = bst
            if d:
                p.right = n.left if n.left is not None else n.right
            else:
                p.left = n.left if n.left is not None else n.right
    elif n.left is None and n.right is None:
        # Leaf
        # We can simply delete it
        # this means setting the place in the parent to None
        # or create a new node
        if p is None:
            # root
            new_root = None
        else:
            if d:
                p.right = None
            else:
                p.left = None
            new_root = bst
    else:
        raise RuntimeError("Unexpected case!")
    assert new_root != -1
    return [True, new_root]
    

##################################################
################ AVL tree section ################
##################################################

class AVLNode(Node):
    """ Implementation of a node representing a avl tree
        This is a usual node of a bst and in addition it also keeps track 
        of its balance factor (bf)
    """
    def __init__(self, x:"val", left:Optional[Node], right:Optional[Node], bf:Optional[int]):
        Node.__init__(self, x, left, right)
        self.bf = bf
    
    def graphviz(anno:annotator=None):
        from copy import deepcopy
        if anno is None:
            anno = {}
        
        # Use the annotator to add the balance factor
        anno2 = deepcopy(anno)
        for n in self:
            anno2.annotate_node(node, f" : {n.bf} - {anno.node2str(n)}")
        return Node.graphviz(self, anno2)
        
        
    

    