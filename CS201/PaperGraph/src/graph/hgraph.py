"""Heterogeneous graph data structure.

Self-implemented using Python dicts and lists. No external graph libraries.
Supports multiple node types and edge types with type-indexed adjacency lists.
"""

from collections import defaultdict
from .node import Node
from .edge import Edge


class HeterogeneousGraph:
    def __init__(self):
        self._nodes = {}                # node_id -> Node
        self._adjacency = defaultdict(list)     # node_id -> list[Edge] (outgoing)
        self._adjacency_rev = defaultdict(list) # node_id -> list[Edge] (incoming)
        self._node_type_index = defaultdict(set) # node_type -> set[node_id]
        self._edge_type_index = defaultdict(list) # edge_type -> list[Edge]

    def add_node(self, node_id, node_type, properties=None):
        """Add a node to the graph. Overwrites if node_id already exists."""
        node = Node(node_id, node_type, properties)
        old = self._nodes.get(node_id)
        if old:
            self._node_type_index[old.type].discard(node_id)
        self._nodes[node_id] = node
        self._node_type_index[node_type].add(node_id)

    def add_edge(self, source, target, edge_type, weight=1.0, properties=None):
        """Add a directed edge. Both nodes must already exist."""
        if source not in self._nodes or target not in self._nodes:
            raise ValueError(f"Node not found: {source} or {target}")
        edge = Edge(source, target, edge_type, weight, properties)
        self._adjacency[source].append(edge)
        self._adjacency_rev[target].append(edge)
        self._edge_type_index[edge_type].append(edge)

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type):
        """Return list of node_ids for given type."""
        return list(self._node_type_index.get(node_type, set()))

    def get_neighbors(self, node_id, edge_type=None):
        """Return outgoing edges from node_id, optionally filtered by edge_type."""
        edges = self._adjacency.get(node_id, [])
        if edge_type:
            return [e for e in edges if e.type == edge_type]
        return edges

    def get_in_neighbors(self, node_id, edge_type=None):
        """Return incoming edges to node_id, optionally filtered by edge_type."""
        edges = self._adjacency_rev.get(node_id, [])
        if edge_type:
            return [e for e in edges if e.type == edge_type]
        return edges

    def get_edges_by_type(self, edge_type):
        """Return all edges of given type."""
        return self._edge_type_index.get(edge_type, [])

    def num_nodes(self, node_type=None):
        if node_type:
            return len(self._node_type_index.get(node_type, set()))
        return len(self._nodes)

    def num_edges(self, edge_type=None):
        if edge_type:
            return len(self._edge_type_index.get(edge_type, []))
        return sum(len(v) for v in self._adjacency.values())

    def all_node_ids(self):
        return list(self._nodes.keys())

    def all_edge_types(self):
        return list(self._edge_type_index.keys())

    def all_node_types(self):
        return list(self._node_type_index.keys())

    def has_node(self, node_id):
        return node_id in self._nodes

    def has_edge(self, source, target, edge_type):
        for e in self._adjacency.get(source, []):
            if e.target == target and e.type == edge_type:
                return True
        return False

    def summary(self):
        """Return a summary string of graph statistics."""
        lines = ["=== HeterogeneousGraph Summary ==="]
        lines.append(f"Total nodes: {self.num_nodes()}")
        for nt in sorted(self.all_node_types()):
            lines.append(f"  {nt}: {self.num_nodes(nt)} nodes")
        lines.append(f"Total edges: {self.num_edges()}")
        for et in sorted(self.all_edge_types()):
            lines.append(f"  {et}: {self.num_edges(et)} edges")
        return "\n".join(lines)