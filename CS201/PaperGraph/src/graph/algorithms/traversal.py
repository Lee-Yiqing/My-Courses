"""Type-filtered BFS and DFS traversal on heterogeneous graph.

Course connection: BFS (lecture W09-12, section 3.1) and DFS (section 3.2).
Extension: only traverse edges of specified types and visit nodes of specified types.
"""

from collections import deque

from ..hgraph import HeterogeneousGraph


def type_filtered_bfs(graph, start_id, edge_types=None, node_types=None, max_depth=None):
    """BFS that only traverses edges of specified types and visits nodes of specified types.

    Args:
        graph: HeterogeneousGraph instance
        start_id: starting node ID
        edge_types: set of edge types to traverse (None = all)
        node_types: set of node types to visit (None = all)
        max_depth: optional depth limit (in edge hops)

    Returns:
        dict {node_id: distance_from_start} for all reachable nodes
    """
    if not graph.has_node(start_id):
        return {}

    visited = set()
    queue = deque([(start_id, 0)])
    distances = {}

    while queue:
        node_id, dist = queue.popleft()
        if node_id in visited:
            continue

        # Check node type filter
        node = graph.get_node(node_id)
        if node_types and node.type not in node_types:
            continue

        visited.add(node_id)
        distances[node_id] = dist

        if max_depth is not None and dist >= max_depth:
            continue

        for edge in graph.get_neighbors(node_id):
            if edge_types and edge.type not in edge_types:
                continue
            neighbor = edge.target
            if neighbor not in visited:
                queue.append((neighbor, dist + 1))

    return distances


def type_filtered_dfs(graph, start_id, edge_types=None, node_types=None, max_depth=None):
    """DFS that only traverses edges of specified types and visits nodes of specified types.

    Args:
        graph: HeterogeneousGraph instance
        start_id: starting node ID
        edge_types: set of edge types to traverse (None = all)
        node_types: set of node types to visit (None = all)
        max_depth: optional depth limit

    Returns:
        dict {node_id: depth_from_start} for all reachable nodes
    """
    if not graph.has_node(start_id):
        return {}

    visited = set()
    result = {}

    def _dfs(node_id, depth):
        if node_id in visited:
            return
        node = graph.get_node(node_id)
        if node_types and node.type not in node_types:
            return

        visited.add(node_id)
        result[node_id] = depth

        if max_depth is not None and depth >= max_depth:
            return

        for edge in graph.get_neighbors(node_id):
            if edge_types and edge.type not in edge_types:
                continue
            neighbor = edge.target
            if neighbor not in visited:
                _dfs(neighbor, depth + 1)

    _dfs(start_id, 0)
    return result


def bfs_shortest_path(graph, start_id, end_id, edge_types=None, node_types=None):
    """BFS-based shortest path (unweighted) between two nodes with type filtering.

    Returns:
        (distance, path_list) or (None, []) if no path found
    """
    if not graph.has_node(start_id) or not graph.has_node(end_id):
        return None, []

    visited = {start_id}
    queue = deque([(start_id, 0, [start_id])])

    while queue:
        node_id, dist, path = queue.popleft()

        if node_id == end_id:
            return dist, path

        node = graph.get_node(node_id)
        if node_types and node.type not in node_types:
            continue

        for edge in graph.get_neighbors(node_id):
            if edge_types and edge.type not in edge_types:
                continue
            neighbor = edge.target
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1, path + [neighbor]))

    return None, []