"""Multi-weight Dijkstra on heterogeneous graph.

Course connection: heap-optimized Dijkstra (lecture W09-12, shortest path section).
Extension: edge weight = base_weight * weight_profile[edge_type], allowing
different edge types to contribute differently to path cost.
Uses heapq for priority queue, same as course template.
"""

import heapq

from ..hgraph import HeterogeneousGraph


def multi_weight_dijkstra(graph, start_id, end_id, weight_profile=None,
                          node_type_filter=None):
    """Shortest weighted path on heterogeneous graph with type-dependent weights.

    Args:
        graph: HeterogeneousGraph
        start_id: start node ID
        end_id: target node ID
        weight_profile: dict {edge_type: weight_multiplier}.
            e.g. {"cite": 0.5, "sequence": 0.3, "topic": 1.0}
            Lower weight = easier traversal (shorter "reading effort" path).
            If None, all edge types use weight=1.0.
        node_type_filter: set of node types to allow traversal through.
            If None, all types allowed.

    Returns:
        (distance, path_list) or (None, []) if no path found
    """
    if not graph.has_node(start_id) or not graph.has_node(end_id):
        return None, []

    if weight_profile is None:
        weight_profile = {}

    # Initialize: same pattern as course Dijkstra template
    dist = {start_id: 0.0}
    prev = {}
    heap = [(0.0, start_id)]

    while heap:
        d, u = heapq.heappop(heap)

        if u == end_id:
            # Reconstruct path
            path = []
            node = end_id
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(start_id)
            path.reverse()
            return d, path

        # Skip if we already found a shorter path to u
        if d > dist.get(u, float('inf')):
            continue

        # Node type filter
        if node_type_filter:
            node_u = graph.get_node(u)
            if node_u.type not in node_type_filter and u != start_id:
                continue

        # Relax neighbors
        for edge in graph.get_neighbors(u):
            # Calculate effective weight based on edge type
            if edge.type in weight_profile:
                effective_weight = edge.weight * weight_profile[edge.type]
            else:
                effective_weight = edge.weight  # default

            v = edge.target

            # Node type filter for target
            if node_type_filter:
                node_v = graph.get_node(v)
                if node_v.type not in node_type_filter and v != end_id:
                    continue

            new_dist = d + effective_weight
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return None, []  # no path found


def reading_path_profile():
    """Default weight profile optimized for reading path planning.

    cite edges = low weight (natural reading progression)
    sequence edges = low (following a research line)
    topic edges = high (jumping across topics = more effort)
    author_of = high (switching to author perspective = effort)
    method = moderate
    """
    return {
        "cite": 0.5,
        "sequence": 0.3,
        "method": 0.8,
        "topic": 2.0,
        "author_of": 3.0,
        "cited_by": 5.0,  # reverse citation = not natural reading order
    }


def influence_path_profile():
    """Weight profile for finding influential/cited papers.

    cited_by = low (follow who cites this paper = influence chain)
    cite = moderate
    """
    return {
        "cited_by": 0.5,
        "cite": 1.0,
        "author_of": 2.0,
        "topic": 2.0,
        "method": 2.0,
        "sequence": 1.5,
    }