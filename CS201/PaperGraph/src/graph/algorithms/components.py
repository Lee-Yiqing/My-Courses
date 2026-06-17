"""Connected components and bridge edge detection on heterogeneous graph.

Course connection:
- Connected components via BFS (lecture W09-12, section on connectivity)
- Bridge edge detection via DFS with disc/low arrays (extension of Tarjan SCC,
  lecture W09-12 section on Tarjan/Kosaraju)

Bridge edges: edges whose removal disconnects a component.
In citation graphs, bridges identify "knowledge connectors" — papers that
link otherwise isolated research communities.
"""

from collections import deque

from ..hgraph import HeterogeneousGraph
from .traversal import type_filtered_bfs


def connected_components(graph, node_type_filter=None, edge_type_filter=None):
    """Find connected components in a type-filtered undirected subgraph.

    Creates an undirected version of the filtered subgraph, then uses BFS
    to find each component (same pattern as course lecture).

    Args:
        graph: HeterogeneousGraph
        node_type_filter: only include these node types (None = all)
        edge_type_filter: only include these edge types (None = all)

    Returns:
        list of sets, each set = node_ids in one component.
        Sorted by size (largest first).
    """
    # Get target nodes
    if node_type_filter:
        target_ids = set()
        for nt in node_type_filter:
            target_ids.update(graph.get_nodes_by_type(nt))
    else:
        target_ids = set(graph.all_node_ids())

    # Build undirected adjacency from filtered edges
    undirected_adj = {}
    for nid in target_ids:
        undirected_adj[nid] = set()

    edge_types = edge_type_filter or graph.all_edge_types()
    for et in edge_types:
        for edge in graph.get_edges_by_type(et):
            if edge.source in target_ids and edge.target in target_ids:
                undirected_adj[edge.source].add(edge.target)
                undirected_adj[edge.target].add(edge.source)

    # BFS-based component finding (same as course)
    visited = set()
    components = []

    for start in target_ids:
        if start in visited:
            continue

        # BFS from start
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in undirected_adj.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    # Sort by size
    components.sort(key=lambda c: -len(c))
    return components


def bridge_edges(graph, node_type_filter=None, edge_type_filter=None):
    """Find bridge edges (edges whose removal disconnects a component).

    Uses DFS with disc/low arrays — same technique as Tarjan's SCC algorithm
    from the course. Extension: applied to undirected type-filtered subgraph
    to find bridges instead of strongly connected components.

    An edge (u, v) is a bridge iff low[v] > disc[u]
    (v's subtree cannot reach back to u or u's ancestors).

    Args:
        graph: HeterogeneousGraph
        node_type_filter: only include these node types
        edge_type_filter: only include these edge types

    Returns:
        list of (node_id1, node_id2, edge_type) tuples that are bridges.
        Each bridge includes the edge_type for context.
    """
    # Get target nodes
    if node_type_filter:
        target_ids = set()
        for nt in node_type_filter:
            target_ids.update(graph.get_nodes_by_type(nt))
    else:
        target_ids = set(graph.all_node_ids())

    # Build undirected adjacency with edge type tracking
    undirected_adj = {}
    edge_type_map = {}  # (u,v) or (v,u) -> edge_type
    edge_types = edge_type_filter or graph.all_edge_types()
    for et in edge_types:
        for edge in graph.get_edges_by_type(et):
            if edge.source in target_ids and edge.target in target_ids:
                s, t = edge.source, edge.target
                undirected_adj.setdefault(s, set()).add(t)
                undirected_adj.setdefault(t, set()).add(s)
                edge_type_map[(s, t)] = et
                edge_type_map[(t, s)] = et

    # Tarjan-style DFS with disc and low arrays
    disc = {}
    low = {}
    timer = [0]
    bridges = []
    visited = set()

    def _dfs(u, parent):
        visited.add(u)
        disc[u] = timer[0]
        low[u] = timer[0]
        timer[0] += 1

        for v in undirected_adj.get(u, set()):
            if v not in target_ids:
                continue
            if v not in visited:
                _dfs(v, u)
                low[u] = min(low[u], low[v])
                # Bridge condition: v cannot reach back to u or ancestors
                if low[v] > disc[u]:
                    et = edge_type_map.get((u, v), "unknown")
                    bridges.append((u, v, et))
            elif v != parent:
                # Back edge: update low[u]
                low[u] = min(low[u], disc[v])

    for node in target_ids:
        if node not in visited:
            _dfs(node, None)

    return bridges


def component_labels(graph, components):
    """Label components with their dominant domain and top keywords.

    Returns:
        list of dicts with component info:
        {component_id, size, dominant_domain, top_keywords, member_ids}
    """
    labels = []
    for i, comp in enumerate(components):
        domains = []
        keywords = []
        for nid in comp:
            node = graph.get_node(nid)
            if node.type in ("paper", "blog_post", "report"):
                domains.append(node.properties.get("domain", ""))
                # Collect keywords connected to this paper
                for edge in graph.get_in_neighbors(nid, "topic"):
                    kw_node = graph.get_node(edge.source)
                    if kw_node and kw_node.type == "keyword":
                        keywords.append(kw_node.properties.get("name", ""))

        # Count domains
        from collections import Counter
        domain_counts = Counter(domains)
        keyword_counts = Counter(keywords)

        labels.append({
            "component_id": i,
            "size": len(comp),
            "dominant_domain": domain_counts.most_common(1)[0][0] if domain_counts else "",
            "domain_distribution": dict(domain_counts),
            "top_keywords": keyword_counts.most_common(5),
            "member_ids": list(comp),
        })

    return labels