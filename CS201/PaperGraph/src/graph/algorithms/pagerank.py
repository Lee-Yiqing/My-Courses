"""Heterogeneous PageRank on heterogeneous graph.

Not in the course syllabus, but builds on adjacency list iteration
(the same pattern as BFS neighborhood expansion). Innovation point:
different edge types have different transition probability weights,
extending the random surfer model to typed edges.

Algorithm:
1. Initialize all nodes with score 1/N
2. Each iteration: score(u) = (1-d)/N + d * sum(score(v) * P(v->u))
   where P(v->u) depends on edge type and v's outgoing degree for that type
3. Convergence: max score change < tolerance
"""

from ..hgraph import HeterogeneousGraph


def heterogeneous_pagerank(graph, damping=0.85, max_iter=100,
                           tolerance=1e-6, type_weights=None,
                           node_type_filter=None):
    """PageRank on heterogeneous graph with type-weighted transitions.

    Args:
        graph: HeterogeneousGraph
        damping: damping factor (default 0.85)
        max_iter: max iterations
        tolerance: convergence threshold
        type_weights: dict {edge_type: transition_weight}.
            Higher = more likely for random surfer to follow this edge type.
            If None, uniform weights.
        node_type_filter: only compute PageRank for these node types.
            If None, compute for all nodes.

    Returns:
        dict {node_id: pagerank_score}
    """
    # Determine target nodes
    if node_type_filter:
        target_ids = []
        for nt in node_type_filter:
            target_ids.extend(graph.get_nodes_by_type(nt))
    else:
        target_ids = graph.all_node_ids()

    if not target_ids:
        return {}

    N = len(target_ids)
    default_type_weight = 1.0

    # Initialize scores
    scores = {nid: 1.0 / N for nid in target_ids}

    # Build outgoing degree per edge type for each node
    # This is needed to normalize transition probabilities
    out_degree_by_type = {}
    for nid in target_ids:
        out_degree_by_type[nid] = {}
        for edge in graph.get_neighbors(nid):
            if edge.type not in out_degree_by_type[nid]:
                out_degree_by_type[nid][edge.type] = 0
            out_degree_by_type[nid][edge.type] += 1

    for iteration in range(max_iter):
        new_scores = {nid: 0.0 for nid in target_ids}

        # For each target node, accumulate contributions from inbound edges
        for nid in target_ids:
            for edge in graph.get_in_neighbors(nid):
                source = edge.source
                if source not in scores:
                    continue  # source not in target set

                edge_type = edge.type
                tw = type_weights.get(edge_type, default_type_weight) if type_weights else default_type_weight

                # Transition probability: type_weight / out_degree_of_this_type_from_source
                source_out_deg = out_degree_by_type.get(source, {}).get(edge_type, 1)
                if source_out_deg == 0:
                    source_out_deg = 1

                transition_prob = tw / source_out_deg

                # Normalize: divide by total type-weighted out-degree of source
                total_tw_out = 0.0
                for et, deg in out_degree_by_type.get(source, {}).items():
                    etw = type_weights.get(et, default_type_weight) if type_weights else default_type_weight
                    total_tw_out += etw * deg
                if total_tw_out == 0:
                    total_tw_out = 1.0

                transition_prob = tw / total_tw_out

                new_scores[nid] += scores[source] * transition_prob

        # Apply damping
        for nid in target_ids:
            new_scores[nid] = (1 - damping) / N + damping * new_scores[nid]

        # Check convergence
        max_change = 0.0
        for nid in target_ids:
            change = abs(new_scores[nid] - scores[nid])
            if change > max_change:
                max_change = change

        scores = new_scores

        if max_change < tolerance:
            break

    return scores


def top_pagerank_papers(graph, top_k=10, **kwargs):
    """Get top-k papers by PageRank score.

    Returns:
        list of (paper_id, score, title) sorted by score descending
    """
    scores = heterogeneous_pagerank(graph, node_type_filter=["paper", "blog_post", "report"],
                                    **kwargs)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    results = []
    for nid, score in ranked[:top_k]:
        node = graph.get_node(nid)
        title = node.properties.get("title", "")
        results.append((nid, score, title))

    return results