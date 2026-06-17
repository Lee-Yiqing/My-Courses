"""Meta-path subgraph extraction on heterogeneous graph.

Course connection: constrained DFS/BFS — same traversal algorithms with
type sequence constraints. Each step checks that the current node type
and outgoing edge type match the next element in the meta-path sequence.

Common meta-paths:
- PAP: Paper -> Author -> Paper (papers sharing an author)
- PKP: Paper -> Keyword -> Paper (papers sharing a keyword)
- PP_cite: Paper -> cite -> Paper (citation chain)
- PP_seq: Paper -> sequence -> Paper (research lineage)
"""

from collections import defaultdict

from ..hgraph import HeterogeneousGraph


def extract_metapath_subgraph(graph, start_id, metapath, max_results=50):
    """Extract nodes reachable via paths matching a meta-path pattern.

    Args:
        graph: HeterogeneousGraph
        start_id: starting node ID
        metapath: list of (node_type, edge_type) pairs defining the path pattern.
            Example for PAP (Paper-Author-Paper):
            [("paper", "author_of"), ("author", "author_of")]
            Note: edge_type is the type of edge FROM the current node type.
            For "author_of": author -> paper, so traversing it backward
            from a paper means finding its authors.

            Simplified format: just list alternating node_types and edge_types:
            ["paper", "author_of", "author", "author_of", "paper"]

    Returns:
        dict with:
        - "target_nodes": set of destination node_ids reachable via this metapath
        - "paths": list of actual paths found (each path = list of node_ids)
        - "path_count": dict {target_node_id: number_of_paths_reaching_it}
    """
    if not graph.has_node(start_id):
        return {"target_nodes": set(), "paths": [], "path_count": {}}

    # Parse simplified metapath format
    if isinstance(metapath, list) and len(metapath) >= 3 and isinstance(metapath[0], str):
        # Simplified format: ["node_type", "edge_type", "node_type", ...]
        node_types_seq = [metapath[i] for i in range(0, len(metapath), 2)]
        edge_types_seq = [metapath[i] for i in range(1, len(metapath), 2)]
    else:
        # List of tuples format
        node_types_seq = [nt for nt, et in metapath]
        edge_types_seq = [et for nt, et in metapath]

    target_node_type = node_types_seq[-1]

    # DFS-based path search with type constraints
    results_nodes = set()
    results_paths = []
    path_count = defaultdict(int)

    def _dfs(current_id, step, path):
        if step == len(edge_types_seq):
            # Reached end of metapath — record result
            node = graph.get_node(current_id)
            if node and node.type == target_node_type:
                results_nodes.add(current_id)
                results_paths.append(path + [current_id])
                path_count[current_id] += 1
            return

        expected_edge_type = edge_types_seq[step]
        expected_next_type = node_types_seq[step + 1]

        # Find edges of expected type from current node
        # Some edge types go in "forward" direction (paper->keyword via "method")
        # Others go in "reverse" direction (paper<-author via "author_of")
        # We need to handle both directions

        # Forward edges (current -> neighbor)
        for edge in graph.get_neighbors(current_id, expected_edge_type):
            neighbor = edge.target
            neighbor_node = graph.get_node(neighbor)
            if neighbor_node and neighbor_node.type == expected_next_type:
                _dfs(neighbor, step + 1, path + [current_id])

        # Reverse edges (neighbor -> current) — e.g., "author_of" goes author->paper
        # To go from paper to author, we traverse "author_of" in reverse
        for edge in graph.get_in_neighbors(current_id, expected_edge_type):
            neighbor = edge.source
            neighbor_node = graph.get_node(neighbor)
            if neighbor_node and neighbor_node.type == expected_next_type:
                _dfs(neighbor, step + 1, path + [current_id])

        if len(results_paths) >= max_results:
            return  # early stop

    _dfs(start_id, 0, [])
    return {
        "target_nodes": results_nodes,
        "paths": results_paths[:max_results],
        "path_count": dict(path_count),
    }


# Predefined common meta-paths
METAPATH_PAP = ["paper", "author_of", "author", "author_of", "paper"]
METAPATH_PKP = ["paper", "method", "keyword", "topic", "paper"]
METAPATH_PP_CITE = ["paper", "cite", "paper"]
METAPATH_PP_SEQ = ["paper", "sequence", "paper"]
METAPATH_PKP_TOPIC = ["paper", "topic", "keyword", "topic", "paper"]


def related_papers_via_metapath(graph, paper_id, top_k=5):
    """Find related papers using multiple meta-paths.

    Returns:
        list of (paper_id, score, explanation) sorted by score descending.
        score = sum of path_counts across all meta-paths.
    """
    all_scores = defaultdict(float)
    all_explanations = defaultdict(list)

    meta_paths = {
        "PAP (shared author)": METAPATH_PAP,
        "PKP (shared keyword)": METAPATH_PKP,
        "PP (citation)": METAPATH_PP_CITE,
    }

    for name, mp in meta_paths.items():
        result = extract_metapath_subgraph(graph, paper_id, mp, max_results=30)
        for nid, count in result["path_count"].items():
            if nid != paper_id:  # exclude self
                all_scores[nid] += count
                all_explanations[nid].append(f"{name}: {count} paths")

    ranked = sorted(all_scores.items(), key=lambda x: -x[1])
    output = []
    for nid, score in ranked[:top_k]:
        node = graph.get_node(nid)
        title = node.properties.get("title", "") if node else ""
        explanation = "; ".join(all_explanations[nid])
        output.append((nid, score, title, explanation))

    return output