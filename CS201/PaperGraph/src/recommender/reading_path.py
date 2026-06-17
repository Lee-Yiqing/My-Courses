"""Reading path planning using multi-weight Dijkstra.

Plans an optimal reading order from a start paper to a target paper.
The path minimizes "reading effort" by weighting citation/sequence edges
low (natural progression) and topic/author edges high (requires context switch).

If no direct path exists through paper nodes only, falls back to routing
through keyword/author nodes as "bridge" intermediaries.
"""

from ..graph.hgraph import HeterogeneousGraph
from ..graph.algorithms.dijkstra import multi_weight_dijkstra, reading_path_profile, influence_path_profile
from ..graph.algorithms.pagerank import top_pagerank_papers


def plan_reading_path(graph, start_id, end_id, mode="reading"):
    """Plan a reading path from start paper to end paper.

    Args:
        graph: HeterogeneousGraph
        start_id: starting paper ID
        end_id: target paper ID
        mode: "reading" (minimize effort) or "influence" (follow influence chain)

    Returns:
        dict with:
        - path: list of paper_ids in reading order
        - distance: total weighted distance
        - annotations: list explaining each step
        - mode: which profile was used
    """
    if not graph.has_node(start_id) or not graph.has_node(end_id):
        return {"path": [], "distance": None, "annotations": [], "mode": mode,
                "error": "Start or end node not found"}

    profile = reading_path_profile() if mode == "reading" else influence_path_profile()

    # Try 1: Direct path through paper nodes only
    dist, path = multi_weight_dijkstra(graph, start_id, end_id,
                                        weight_profile=profile,
                                        node_type_filter={"paper", "blog_post", "report"})
    if path:
        annotations = annotate_path(graph, path, profile)
        return {"path": path, "distance": dist, "annotations": annotations, "mode": mode}

    # Try 2: Allow routing through keyword nodes (broader connectivity)
    dist, path = multi_weight_dijkstra(graph, start_id, end_id,
                                        weight_profile=profile,
                                        node_type_filter={"paper", "blog_post", "report", "keyword"})
    if path:
        annotations = annotate_path(graph, path, profile)
        # Filter to only show paper nodes in the reading order
        paper_path = [nid for nid in path if graph.get_node(nid).type in ("paper", "blog_post", "report")]
        return {"path": paper_path, "distance": dist, "annotations": annotations, "mode": mode}

    # Try 3: Allow routing through all node types
    dist, path = multi_weight_dijkstra(graph, start_id, end_id,
                                        weight_profile=profile)
    if path:
        annotations = annotate_path(graph, path, profile)
        paper_path = [nid for nid in path if graph.get_node(nid).type in ("paper", "blog_post", "report")]
        return {"path": paper_path, "distance": dist, "annotations": annotations, "mode": mode}

    # Try 4: BFS unweighted shortest path as fallback
    from ..graph.algorithms.traversal import bfs_shortest_path
    dist, path = bfs_shortest_path(graph, start_id, end_id)
    if path:
        annotations = annotate_path(graph, path, None)
        paper_path = [nid for nid in path if graph.get_node(nid).type in ("paper", "blog_post", "report")]
        return {"path": paper_path, "distance": dist, "annotations": annotations, "mode": "bfs_fallback"}

    return {"path": [], "distance": None, "annotations": [], "mode": mode,
            "error": "No path found between these papers"}


def plan_path_to_most_influential(graph, start_id):
    """Plan reading path from start paper to the most influential paper.

    Uses PageRank to find the top paper, then plans a reading path to it.
    """
    top = top_pagerank_papers(graph, top_k=1)
    if not top:
        return {"path": [], "distance": None, "annotations": [], "mode": "influence",
                "error": "No influential paper found"}

    end_id = top[0][0]
    result = plan_reading_path(graph, start_id, end_id, mode="influence")
    result["target_paper"] = end_id
    result["target_title"] = top[0][2]
    return result


def annotate_path(graph, path, weight_profile):
    """Annotate each step in the path with edge type and reasoning."""
    annotations = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # Find the connecting edge
        connecting_edges = []
        for edge in graph.get_neighbors(u):
            if edge.target == v:
                connecting_edges.append(edge)

        if connecting_edges:
            # Pick the lowest-weight edge
            best = connecting_edges[0]
            if weight_profile and best.type in weight_profile:
                effort = round(best.weight * weight_profile[best.type], 2)
            else:
                effort = round(best.weight, 2)

            reason = explain_edge_type(best.type)
            annotations.append({
                "from": u,
                "to": v,
                "edge_type": best.type,
                "effort": effort,
                "reason": reason,
            })
        else:
            annotations.append({
                "from": u,
                "to": v,
                "edge_type": "unknown",
                "effort": 0,
                "reason": "direct connection",
            })

    return annotations


def explain_edge_type(edge_type):
    """Explain what an edge type means in reading context."""
    explanations = {
        "cite": "This paper cites the next one — natural reading progression",
        "cited_by": "The next paper cites this one — follow who built on this work",
        "sequence": "These papers are in the same research lineage",
        "topic": "These papers share a common research topic",
        "author_of": "Same author connects these papers",
        "method": "This paper uses a technique relevant to the next",
    }
    return explanations.get(edge_type, f"Connected via {edge_type}")


def format_reading_path(result, graph):
    """Format reading path result for CLI display."""
    lines = []
    lines.append(f"=== Reading Path Plan ({result['mode']} mode) ===")

    if "error" in result:
        lines.append(f"Error: {result['error']}")
        return "\n".join(lines)

    if not result["path"]:
        lines.append("No path found.")
        return "\n".join(lines)

    if result.get("target_paper"):
        lines.append(f"Destination: [{result['target_paper']}] {result['target_title']}")

    lines.append(f"Path length: {len(result['path'])} papers")
    lines.append(f"Total effort: {result['distance']}")

    lines.append("\nRecommended reading order:")
    for i, nid in enumerate(result["path"], 1):
        node = graph.get_node(nid)
        title = node.properties.get("title", "")[:55] if node else ""
        year = node.properties.get("year", "") if node else ""
        domain = node.properties.get("domain", "") if node else ""
        lines.append(f"  {i}. [{nid}] ({year}, Domain {domain}) {title}")

    if result["annotations"]:
        lines.append("\nStep-by-step reasoning:")
        for ann in result["annotations"]:
            from_node = graph.get_node(ann["from"])
            to_node = graph.get_node(ann["to"])
            lines.append(f"  {ann['from']} -> {ann['to']}: {ann['edge_type']} (effort={ann['effort']})")
            lines.append(f"    {ann['reason']}")

    return "\n".join(lines)