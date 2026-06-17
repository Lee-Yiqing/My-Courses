"""Related paper recommendation using PageRank + meta-path analysis.

Combines heterogeneous PageRank scores with meta-path frequency to rank
related papers. Each recommendation includes an explanation of WHY it's
related (shared author, shared keyword, citation chain, etc).
"""

from collections import defaultdict

from ..graph.hgraph import HeterogeneousGraph
from ..graph.algorithms.pagerank import heterogeneous_pagerank
from ..graph.algorithms.metapath import extract_metapath_subgraph, METAPATH_PAP, METAPATH_PKP, METAPATH_PKP_TOPIC


def recommend_related(graph, query_id, top_k=5, alpha=0.5, beta=0.5):
    """Recommend papers related to the query paper.

    Args:
        graph: HeterogeneousGraph
        query_id: paper ID to find recommendations for
        top_k: number of recommendations
        alpha: weight for PageRank score
        beta: weight for meta-path frequency

    Returns:
        list of dicts with:
        - paper_id, title, score, explanations, domain
    """
    # Step 1: Compute PageRank
    type_weights = {"cite": 2.0, "cited_by": 2.0, "author_of": 1.0, "topic": 1.0, "method": 1.5}
    pr_scores = heterogeneous_pagerank(graph, node_type_filter=["paper", "blog_post", "report"],
                                       type_weights=type_weights)

    # Step 2: Find papers reachable via meta-paths
    meta_paths = {
        "PAP (shared author)": METAPATH_PAP,
        "PKP (shared keyword)": METAPATH_PKP,
        "PKP via topic": METAPATH_PKP_TOPIC,
    }

    metapath_scores = defaultdict(float)
    explanations = defaultdict(list)

    for name, mp in meta_paths.items():
        result = extract_metapath_subgraph(graph, query_id, mp, max_results=50)
        for nid, count in result["path_count"].items():
            if nid != query_id:
                metapath_scores[nid] += count
                explanations[nid].append(f"{name}: {count} path(s)")

    # Also add direct citation neighbors
    for edge in graph.get_neighbors(query_id, "cite"):
        metapath_scores[edge.target] += 2
        target = graph.get_node(edge.target)
        explanations[edge.target].append(f"cited by {query_id}")

    for edge in graph.get_in_neighbors(query_id, "cite"):
        metapath_scores[edge.source] += 2
        explanations[edge.source].append(f"cites {query_id}")

    for edge in graph.get_in_neighbors(query_id, "cited_by"):
        metapath_scores[edge.source] += 2
        source_node = graph.get_node(edge.source)
        explanations[edge.source].append(f"cites {query_id}")

    # Step 3: Combine PageRank and meta-path scores
    # Normalize each to [0, 1]
    max_pr = max(pr_scores.values()) if pr_scores else 1.0
    max_mp = max(metapath_scores.values()) if metapath_scores else 1.0

    combined = {}
    all_candidates = set(metapath_scores.keys())
    for nid in all_candidates:
        pr_norm = pr_scores.get(nid, 0) / max_pr
        mp_norm = metapath_scores[nid] / max_mp
        combined[nid] = alpha * pr_norm + beta * mp_norm

    # Sort by combined score
    ranked = sorted(combined.items(), key=lambda x: -x[1])

    # Format output
    results = []
    for nid, score in ranked[:top_k]:
        node = graph.get_node(nid)
        if node is None:
            continue
        results.append({
            "paper_id": nid,
            "title": node.properties.get("title", ""),
            "domain": node.properties.get("domain", ""),
            "year": node.properties.get("year", ""),
            "score": round(score, 4),
            "pagerank": round(pr_scores.get(nid, 0), 6),
            "metapath_score": round(metapath_scores.get(nid, 0), 1),
            "explanations": explanations.get(nid, []),
        })

    return results


def format_recommendations(results):
    """Format recommendation results for CLI display."""
    lines = []
    lines.append(f"=== Related Paper Recommendations ===")
    lines.append(f"Found {len(results)} recommendations:")
    for i, r in enumerate(results, 1):
        lines.append(f"\n  {i}. [{r['paper_id']}] {r['title']}")
        lines.append(f"     Domain: {r['domain']}, Year: {r['year']}")
        lines.append(f"     Score: {r['score']} (PR={r['pagerank']}, MP={r['metapath_score']})")
        for exp in r['explanations']:
            lines.append(f"     - {exp}")
    return "\n".join(lines)