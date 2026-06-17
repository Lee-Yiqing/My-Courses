"""Research community discovery using connected components and meta-path analysis.

Finds clusters of papers that form cohesive research communities,
labels them by dominant domain and top keywords, and identifies
cross-community connections.
"""

from collections import defaultdict, Counter

from ..graph.hgraph import HeterogeneousGraph
from ..graph.algorithms.components import connected_components, component_labels, bridge_edges


def discover_communities(graph, edge_types=None):
    """Discover research communities in the paper graph.

    Args:
        graph: HeterogeneousGraph
        edge_types: edge types to consider for community structure.
            Default: cite + cited_by + topic (gives broader connectivity)

    Returns:
        list of community dicts with:
        - community_id, size, dominant_domain, top_keywords,
        - member_ids, member_titles, domain_distribution
    """
    if edge_types is None:
        edge_types = ["cite", "cited_by", "topic"]

    comps = connected_components(graph,
                                 node_type_filter=["paper", "blog_post", "report"],
                                 edge_type_filter=edge_types)

    labels = component_labels(graph, comps)

    # Enrich with paper titles and years
    communities = []
    for label in labels:
        members = []
        for nid in label["member_ids"]:
            node = graph.get_node(nid)
            if node and node.type in ("paper", "blog_post", "report"):
                members.append({
                    "id": nid,
                    "title": node.properties.get("title", ""),
                    "year": node.properties.get("year", ""),
                    "domain": node.properties.get("domain", ""),
                })

        communities.append({
            "community_id": label["component_id"],
            "size": label["size"],
            "dominant_domain": label["dominant_domain"],
            "domain_distribution": label["domain_distribution"],
            "top_keywords": label["top_keywords"],
            "member_ids": label["member_ids"],
            "members": members,
        })

    return communities


def find_cross_community_connections(graph, communities):
    """Find connections between different communities.

    Looks for keyword nodes that connect papers in different communities,
    revealing cross-community research themes.
    """
    # Map each paper to its community
    paper_to_community = {}
    for comm in communities:
        for nid in comm["member_ids"]:
            paper_to_community[nid] = comm["community_id"]

    # Find keyword bridges
    cross_connections = []
    keyword_nodes = graph.get_nodes_by_type("keyword")

    for kw_id in keyword_nodes:
        # Which communities does this keyword connect?
        connected_communities = set()
        connected_papers = {}
        for edge in graph.get_neighbors(kw_id, "topic"):
            paper_id = edge.target
            if paper_id in paper_to_community:
                cid = paper_to_community[paper_id]
                connected_communities.add(cid)
                connected_papers.setdefault(cid, []).append(paper_id)

        if len(connected_communities) >= 2:
            kw_node = graph.get_node(kw_id)
            kw_name = kw_node.properties.get("name", "")
            cross_connections.append({
                "keyword": kw_name,
                "keyword_id": kw_id,
                "connects_communities": list(connected_communities),
                "papers_per_community": connected_papers,
            })

    # Sort by number of communities connected
    cross_connections.sort(key=lambda x: -len(x["connects_communities"]))
    return cross_connections


def format_communities(communities):
    """Format community discovery results for CLI display."""
    lines = []
    lines.append("=== Research Community Discovery ===")
    lines.append(f"Found {len(communities)} communities:")

    # Only show communities with >= 2 members in detail
    significant = [c for c in communities if c["size"] >= 2]
    singleton = [c for c in communities if c["size"] == 1]

    for comm in significant:
        lines.append(f"\n  Community {comm['community_id']}: {comm['size']} papers")
        lines.append(f"  Dominant domain: {comm['dominant_domain']}")
        lines.append(f"  Domain distribution: {comm['domain_distribution']}")
        kw_str = ", ".join(f"{k}({v})" for k, v in comm["top_keywords"][:5])
        lines.append(f"  Top keywords: {kw_str}")
        lines.append(f"  Members:")
        for m in comm["members"]:
            lines.append(f"    [{m['id']}] ({m['year']}) {m['title'][:55]}")

    lines.append(f"\n  Singleton papers (not in any community): {len(singleton)}")
    for comm in singleton[:5]:
        m = comm["members"][0] if comm["members"] else {}
        lines.append(f"    [{m.get('id','')}] {m.get('title','')[:50]}")
    if len(singleton) > 5:
        lines.append(f"    ... and {len(singleton)-5} more")

    return "\n".join(lines)