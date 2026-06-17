"""Knowledge gap detection using bridge edge analysis.

Bridge edges in the citation/topic graph represent single points of connection
between otherwise isolated research communities. The papers at these bridges
are "knowledge connectors" — they link two research areas that would otherwise
be disconnected.

Gaps are identified where two communities could benefit from more cross-domain
research but currently only connect through a single paper or keyword.
"""

from collections import defaultdict

from ..graph.hgraph import HeterogeneousGraph
from ..graph.algorithms.components import connected_components, bridge_edges


def detect_knowledge_gaps(graph, edge_types=None):
    """Detect knowledge gaps and connector papers in the literature.

    Args:
        graph: HeterogeneousGraph
        edge_types: edge types for the underlying undirected subgraph.
            Default: cite + cited_by + topic + sequence

    Returns:
        dict with:
        - gaps: list of gap descriptions
        - connectors: list of connector papers (bridge papers)
        - isolated_papers: papers not connected to any community
    """
    if edge_types is None:
        edge_types = ["cite", "cited_by", "topic", "sequence"]

    # Find components and bridges
    comps = connected_components(graph,
                                 node_type_filter=["paper", "blog_post", "report"],
                                 edge_type_filter=edge_types)

    bridges = bridge_edges(graph,
                          node_type_filter=["paper", "blog_post", "report"],
                          edge_type_filter=edge_types)

    # Map papers to components
    paper_to_comp = {}
    for i, comp in enumerate(comps):
        for nid in comp:
            paper_to_comp[nid] = i

    # Analyze each bridge as a knowledge gap
    gaps = []
    connectors = []

    for u, v, et in bridges:
        comp_u = paper_to_comp.get(u, -1)
        comp_v = paper_to_comp.get(v, -1)

        node_u = graph.get_node(u)
        node_v = graph.get_node(v)

        # Determine the domains on each side
        domain_u = node_u.properties.get("domain", "?") if node_u else "?"
        domain_v = node_v.properties.get("domain", "?") if node_v else "?"

        # Check if this bridge connects different domains
        is_cross_domain = domain_u != domain_v

        gap = {
            "bridge_edge": (u, v, et),
            "connector_paper": u,
            "connector_title": node_u.properties.get("title", "") if node_u else "",
            "connected_paper": v,
            "connected_title": node_v.properties.get("title", "") if node_v else "",
            "edge_type": et,
            "is_cross_domain": is_cross_domain,
            "domain_from": domain_u,
            "domain_to": domain_v,
            "gap_description": describe_gap(domain_u, domain_v, node_u, node_v, et, is_cross_domain),
        }
        gaps.append(gap)

        # Connector paper analysis
        # Find keywords unique to this bridge paper that bridge the two domains
        bridge_keywords = []
        for edge in graph.get_in_neighbors(u, "topic"):
            kw_node = graph.get_node(edge.source)
            if kw_node and kw_node.type == "keyword":
                bridge_keywords.append(kw_node.properties.get("name", ""))

        connector = {
            "paper_id": u,
            "title": node_u.properties.get("title", "") if node_u else "",
            "domain": domain_u,
            "year": node_u.properties.get("year", "") if node_u else "",
            "bridge_keywords": bridge_keywords,
            "connects_to": v,
            "connects_to_domain": domain_v,
        }
        connectors.append(connector)

    # Identify isolated papers (in singleton components)
    isolated = []
    for comp in comps:
        if len(comp) == 1:
            nid = list(comp)[0]
            node = graph.get_node(nid)
            if node and node.type in ("paper", "blog_post", "report"):
                isolated.append({
                    "paper_id": nid,
                    "title": node.properties.get("title", ""),
                    "domain": node.properties.get("domain", ""),
                    "year": node.properties.get("year", ""),
                })

    return {
        "gaps": gaps,
        "connectors": connectors,
        "isolated_papers": isolated,
        "total_bridges": len(bridges),
        "total_components": len(comps),
        "cross_domain_bridges": sum(1 for g in gaps if g["is_cross_domain"]),
    }


def describe_gap(domain_from, domain_to, node_from, node_from_v, edge_type, is_cross_domain):
    """Generate a human-readable description of a knowledge gap."""
    domain_names = {
        "A": "Vibe Coding / AI-Assisted Programming",
        "B": "LLM Code Generation / Code Models",
        "C": "Human-AI Collaboration / Developer Experience",
        "D": "Automated Code Repair / Code Intelligence",
        "E": "Prompt Engineering / Code Security",
    }

    d1 = domain_names.get(domain_from, f"Domain {domain_from}")
    d2 = domain_names.get(domain_to, f"Domain {domain_to}")

    if is_cross_domain:
        title = node_from.properties.get("title", "this paper")[:40]
        return (f"{d1} and {d2} are connected only through \"{title}\" "
                f"(via {edge_type} edge). Removing this connection would "
                f"disconnect these two research areas — a potential knowledge gap.")
    else:
        return (f"Within {d1}, two sub-communities are connected only "
                f"through a single {edge_type} edge. This suggests a "
                f"potential gap in research coverage.")


def format_gaps(result):
    """Format knowledge gap results for CLI display."""
    lines = []
    lines.append("=== Knowledge Gap Detection ===")
    lines.append(f"Total components: {result['total_components']}")
    lines.append(f"Total bridge edges: {result['total_bridges']}")
    lines.append(f"Cross-domain bridges: {result['cross_domain_bridges']}")

    if result["gaps"]:
        lines.append("\nKnowledge gaps (cross-domain bridges):")
        cross_gaps = [g for g in result["gaps"] if g["is_cross_domain"]]
        other_gaps = [g for g in result["gaps"] if not g["is_cross_domain"]]

        for g in cross_gaps:
            lines.append(f"\n  Gap: {g['domain_from']} <-> {g['domain_to']}")
            lines.append(f"  Connector: [{g['connector_paper']}] {g['connector_title'][:50]}")
            lines.append(f"  Connected to: [{g['connected_paper']}] {g['connected_title'][:50]}")
            lines.append(f"  Via: {g['edge_type']}")
            lines.append(f"  Description: {g['gap_description'][:100]}")

        if other_gaps:
            lines.append(f"\n  Intra-domain bridges: {len(other_gaps)}")
            for g in other_gaps[:3]:
                lines.append(f"    [{g['connector_paper']}] <-> [{g['connected_paper']}] via {g['edge_type']}")

    if result["connectors"]:
        lines.append("\nConnector papers (bridge nodes):")
        for c in result["connectors"][:5]:
            lines.append(f"  [{c['paper_id']}] {c['title'][:45]} ({c['domain']}) -> {c['connects_to_domain']}")
            if c["bridge_keywords"]:
                lines.append(f"    Bridge keywords: {', '.join(c['bridge_keywords'][:3])}")

    if result["isolated_papers"]:
        lines.append(f"\nIsolated papers (not in any community): {len(result['isolated_papers'])}")
        for p in result["isolated_papers"][:3]:
            lines.append(f"  [{p['paper_id']}] {p['title'][:45]} (Domain {p['domain']})")

    return "\n".join(lines)