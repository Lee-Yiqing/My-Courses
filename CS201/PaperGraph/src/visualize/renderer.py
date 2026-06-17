"""Static graph visualization using matplotlib.

Self-implemented force-directed layout (no networkx).
Different edge types use different colors/styles.
Different node types use different shapes/colors.
"""

import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from ..graph.hgraph import HeterogeneousGraph

# Visual style mappings
EDGE_STYLES = {
    "cite":       {"color": "#e74c3c", "style": "-",  "width": 1.5, "alpha": 0.7, "arrow": True},
    "cited_by":   {"color": "#e74c3c", "style": "-",  "width": 1.0, "alpha": 0.3, "arrow": True},
    "author_of":  {"color": "#2ecc71", "style": "--", "width": 0.5, "alpha": 0.25, "arrow": True},
    "topic":      {"color": "#3498db", "style": ":",  "width": 0.5, "alpha": 0.25, "arrow": True},
    "method":     {"color": "#e67e22", "style": ":",  "width": 0.5, "alpha": 0.25, "arrow": True},
    "sequence":   {"color": "#9b59b6", "style": "-",  "width": 2.0, "alpha": 0.8, "arrow": True},
}

DOMAIN_COLORS = {
    "A": "#f1c40f",  # gold
    "B": "#3498db",  # blue
    "C": "#2ecc71",  # green
    "D": "#e74c3c",  # red
    "E": "#9b59b6",  # purple
}

NODE_TYPE_SHAPES = {
    "paper":      "o",   # circle
    "blog_post":  "*",   # star
    "report":     "D",   # diamond
    "author":     "s",   # square
    "keyword":    "^",   # triangle
}


def force_directed_layout(graph, iterations=50, k=0.8, gravity=0.01,
                          node_type_filter=None, paper_size_boost=True):
    """Simple force-directed layout. No external libraries.

    Args:
        graph: HeterogeneousGraph
        iterations: number of relaxation iterations
        k: repulsion strength
        gravity: pull toward center
        node_type_filter: only layout these node types

    Returns:
        dict {node_id: (x, y)} position mapping
    """
    # Get target nodes
    if node_type_filter:
        target_ids = set()
        for nt in node_type_filter:
            target_ids.update(graph.get_nodes_by_type(nt))
    else:
        target_ids = set(graph.all_node_ids())

    # Seed positions: cluster papers by domain, scatter others
    positions = {}
    domain_centers = {
        "A": (-3, 3), "B": (3, 3), "C": (-3, -3),
        "D": (3, -3), "E": (0, 0),
    }

    for nid in target_ids:
        node = graph.get_node(nid)
        if node.type in ("paper", "blog_post", "report"):
            domain = node.properties.get("domain", "B")
            cx, cy = domain_centers.get(domain, (0, 0))
            positions[nid] = (cx + random.uniform(-1.5, 1.5),
                              cy + random.uniform(-1.5, 1.5))
        elif node.type == "author":
            # Place authors near their papers
            papers = [e.target for e in graph.get_neighbors(nid, "author_of")
                      if e.target in target_ids]
            if papers:
                avg_x = sum(positions.get(p, (0,0))[0] for p in papers) / len(papers)
                avg_y = sum(positions.get(p, (0,0))[1] for p in papers) / len(papers)
                positions[nid] = (avg_x + random.uniform(-0.5, 0.5),
                                  avg_y + random.uniform(-0.5, 0.5))
            else:
                positions[nid] = (random.uniform(-4, 4), random.uniform(-4, 4))
        elif node.type == "keyword":
            # Place keywords near their papers
            papers = [e.target for e in graph.get_neighbors(nid, "topic")
                      if e.target in target_ids]
            if papers:
                avg_x = sum(positions.get(p, (0,0))[0] for p in papers) / len(papers)
                avg_y = sum(positions.get(p, (0,0))[1] for p in papers) / len(papers)
                positions[nid] = (avg_x + random.uniform(-0.3, 0.3),
                                  avg_y + random.uniform(-0.3, 0.3))
            else:
                positions[nid] = (random.uniform(-4, 4), random.uniform(-4, 4))

    # Build undirected adjacency for layout
    adj = {}
    for nid in target_ids:
        adj[nid] = set()
    for et in graph.all_edge_types():
        for edge in graph.get_edges_by_type(et):
            if edge.source in target_ids and edge.target in target_ids:
                adj[edge.source].add(edge.target)
                adj[edge.target].add(edge.source)

    # Force-directed iterations
    for _ in range(iterations):
        forces = {nid: [0.0, 0.0] for nid in target_ids}

        # Repulsion: all pairs
        for u in target_ids:
            for v in target_ids:
                if u == v:
                    continue
                dx = positions[u][0] - positions[v][0]
                dy = positions[u][1] - positions[v][1]
                dist = math.sqrt(dx*dx + dy*dy) + 0.01
                f = k / (dist * dist)
                forces[u][0] += f * dx / dist
                forces[u][1] += f * dy / dist

        # Attraction: connected pairs
        for u in target_ids:
            for v in adj.get(u, set()):
                if v not in target_ids:
                    continue
                dx = positions[v][0] - positions[u][0]
                dy = positions[v][1] - positions[u][1]
                dist = math.sqrt(dx*dx + dy*dy) + 0.01
                f = dist * 0.05
                forces[u][0] += f * dx / dist
                forces[u][1] += f * dy / dist

        # Gravity: pull toward center
        for nid in target_ids:
            forces[nid][0] -= gravity * positions[nid][0]
            forces[nid][1] -= gravity * positions[nid][1]

        # Update positions
        max_move = 0.5
        for nid in target_ids:
            dx = min(max(forces[nid][0], -max_move), max_move)
            dy = min(max(forces[nid][1], -max_move), max_move)
            positions[nid] = (positions[nid][0] + dx, positions[nid][1] + dy)

    return positions


def render_full_graph(graph, output_path="docs/graph_full.png",
                      show_author=True, show_keyword=True,
                      show_edge_types=None, title="PaperGraph: Heterogeneous Literature Graph"):
    """Render the full heterogeneous graph.

    Args:
        graph: HeterogeneousGraph
        output_path: where to save the PNG
        show_author: include author nodes
        show_keyword: include keyword nodes
        show_edge_types: which edge types to draw (None = all)
        title: plot title
    """
    if show_edge_types is None:
        show_edge_types = ["cite", "cited_by", "author_of", "topic", "method", "sequence"]

    # Determine which node types to layout
    node_types = ["paper", "blog_post", "report"]
    if show_author:
        node_types.append("author")
    if show_keyword:
        node_types.append("keyword")

    print("Computing layout...")
    positions = force_directed_layout(graph, iterations=80,
                                      node_type_filter=node_types)

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Draw edges first (behind nodes)
    drawn_edges = set()
    for et in show_edge_types:
        style = EDGE_STYLES.get(et, {"color": "gray", "style": "-", "width": 0.5, "alpha": 0.3, "arrow": True})
        for edge in graph.get_edges_by_type(et):
            u, v = edge.source, edge.target
            if u not in positions or v not in positions:
                continue
            # Avoid drawing duplicate undirected edges
            pair = (min(u,v), max(u,v))
            if et in ("cite", "cited_by") and pair in drawn_edges:
                continue
            drawn_edges.add(pair)

            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->" if style["arrow"] else "-",
                                        color=style["color"],
                                        linestyle=style["style"],
                                        lw=style["width"],
                                        alpha=style["alpha"],
                                        shrinkA=3, shrinkB=3))

    # Draw nodes
    for nid, (x, y) in positions.items():
        node = graph.get_node(nid)
        ntype = node.type

        if ntype in ("paper", "blog_post", "report"):
            domain = node.properties.get("domain", "B")
            color = DOMAIN_COLORS.get(domain, "gray")
            shape = NODE_TYPE_SHAPES.get(ntype, "o")
            size = 120 if ntype == "paper" else 180
            label = nid
            ax.scatter(x, y, s=size, c=color, marker=shape, zorder=5,
                       edgecolors='black', linewidths=0.5)
            ax.annotate(label, (x, y), fontsize=7, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points', fontweight='bold')

        elif ntype == "author":
            ax.scatter(x, y, s=20, c='gray', marker='s', zorder=3, alpha=0.4)
            # Don't label authors (too many)

        elif ntype == "keyword":
            kw_name = node.properties.get("name", "")
            # Only label keywords that connect >= 3 papers
            n_papers = len(graph.get_neighbors(nid, "topic"))
            if n_papers >= 3:
                ax.scatter(x, y, s=40, c='#a8e6cf', marker='^', zorder=4, alpha=0.6)
                ax.annotate(kw_name, (x, y), fontsize=6, ha='center',
                            xytext=(0, 3), textcoords='offset points', alpha=0.7)
            else:
                ax.scatter(x, y, s=15, c='#a8e6cf', marker='^', zorder=3, alpha=0.2)

    # Legend
    legend_elements = []
    # Node types
    for domain, color in DOMAIN_COLORS.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Domain {domain}',
                                       markerfacecolor=color, markersize=10))
    legend_elements.append(Line2D([0], [0], marker='^', color='w', label='Keyword',
                                   markerfacecolor='#a8e6cf', markersize=8))
    legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Author',
                                   markerfacecolor='gray', markersize=8))

    # Edge types
    for et, style in EDGE_STYLES.items():
        if et in show_edge_types:
            legend_elements.append(Line2D([0], [0], color=style["color"],
                                           linestyle=style["style"],
                                           linewidth=style["width"],
                                           label=f'{et} edge'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")


def render_papers_only(graph, output_path="docs/graph_papers.png",
                       edge_types=["cite", "cited_by", "sequence"],
                       title="PaperGraph: Citation & Sequence Network"):
    """Render only paper nodes with specified edge types (cleaner view)."""
    positions = force_directed_layout(graph, iterations=100,
                                      node_type_filter=["paper", "blog_post", "report"])

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Draw edges
    for et in edge_types:
        style = EDGE_STYLES.get(et, {"color": "gray", "style": "-", "width": 1.0, "alpha": 0.5, "arrow": True})
        for edge in graph.get_edges_by_type(et):
            u, v = edge.source, edge.target
            if u not in positions or v not in positions:
                continue
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->",
                                        color=style["color"],
                                        linestyle=style["style"],
                                        lw=style["width"],
                                        alpha=style["alpha"]))

    # Draw paper nodes with labels
    for nid, (x, y) in positions.items():
        node = graph.get_node(nid)
        domain = node.properties.get("domain", "B")
        color = DOMAIN_COLORS.get(domain, "gray")
        ax.scatter(x, y, s=150, c=color, marker='o', zorder=5,
                   edgecolors='black', linewidths=0.8)

        title_short = node.properties.get("title", nid)[:25]
        ax.annotate(f"{nid}\n{title_short}", (x, y), fontsize=7,
                    ha='center', va='top',
                    xytext=(0, -8), textcoords='offset points')

    # Legend
    legend_elements = []
    for domain, color in DOMAIN_COLORS.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       label=f'Domain {domain}',
                                       markerfacecolor=color, markersize=10))
    for et in edge_types:
        style = EDGE_STYLES.get(et)
        if style:
            legend_elements.append(Line2D([0], [0], color=style["color"],
                                           linestyle=style["style"],
                                           linewidth=style["width"],
                                           label=f'{et} edge'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")


def render_pagerank_heatmap(graph, output_path="docs/pagerank_heatmap.png"):
    """Render paper nodes colored by PageRank score."""
    from ..graph.algorithms.pagerank import heterogeneous_pagerank

    scores = heterogeneous_pagerank(graph,
                                     node_type_filter=["paper", "blog_post", "report"],
                                     type_weights={"cite": 2.0, "cited_by": 2.0, "author_of": 1.0, "topic": 1.0})

    positions = force_directed_layout(graph, iterations=100,
                                      node_type_filter=["paper", "blog_post", "report"])

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_title("PaperGraph: PageRank Influence Heatmap", fontsize=16, fontweight='bold')

    # Get scores for paper nodes
    paper_ids = [nid for nid in positions.keys()]
    paper_scores = [scores.get(nid, 0) for nid in paper_ids]

    # Normalize scores to color range
    max_score = max(paper_scores) if paper_scores else 1
    min_score = min(paper_scores) if paper_scores else 0
    range_score = max_score - min_score if max_score != min_score else 1

    # Color by score
    colors = [(s - min_score) / range_score for s in paper_scores]
    sizes = [80 + 200 * ((s - min_score) / range_score) for s in paper_scores]

    xs = [positions[nid][0] for nid in paper_ids]
    ys = [positions[nid][1] for nid in paper_ids]

    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap='YlOrRd',
                         zorder=5, edgecolors='black', linewidths=0.5)

    # Label top-5 papers
    top5 = sorted(zip(paper_ids, paper_scores), key=lambda x: -x[1])[:5]
    for nid, score in top5:
        x, y = positions[nid]
        node = graph.get_node(nid)
        title_short = node.properties.get("title", "")[:25]
        ax.annotate(f"{nid}: {title_short}", (x, y), fontsize=9,
                    ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points',
                    fontweight='bold', color='darkred')

    # Label other papers with just their ID
    for nid in paper_ids:
        if nid not in [t[0] for t in top5]:
            x, y = positions[nid]
            ax.annotate(nid, (x, y), fontsize=7,
                        ha='center', va='top',
                        xytext=(0, -5), textcoords='offset points', alpha=0.7)

    plt.colorbar(scatter, ax=ax, label='PageRank Score', shrink=0.8)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")