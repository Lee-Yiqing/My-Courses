"""Pre-compute all algorithm results and dump to JSON for static deployment.

Run this script to generate all data files that the frontend needs.
GitHub Actions will run this on every push, then deploy to GitHub Pages.
"""

import json
from collections import defaultdict
from pathlib import Path

from src.graph.builder import build_graph
from src.graph.algorithms.traversal import type_filtered_bfs
from src.graph.algorithms.dijkstra import multi_weight_dijkstra, reading_path_profile
from src.graph.algorithms.pagerank import heterogeneous_pagerank, top_pagerank_papers
from src.graph.algorithms.components import connected_components, bridge_edges, component_labels
from src.graph.algorithms.metapath import related_papers_via_metapath
from src.graph.algorithms.traces import (
    bfs_trace_on_subgraph, dijkstra_trace_on_subgraph, pagerank_trace_on_subgraph,
    components_trace_on_subgraph, bridges_trace_on_subgraph,
)

DATA_DIR = Path(__file__).resolve().parent / "static" / "data"
DATA_DIR.mkdir(exist_ok=True)


def serialize_node(g, nid):
    node = g.get_node(nid)
    return {
        "id": nid,
        "type": node.type,
        "label": node.properties.get("title", node.properties.get("name", nid)),
        "domain": node.properties.get("domain", ""),
        "year": node.properties.get("year", 0),
        "confidence": node.properties.get("confidence", ""),
        "arxiv_id": node.properties.get("arxiv_id", ""),
    }


def serialize_paper(g, nid):
    """Rich serialization for paper nodes — includes authors and keywords."""
    node = g.get_node(nid)
    # Collect authors
    authors = []
    for edge in g.get_in_neighbors(nid, "author_of"):
        author_node = g.get_node(edge.source)
        if author_node and author_node.type == "author":
            authors.append(author_node.properties.get("name", edge.source))
    # Collect keywords via topic edges (keyword -> paper)
    keywords = []
    for edge in g.get_in_neighbors(nid, "topic"):
        kw_node = g.get_node(edge.source)
        if kw_node and kw_node.type == "keyword":
            keywords.append(kw_node.properties.get("name", edge.source))
    # Collect keywords via method edges (paper -> keyword)
    for edge in g.get_neighbors(nid, "method"):
        kw_node = g.get_node(edge.target)
        if kw_node and kw_node.type == "keyword":
            kw_name = kw_node.properties.get("name", edge.target)
            if kw_name not in keywords:
                keywords.append(kw_name)

    return {
        "id": nid,
        "type": node.type,
        "label": node.properties.get("title", node.properties.get("name", nid)),
        "domain": node.properties.get("domain", ""),
        "year": node.properties.get("year", 0),
        "confidence": node.properties.get("confidence", ""),
        "arxiv_id": node.properties.get("arxiv_id", ""),
        "authors": authors,
        "keywords": keywords,
    }


def main():
    print("Building graph...")
    g = build_graph()
    print(g.summary())

    # === 1. Full graph data (all nodes + edges) ===
    print("\nComputing graph structure...")
    nodes = []
    for nid in g.all_node_ids():
        nodes.append(serialize_node(g, nid))

    edges = []
    seen = set()
    for nid in g.all_node_ids():
        for e in g.get_neighbors(nid):
            key = (e.source, e.target, e.type)
            if key not in seen:
                seen.add(key)
                edges.append({
                    "source": e.source, "target": e.target,
                    "type": e.type, "weight": e.weight,
                })

    graph_data = {"nodes": nodes, "edges": edges, "stats": {
        "num_nodes": len(nodes), "num_edges": len(edges),
    }}
    with open(DATA_DIR / "graph.json", "w") as f:
        json.dump(graph_data, f, ensure_ascii=False)
    print(f"  graph.json: {len(nodes)} nodes, {len(edges)} edges")

    # === 2. Paper-only subgraph with implicit edges ===
    print("\nComputing paper subgraph + implicit edges...")
    paper_ids = g.get_nodes_by_type("paper")
    blog_ids = g.get_nodes_by_type("blog_post")
    report_ids = g.get_nodes_by_type("report")
    all_paper_ids = paper_ids + blog_ids + report_ids

    paper_nodes = [serialize_paper(g, pid) for pid in all_paper_ids]

    # Build implicit edges between papers
    implicit_edges = []

    # a) cite edges (paper -> paper, already exist)
    for edge in g.get_edges_by_type("cite"):
        if edge.source in all_paper_ids and edge.target in all_paper_ids:
            implicit_edges.append({
                "source": edge.source, "target": edge.target,
                "type": "cite", "weight": edge.weight,
            })

    # b) sequence edges
    for edge in g.get_edges_by_type("sequence"):
        if edge.source in all_paper_ids and edge.target in all_paper_ids:
            implicit_edges.append({
                "source": edge.source, "target": edge.target,
                "type": "sequence", "weight": edge.weight,
            })

    # c) shared_keyword edges: papers sharing a keyword get an undirected edge
    keyword_to_papers = defaultdict(set)
    for pid in all_paper_ids:
        for edge in g.get_in_neighbors(pid, "topic"):
            if g.get_node(edge.source) and g.get_node(edge.source).type == "keyword":
                keyword_to_papers[edge.source].add(pid)
        for edge in g.get_neighbors(pid, "method"):
            if g.get_node(edge.target) and g.get_node(edge.target).type == "keyword":
                keyword_to_papers[edge.target].add(pid)

    for kw_id, p_set in keyword_to_papers.items():
        p_list = sorted(p_set)
        for i in range(len(p_list)):
            for j in range(i + 1, len(p_list)):
                # Weight = number of shared keywords (more shared = stronger)
                implicit_edges.append({
                    "source": p_list[i], "target": p_list[j],
                    "type": "shared_keyword", "weight": 1.0,
                })

    # Deduplicate shared_keyword edges and compute real weights
    kw_edge_counts = defaultdict(int)
    kw_edges_set = set()
    for e in implicit_edges:
        if e["type"] == "shared_keyword":
            pair = tuple(sorted([e["source"], e["target"]]))
            kw_edge_counts[pair] += 1
            kw_edges_set.add(pair)

    # Replace with weighted versions
    implicit_edges = [e for e in implicit_edges if e["type"] != "shared_keyword"]
    for pair in kw_edges_set:
        count = kw_edge_counts[pair]
        implicit_edges.append({
            "source": pair[0], "target": pair[1],
            "type": "shared_keyword", "weight": count,
        })

    # d) shared_author edges
    author_to_papers = defaultdict(set)
    for pid in all_paper_ids:
        for edge in g.get_in_neighbors(pid, "author_of"):
            if g.get_node(edge.source) and g.get_node(edge.source).type == "author":
                author_to_papers[edge.source].add(pid)

    auth_edge_counts = defaultdict(int)
    for auth_id, p_set in author_to_papers.items():
        p_list = sorted(p_set)
        for i in range(len(p_list)):
            for j in range(i + 1, len(p_list)):
                pair = tuple(sorted([p_list[i], p_list[j]]))
                auth_edge_counts[pair] += 1

    for pair, count in auth_edge_counts.items():
        if count >= 2:  # Only show edges where authors share 2+ papers
            implicit_edges.append({
                "source": pair[0], "target": pair[1],
                "type": "shared_author", "weight": count,
            })

    paper_graph = {
        "nodes": paper_nodes,
        "edges": implicit_edges,
        "stats": {"num_nodes": len(paper_nodes), "num_edges": len(implicit_edges)},
    }
    with open(DATA_DIR / "paper_graph.json", "w") as f:
        json.dump(paper_graph, f, ensure_ascii=False)
    cite_count = sum(1 for e in implicit_edges if e["type"] == "cite")
    skw_count = sum(1 for e in implicit_edges if e["type"] == "shared_keyword")
    sa_count = sum(1 for e in implicit_edges if e["type"] == "shared_author")
    seq_count = sum(1 for e in implicit_edges if e["type"] == "sequence")
    print(f"  paper_graph.json: {len(paper_nodes)} papers, {len(implicit_edges)} edges")
    print(f"    cite={cite_count}, shared_keyword={skw_count}, shared_author={sa_count}, sequence={seq_count}")

    # === 3. PageRank scores ===
    print("\nComputing PageRank...")
    pr_scores = heterogeneous_pagerank(g, node_type_filter=["paper", "blog_post", "report"])
    ranked = sorted(pr_scores.items(), key=lambda x: -x[1])
    pagerank_data = []
    for nid, score in ranked:
        pagerank_data.append({
            **serialize_node(g, nid),
            "score": round(score, 6),
        })
    with open(DATA_DIR / "pagerank.json", "w") as f:
        json.dump(pagerank_data, f, ensure_ascii=False)
    print(f"  pagerank.json: {len(pagerank_data)} papers ranked")

    # === 4. Communities ===
    print("\nComputing communities...")
    comps = connected_components(g, node_type_filter={"paper"}, edge_type_filter={"cite", "topic"})
    labels = component_labels(g, comps)
    communities_data = []
    for label in labels:
        if label["size"] >= 2:
            members = [serialize_node(g, nid) for nid in label["member_ids"]]
            communities_data.append({
                "id": label["component_id"],
                "size": label["size"],
                "dominant_domain": label["dominant_domain"],
                "top_keywords": label["top_keywords"],
                "members": members,
            })
    with open(DATA_DIR / "communities.json", "w") as f:
        json.dump(communities_data, f, ensure_ascii=False)
    print(f"  communities.json: {len(communities_data)} communities")

    # === 5. Knowledge gaps (bridge edges) ===
    print("\nComputing knowledge gaps...")
    bridges = bridge_edges(g, node_type_filter={"paper"}, edge_type_filter={"cite", "topic"})
    gaps_data = []
    for s, t, et in bridges:
        gaps_data.append({
            "source": serialize_node(g, s),
            "target": serialize_node(g, t),
            "edge_type": et,
        })
    with open(DATA_DIR / "gaps.json", "w") as f:
        json.dump(gaps_data, f, ensure_ascii=False)
    print(f"  gaps.json: {len(gaps_data)} bridge edges")

    # === 6. BFS neighborhoods ===
    print("\nComputing BFS neighborhoods...")
    bfs_data = {}
    for pid in paper_ids:
        result = type_filtered_bfs(g, pid, edge_types={"cite", "topic", "sequence"},
                                   node_types={"paper"}, max_depth=3)
        neighbors = [{"id": nid, "distance": d} for nid, d in result.items() if nid != pid]
        if neighbors:
            bfs_data[pid] = neighbors
    with open(DATA_DIR / "bfs.json", "w") as f:
        json.dump(bfs_data, f, ensure_ascii=False)
    print(f"  bfs.json: {len(bfs_data)} papers with neighborhoods")

    # === 7. Reading paths ===
    print("\nComputing reading paths...")
    wp = reading_path_profile()
    top5_ids = [r["id"] for r in pagerank_data[:5]]
    paths_data = {}
    for pid in paper_ids:
        best_path = None
        best_end = None
        for end_id in top5_ids:
            if pid == end_id:
                continue
            dist, path = multi_weight_dijkstra(g, pid, end_id, wp)
            if dist is not None and (best_path is None or dist < best_path):
                best_path = dist
                best_end = end_id
        if best_path is not None:
            dist, path = multi_weight_dijkstra(g, pid, best_end, wp)
            path_info = [serialize_node(g, nid) for nid in path]
            paths_data[pid] = {
                "end_id": best_end,
                "distance": round(dist, 4),
                "path": path_info,
            }
    with open(DATA_DIR / "paths.json", "w") as f:
        json.dump(paths_data, f, ensure_ascii=False)
    print(f"  paths.json: {len(paths_data)} papers with reading paths")

    # === 8. Recommendations ===
    print("\nComputing recommendations...")
    recommend_data = {}
    for pid in paper_ids:
        recs = related_papers_via_metapath(g, pid, top_k=5)
        recommend_data[pid] = [
            {"id": nid, "score": round(score, 4), "title": title, "explanation": explanation}
            for nid, score, title, explanation in recs
        ]
    with open(DATA_DIR / "recommend.json", "w") as f:
        json.dump(recommend_data, f, ensure_ascii=False)
    print(f"  recommend.json: {len(recommend_data)} papers with recommendations")

    # === 9. Search index ===
    print("\nBuilding search index...")
    search_data = []
    for nid in g.all_node_ids():
        node = g.get_node(nid)
        label = node.properties.get("title", node.properties.get("name", ""))
        search_data.append({
            "id": nid, "type": node.type, "label": label,
            "domain": node.properties.get("domain", ""),
            "year": node.properties.get("year", 0),
        })
    with open(DATA_DIR / "search.json", "w") as f:
        json.dump(search_data, f, ensure_ascii=False)
    print(f"  search.json: {len(search_data)} items indexed")

    # === 10. Algorithm execution traces (on paper subgraph) ===
    print("\nComputing algorithm traces...")

    # Build simple dict-based subgraph for trace functions
    subgraph_nodes = {n["id"]: n for n in paper_nodes}
    subgraph_edges = implicit_edges

    # BFS trace from B1 (HumanEval — most central paper)
    bfs_trace_data = bfs_trace_on_subgraph(subgraph_nodes, subgraph_edges, "B1", max_depth=3)
    with open(DATA_DIR / "bfs_trace.json", "w") as f:
        json.dump(bfs_trace_data, f, ensure_ascii=False)
    print(f"  bfs_trace.json: {len(bfs_trace_data)} steps from B1")

    # Dijkstra trace from B1 to D1 (SWE-bench — popular reading target)
    dijkstra_trace_data = dijkstra_trace_on_subgraph(subgraph_nodes, subgraph_edges, "B1", "D1")
    with open(DATA_DIR / "dijkstra_trace.json", "w") as f:
        json.dump(dijkstra_trace_data, f, ensure_ascii=False)
    print(f"  dijkstra_trace.json: {len(dijkstra_trace_data)} steps B1→D1")

    # PageRank trace on paper subgraph
    pagerank_trace_data = pagerank_trace_on_subgraph(subgraph_nodes, subgraph_edges)
    with open(DATA_DIR / "pagerank_trace.json", "w") as f:
        json.dump(pagerank_trace_data, f, ensure_ascii=False)
    print(f"  pagerank_trace.json: {len(pagerank_trace_data)} iterations")

    # Components trace (on full subgraph)
    components_trace_data = components_trace_on_subgraph(subgraph_nodes, subgraph_edges)
    with open(DATA_DIR / "components_trace.json", "w") as f:
        json.dump(components_trace_data, f, ensure_ascii=False)
    print(f"  components_trace.json: {len(components_trace_data)} components (full subgraph)")

    # Bridges trace on cite-only subgraph (more meaningful for knowledge gaps)
    cite_only_edges = [e for e in implicit_edges if e["type"] in ("cite", "sequence")]
    cite_components_trace = components_trace_on_subgraph(subgraph_nodes, cite_only_edges)
    with open(DATA_DIR / "cite_components_trace.json", "w") as f:
        json.dump(cite_components_trace, f, ensure_ascii=False)
    print(f"  cite_components_trace.json: {len(cite_components_trace)} components (cite-only)")

    # Bridges trace on full subgraph
    bridges_trace_data = bridges_trace_on_subgraph(subgraph_nodes, subgraph_edges)
    with open(DATA_DIR / "bridges_trace.json", "w") as f:
        json.dump(bridges_trace_data, f, ensure_ascii=False)
    print(f"  bridges_trace.json: {len(bridges_trace_data)} DFS steps (full subgraph)")

    # Bridges trace on cite-only subgraph (for knowledge gaps visualization)
    cite_bridges_trace = bridges_trace_on_subgraph(subgraph_nodes, cite_only_edges)
    # Count bridges from the DFS steps (not summary step)
    cite_bridges_count = sum(len(s.get("new_bridges", [])) for s in cite_bridges_trace if s.get("node") is not None)
    with open(DATA_DIR / "cite_bridges_trace.json", "w") as f:
        json.dump(cite_bridges_trace, f, ensure_ascii=False)
    print(f"  cite_bridges_trace.json: {len(cite_bridges_trace)} DFS steps, {cite_bridges_count} bridges (cite-only)")

    # === 11. Generate self-contained HTML ===
    print("\nGenerating self-contained HTML...")
    all_data = {
        "paper_graph": paper_graph,
        "graph": graph_data,
        "search": search_data,
        "pagerank": pagerank_data,
        "communities": communities_data,
        "gaps": gaps_data,
        "bfs": bfs_data,
        "paths": paths_data,
        "recommend": recommend_data,
        "bfs_trace": bfs_trace_data,
        "dijkstra_trace": dijkstra_trace_data,
        "pagerank_trace": pagerank_trace_data,
        "components_trace": components_trace_data,
        "bridges_trace": bridges_trace_data,
        "cite_components_trace": cite_components_trace,
        "cite_bridges_trace": cite_bridges_trace,
    }
    data_json = json.dumps(all_data, ensure_ascii=False)

    html_template = (Path(__file__).resolve().parent / "static" / "index.html").read_text()
    inject = f'<script id="embedded-data" type="application/json">{data_json}</script>\n'
    html_out = html_template.replace('<script>\n', inject + '<script>\n')
    (Path(__file__).resolve().parent / "static" / "index_viewer.html").write_text(html_out)
    print(f"  index_viewer.html: self-contained viewer ({len(html_out)} bytes)")

    print("\n=== All data computed and saved to static/data/ ===")


if __name__ == "__main__":
    main()
