"""Flask backend serving graph data and algorithm results via REST API."""

import json
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

from src.graph.builder import build_graph
from src.graph.algorithms.traversal import type_filtered_bfs, type_filtered_dfs
from src.graph.algorithms.dijkstra import multi_weight_dijkstra
from src.graph.algorithms.pagerank import heterogeneous_pagerank
from src.graph.algorithms.components import connected_components, bridge_edges
from src.graph.algorithms.metapath import extract_metapath_subgraph

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))
graph = None


def init_graph():
    global graph
    if graph is None:
        graph = build_graph()
    return graph


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/graph")
def get_graph():
    g = init_graph()
    nodes = []
    for nid, node in g._nodes.items():
        nodes.append({
            "id": nid,
            "type": node.type,
            "label": node.properties.get("title", node.properties.get("name", nid)),
            "domain": node.properties.get("domain", ""),
            "year": node.properties.get("year", 0),
            "confidence": node.properties.get("confidence", ""),
        })

    edges = []
    seen = set()
    for nid in g.all_node_ids():
        for e in g.get_neighbors(nid):
            key = (e.source, e.target, e.type)
            if key not in seen:
                seen.add(key)
                edges.append({
                    "source": e.source,
                    "target": e.target,
                    "type": e.type,
                    "weight": e.weight,
                })

    return jsonify({"nodes": nodes, "edges": edges, "stats": {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
    }})


@app.route("/api/search")
def search():
    g = init_graph()
    query = request.args.get("q", "").lower()
    edge_types = request.args.get("edge_types", "").split(",")
    edge_types = [t for t in edge_types if t] or None
    node_types = request.args.get("node_types", "").split(",")
    node_types = [t for t in node_types if t] or None

    results = []
    for nid, node in g._nodes.items():
        label = node.properties.get("title", node.properties.get("name", ""))
        if query and query in label.lower():
            if node_types and node.type not in node_types:
                continue
            results.append({
                "id": nid,
                "type": node.type,
                "label": label,
                "domain": node.properties.get("domain", ""),
                "year": node.properties.get("year", 0),
            })

    return jsonify({"results": results, "total": len(results)})


@app.route("/api/bfs")
def run_bfs():
    g = init_graph()
    start = request.args.get("start", "")
    edge_types = request.args.get("edge_types", "").split(",")
    edge_types = set(t for t in edge_types if t) or None
    node_types = request.args.get("node_types", "").split(",")
    node_types = set(t for t in node_types if t) or None
    max_depth = int(request.args.get("max_depth", "3"))

    if not g.has_node(start):
        return jsonify({"error": f"Node {start} not found"}), 404

    visited = type_filtered_bfs(g, start, edge_types, node_types, max_depth)
    nodes = []
    for nid in visited:
        node = g.get_node(nid)
        nodes.append({
            "id": nid,
            "type": node.type,
            "label": node.properties.get("title", node.properties.get("name", nid)),
            "distance": visited[nid],
        })

    return jsonify({"start": start, "visited": nodes, "total": len(nodes)})


@app.route("/api/path")
def find_path():
    g = init_graph()
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    # Weight profile from query params
    cite_w = float(request.args.get("cite_weight", "0.5"))
    topic_w = float(request.args.get("topic_weight", "1.0"))
    author_w = float(request.args.get("author_weight", "2.0"))
    method_w = float(request.args.get("method_weight", "0.8"))
    seq_w = float(request.args.get("sequence_weight", "0.3"))

    weight_profile = {
        "cite": cite_w, "topic": topic_w, "author_of": author_w,
        "method": method_w, "sequence": seq_w,
    }

    if not g.has_node(start) or not g.has_node(end):
        return jsonify({"error": "Start or end node not found"}), 404

    result = multi_weight_dijkstra(g, start, end, weight_profile)
    if result is None:
        return jsonify({"error": "No path found", "start": start, "end": end})

    distance, path = result
    path_info = []
    for nid in path:
        node = g.get_node(nid)
        path_info.append({
            "id": nid,
            "type": node.type,
            "label": node.properties.get("title", node.properties.get("name", nid)),
        })

    return jsonify({"distance": distance, "path": path_info, "length": len(path)})


@app.route("/api/pagerank")
def run_pagerank():
    g = init_graph()
    damping = float(request.args.get("damping", "0.85"))
    node_type = request.args.get("node_type", "paper")
    cite_tw = float(request.args.get("cite_tw", "0.4"))
    topic_tw = float(request.args.get("topic_tw", "0.2"))
    author_tw = float(request.args.get("author_tw", "0.2"))
    method_tw = float(request.args.get("method_tw", "0.1"))
    seq_tw = float(request.args.get("sequence_tw", "0.1"))

    type_weights = {
        "cite": cite_tw, "topic": topic_tw, "author_of": author_tw,
        "method": method_tw, "sequence": seq_tw,
    }

    scores = heterogeneous_pagerank(g, damping=damping, type_weights=type_weights,
                                     node_type_filter=node_type)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    results = []
    for nid, score in ranked[:30]:
        node = g.get_node(nid)
        results.append({
            "id": nid,
            "type": node.type,
            "label": node.properties.get("title", node.properties.get("name", nid)),
            "score": round(score, 6),
            "domain": node.properties.get("domain", ""),
        })

    return jsonify({"scores": results, "total": len(scores)})


@app.route("/api/communities")
def find_communities():
    g = init_graph()
    edge_types = request.args.get("edge_types", "cite,topic").split(",")
    node_type = request.args.get("node_type", "paper")

    components = connected_components(g, node_type_filter=node_type, edge_type_filter=set(edge_types))

    communities = []
    for comp in components:
        if len(comp) < 2:
            continue
        labels = []
        for nid in comp:
            node = g.get_node(nid)
            labels.append(node.properties.get("title", nid))
        communities.append({"nodes": list(comp), "size": len(comp), "labels": labels[:3]})

    return jsonify({"communities": communities, "total": len(communities)})


@app.route("/api/gaps")
def find_gaps():
    g = init_graph()
    edge_types = request.args.get("edge_types", "cite,topic").split(",")
    node_type = request.args.get("node_type", "paper")

    bridges = bridge_edges(g, node_type_filter=node_type, edge_type_filter=set(edge_types))

    gap_info = []
    for s, t in bridges:
        sn = g.get_node(s)
        tn = g.get_node(t)
        gap_info.append({
            "source": {"id": s, "label": sn.properties.get("title", s)},
            "target": {"id": t, "label": tn.properties.get("title", t)},
        })

    return jsonify({"gaps": gap_info, "total": len(gap_info)})


@app.route("/api/recommend")
def recommend():
    g = init_graph()
    query = request.args.get("query", "")
    top_k = int(request.args.get("top", "5"))

    if not g.has_node(query):
        return jsonify({"error": f"Node {query} not found"}), 404

    # Use BFS to find neighborhood, then rank by PageRank
    scores = heterogeneous_pagerank(g, node_type_filter="paper")
    neighborhood = type_filtered_bfs(g, query, edge_types=None, node_types={"paper"}, max_depth=2)

    recs = []
    for nid, dist in neighborhood.items():
        if nid == query:
            continue
        score = scores.get(nid, 0)
        node = g.get_node(nid)
        recs.append({
            "id": nid,
            "label": node.properties.get("title", nid),
            "score": round(score, 6),
            "distance": dist,
            "domain": node.properties.get("domain", ""),
        })

    recs.sort(key=lambda x: -x["score"])
    return jsonify({"query": query, "recommendations": recs[:top_k]})


# --- Helper: iterate all edges ---
def _all_edges(g):
    """Flatten all edges from adjacency lists."""
    seen = set()
    for nid in g.all_node_ids():
        for e in g.get_neighbors(nid):
            key = (e.source, e.target, e.type)
            if key not in seen:
                seen.add(key)
                yield e