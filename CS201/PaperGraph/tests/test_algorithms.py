"""Test all 5 graph algorithms on the real PaperGraph."""

from src.graph.builder import build_graph
from src.graph.algorithms.traversal import type_filtered_bfs, bfs_shortest_path
from src.graph.algorithms.dijkstra import multi_weight_dijkstra, reading_path_profile
from src.graph.algorithms.pagerank import top_pagerank_papers
from src.graph.algorithms.components import connected_components, bridge_edges, component_labels
from src.graph.algorithms.metapath import extract_metapath_subgraph, related_papers_via_metapath, METAPATH_PAP


def main():
    print("Building graph...")
    g = build_graph()
    print()

    # === Test 1: Type-filtered BFS ===
    print("--- Test 1: Type-filtered BFS ---")
    # BFS from B1 (HumanEval) only through cite edges, paper nodes
    result = type_filtered_bfs(g, "B1", edge_types={"cite", "cited_by"}, node_types={"paper"})
    print(f"BFS from B1 (cite/cited_by, paper-only): {len(result)} papers reachable")
    for nid, dist in sorted(result.items(), key=lambda x: x[1]):
        node = g.get_node(nid)
        print(f"  dist={dist}: {nid} - {node.properties.get('title', '')[:50]}")

    # BFS from D1 through all edges, all types
    result2 = type_filtered_bfs(g, "D1", node_types={"paper"}, max_depth=3)
    print(f"BFS from D1 (all edges, paper-only, depth=3): {len(result2)} papers reachable")
    print()

    # === Test 2: Multi-weight Dijkstra ===
    print("--- Test 2: Multi-weight Dijkstra ---")
    profile = reading_path_profile()
    dist, path = multi_weight_dijkstra(g, "A1", "D1", weight_profile=profile,
                                        node_type_filter={"paper", "blog_post"})
    if path:
        print(f"Reading path from A1 (Vibe Coding) to D1 (SWE-bench):")
        print(f"  Distance: {dist:.2f}")
        print(f"  Path: {path}")
        for nid in path:
            node = g.get_node(nid)
            print(f"    {nid}: {node.properties.get('title', '')[:60]}")
    else:
        print("  No path found (graph may not be fully connected for papers)")
    print()

    # === Test 3: Heterogeneous PageRank ===
    print("--- Test 3: Heterogeneous PageRank ---")
    type_weights = {"cite": 2.0, "cited_by": 2.0, "author_of": 1.0, "topic": 1.0, "method": 1.5}
    top = top_pagerank_papers(g, top_k=10, type_weights=type_weights)
    print("Top 10 papers by PageRank:")
    for rank, (nid, score, title) in enumerate(top, 1):
        print(f"  {rank}. {nid} (score={score:.6f}): {title[:60]}")
    print()

    # === Test 4: Connected Components ===
    print("--- Test 4: Connected Components & Bridge Edges ---")
    comps = connected_components(g, node_type_filter=["paper", "blog_post", "report"],
                                 edge_type_filter=["cite", "cited_by", "sequence"])
    print(f"Components (cite/sequence edges only): {len(comps)}")
    labels = component_labels(g, comps)
    for label in labels:
        print(f"  Component {label['component_id']}: {label['size']} nodes, "
              f"domain={label['dominant_domain']}, keywords={label['top_keywords'][:3]}")

    bridges = bridge_edges(g, node_type_filter=["paper", "blog_post", "report"],
                          edge_type_filter=["cite", "cited_by", "sequence"])
    print(f"Bridge edges: {len(bridges)}")
    for u, v, et in bridges:
        nu = g.get_node(u)
        nv = g.get_node(v)
        print(f"  Bridge: {u}({nu.properties.get('title','')[:30]}) -- {v}({nv.properties.get('title','')[:30]}) via {et}")
    print()

    # Now with topic edges too (broader connectivity)
    comps2 = connected_components(g, node_type_filter=["paper", "blog_post", "report"],
                                  edge_type_filter=["cite", "cited_by", "sequence", "topic"])
    print(f"Components (cite/sequence/topic): {len(comps2)}")
    for i, comp in enumerate(comps2[:3]):
        print(f"  Component {i}: {len(comp)} nodes")
    print()

    # === Test 5: Meta-path Subgraph ===
    print("--- Test 5: Meta-path Subgraph ---")
    # PAP from B5 (DeepSeek-Coder)
    result = extract_metapath_subgraph(g, "B5", METAPATH_PAP, max_results=10)
    print(f"PAP from B5 (DeepSeek-Coder): {len(result['target_nodes'])} related papers")
    for nid in result["target_nodes"]:
        node = g.get_node(nid)
        print(f"  {nid}: {node.properties.get('title', '')[:60]}")

    # Related papers from D1
    related = related_papers_via_metapath(g, "D1", top_k=5)
    print(f"Related papers for D1 (SWE-bench):")
    for nid, score, title, explanation in related:
        print(f"  {nid} (score={score:.1f}): {title[:50]} -- {explanation}")

    print("\n=== All 5 algorithms tested successfully ===")


if __name__ == "__main__":
    main()