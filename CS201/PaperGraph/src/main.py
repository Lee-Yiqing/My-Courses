"""PaperGraph CLI — unified entry point for all features.

Commands:
  recommend   — Find related papers for a given paper
  reading-path — Plan optimal reading order between two papers
  communities — Discover research communities
  gaps        — Detect knowledge gaps between communities
  pagerank    — Show top papers by influence score
  build       — Build and display graph summary
"""

import argparse
import sys

from src.graph.builder import build_graph
from src.graph.algorithms.pagerank import top_pagerank_papers
from src.recommender.related_papers import recommend_related, format_recommendations
from src.recommender.reading_path import plan_reading_path, plan_path_to_most_influential, format_reading_path
from src.recommender.community import discover_communities, find_cross_community_connections, format_communities
from src.recommender.gaps import detect_knowledge_gaps, format_gaps


def cmd_recommend(graph, args):
    results = recommend_related(graph, args.query, top_k=args.top)
    print(format_recommendations(results))


def cmd_reading_path(graph, args):
    if args.to:
        result = plan_reading_path(graph, args.from_paper, args.to, mode=args.mode)
    else:
        result = plan_path_to_most_influential(graph, args.from_paper)
    print(format_reading_path(result, graph))


def cmd_communities(graph, args):
    communities = discover_communities(graph)
    print(format_communities(communities))

    if args.verbose:
        print("\n--- Cross-community connections ---")
        cross = find_cross_community_connections(graph, communities)
        print(f"Found {len(cross)} cross-community keyword bridges:")
        for conn in cross[:5]:
            print(f"  Keyword: {conn['keyword']} connects {len(conn['connects_communities'])} communities")
            for cid, papers in conn["papers_per_community"].items():
                print(f"    Community {cid}: {[p for p in papers[:3]]}")


def cmd_gaps(graph, args):
    result = detect_knowledge_gaps(graph)
    print(format_gaps(result))


def cmd_pagerank(graph, args):
    type_weights = {"cite": 2.0, "cited_by": 2.0, "author_of": 1.0, "topic": 1.0, "method": 1.5}
    top = top_pagerank_papers(graph, top_k=args.top, type_weights=type_weights)

    print(f"=== Top {args.top} Papers by PageRank ===")
    for rank, (nid, score, title) in enumerate(top, 1):
        node = graph.get_node(nid)
        year = node.properties.get("year", "") if node else ""
        domain = node.properties.get("domain", "") if node else ""
        print(f"  {rank}. [{nid}] (score={score:.6f}, {year}, Domain {domain}) {title[:55]}")


def cmd_build(graph, args):
    print(graph.summary())


def main():
    parser = argparse.ArgumentParser(prog="papergraph",
                                     description="PaperGraph: Heterogeneous literature graph for vibe coding research")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # recommend
    p_rec = subparsers.add_parser("recommend", help="Find related papers")
    p_rec.add_argument("--query", required=True, help="Paper ID to find recommendations for (e.g. B1)")
    p_rec.add_argument("--top", type=int, default=5, help="Number of recommendations")

    # reading-path
    p_path = subparsers.add_parser("reading-path", help="Plan reading order")
    p_path.add_argument("--from", dest="from_paper", required=True, help="Start paper ID")
    p_path.add_argument("--to", default=None, help="End paper ID (default: most influential)")
    p_path.add_argument("--mode", choices=["reading", "influence"], default="reading",
                        help="Path optimization mode")

    # communities
    p_comm = subparsers.add_parser("communities", help="Discover research communities")
    p_comm.add_argument("--verbose", action="store_true", help="Show cross-community connections")

    # gaps
    p_gap = subparsers.add_parser("gaps", help="Detect knowledge gaps")

    # pagerank
    p_pr = subparsers.add_parser("pagerank", help="Show top papers by influence")
    p_pr.add_argument("--top", type=int, default=10, help="Number of papers to show")

    # build
    p_build = subparsers.add_parser("build", help="Build graph and show summary")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Build graph
    print("Building PaperGraph...")
    graph = build_graph()
    print()

    # Dispatch
    dispatch = {
        "recommend": cmd_recommend,
        "reading-path": cmd_reading_path,
        "communities": cmd_communities,
        "gaps": cmd_gaps,
        "pagerank": cmd_pagerank,
        "build": cmd_build,
    }

    cmd_func = dispatch.get(args.command)
    if cmd_func:
        cmd_func(graph, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()