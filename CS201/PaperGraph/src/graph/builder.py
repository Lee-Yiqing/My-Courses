"""Build heterogeneous graph from papers.json.

Creates paper, author, and keyword nodes with 5 edge types:
- cite: paper -> paper (directed, from citations field)
- author_of: author -> paper (directed)
- topic: keyword -> paper (directed, keyword relates to paper)
- method: paper -> keyword (directed, paper proposes/uses method)
- sequence: paper -> paper (directed, chronological succession in same domain)
"""

import json
from collections import defaultdict
from pathlib import Path

from ..graph.hgraph import HeterogeneousGraph

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAPERS_FILE = DATA_DIR / "papers.json"

# Canonical keyword taxonomy — ~30 conceptual categories
# Papers map to these via their raw keywords and domain tag
CANONICAL_KEYWORDS = {
    # Core concepts — each should connect multiple papers
    "code generation": ["code generation", "program synthesis", "codegen", "code completion",
                        "codex", "humaneval", "mbpp", "pass@k", "code evaluation",
                        "benchmark", "livecodebench", "evolving benchmark", "data contamination",
                        "code quality", "maintainability", "code understanding",
                        "code llm", "code language model"],
    "large language model": ["llm", "large language model", "language model", "deepseek-coder",
                             "starcoder2", "code llama", "wizardcoder", "magicoder",
                             "alphacode", "open-source", "open-source code model",
                             "moe", "mixture-of-experts", "infilling", "long context",
                             "evol-instruct", "oss-instruct", "the stack v2",
                             "code language model", "multi-size models",
                             "llm survey", "llm code safety", "llm vulnerability",
                             "llm repair", "llm-based", "llm for software engineering"],
    "automated program repair": ["automated program repair", "program repair", "apr",
                                 "self-repair", "selfrepair", "self-debugging", "self-debug",
                                 "repairagent", "fault localization", "swe-bench",
                                 "swe-agent", "agent-computer interface",
                                 "automated repair", "patch generation", "defects4j",
                                 "autonomous repair", "self-supervised", "tool-augmented",
                                 "multi-agent code repair", "collaborative repair",
                                 "rag repair", "context-aware repair"],
    "ai-assisted programming": ["ai-assisted programming", "ai-assisted development",
                                "ai-assisted se", "vibe coding", "prompt-driven development",
                                "prompt-driven", "developer practices", "developer workflows",
                                "developer agency", "ai coding practices", "ai",
                                "ai era software engineering", "se landscape"],
    "prompt engineering": ["prompt engineering", "prompting", "chain-of-thought", "cot",
                          "structured chain-of-thought", "scot", "few-shot", "zero-shot",
                          "retrieval-augmented generation", "rag", "context-aware",
                          "iterative refinement"],
    "code security": ["code security", "security", "vulnerability", "cwe",
                     "chatgpt security", "code vulnerability", "vulnerability taxonomy",
                     "llm code safety", "llm vulnerability", "security analysis",
                     "insecure code", "ai-generated code security", "security patterns",
                     "cwe-specific", "vulnerability mapping", "ai assistant security",
                     "security survey", "cwe analysis"],
    "developer experience": ["developer productivity", "developer experience",
                            "developer satisfaction", "copilot", "github copilot",
                            "usability", "trust", "verification", "over-reliance",
                            "cognitive load", "expectation vs experience",
                            "developer behavior", "developer role",
                            "new metrics", "measuring productivity",
                            "controlled experiment", "empirical study",
                            "real-world usage", "developer autonomy",
                            "productivity measurement"],
    "software engineering": ["software engineering", "se", "se tasks", "se lifecycle",
                            "systematic literature review", "systematic review", "slr",
                            "survey", "future directions", "se landscape",
                            "deep learning"],
    "competitive programming": ["competitive programming", "massive sampling",
                               "filtering"],
    "human-ai collaboration": ["human-ai collaboration", "collaborative framework",
                               "human-ai pair programming", "collaborative intelligence",
                               "autonomy", "pair programming"],
    "multi-agent system": ["multi-agent", "multi-agent system", "agent",
                          "collaborative agents", "chatdev", "princeton"],
}

# Build reverse mapping: raw keyword -> canonical keyword
KEYWORD_NORMALIZATION = {}
for canonical, raw_list in CANONICAL_KEYWORDS.items():
    for raw in raw_list:
        KEYWORD_NORMALIZATION[raw.lower()] = canonical


def normalize_keyword(kw):
    kw_lower = kw.lower().strip()
    return KEYWORD_NORMALIZATION.get(kw_lower, kw_lower)


def build_graph():
    """Build and return a HeterogeneousGraph from papers.json."""
    with open(PAPERS_FILE, "r") as f:
        papers = json.load(f)

    graph = HeterogeneousGraph()

    # Track unique authors and keywords for ID generation
    author_id_map = {}   # author_name -> "author_XX"
    keyword_id_map = {}  # normalized_kw -> "kw_XX"
    author_counter = 0
    keyword_counter = 0

    # Papers organized by domain for sequence edge detection
    domain_papers = defaultdict(list)

    # === Step 1: Create paper nodes ===
    for paper in papers:
        pid = paper["id"]
        node_type = paper.get("node_type", "paper")
        props = {
            "title": paper["title"],
            "year": paper["year"],
            "domain": paper["domain"],
            "source": paper.get("source", ""),
            "arxiv_id": paper.get("arxiv_id", ""),
            "confidence": paper.get("confidence", "unverified"),
            "abstract": paper.get("abstract", ""),
            "note": paper.get("note", ""),
        }
        graph.add_node(pid, node_type, props)
        domain_papers[paper["domain"]].append(paper)

    # === Step 2: Create author nodes and author_of edges ===
    for paper in papers:
        pid = paper["id"]
        for author_name in paper.get("authors", []):
            # Clean "et al." entries
            if author_name == "et al." or not author_name.strip():
                continue
            if author_name not in author_id_map:
                author_id = f"author_{author_counter}"
                author_id_map[author_name] = author_id
                author_counter += 1
                graph.add_node(author_id, "author", {"name": author_name})
            aid = author_id_map[author_name]
            # Weight: 1 / num_papers_by_author (dilute prolific authors)
            weight = 1.0
            graph.add_edge(aid, pid, "author_of", weight=weight,
                           properties={"author_name": author_name})

    # === Step 3: Create keyword nodes and topic/method edges ===
    # Use canonical keywords only; raw keywords that don't map to canonical ones are dropped
    seen_keywords = set()
    for paper in papers:
        pid = paper["id"]
        raw_keywords = paper.get("keywords", [])
        canonical_kws = set()
        for kw in raw_keywords:
            norm_kw = normalize_keyword(kw)
            if norm_kw:  # Only keep keywords that map to canonical categories
                canonical_kws.add(norm_kw)

        # Also add domain-derived keywords for unverified papers with few keywords
        domain_kw_map = {
            "A": "ai-assisted programming",
            "B": "large language model",
            "C": "developer experience",
            "D": "automated program repair",
            "E": "code security",
        }
        domain_kw = domain_kw_map.get(paper["domain"])
        if domain_kw:
            canonical_kws.add(domain_kw)

        for norm_kw in canonical_kws:
            if norm_kw not in keyword_id_map:
                kw_id = f"kw_{keyword_counter}"
                keyword_id_map[norm_kw] = kw_id
                keyword_counter += 1
                graph.add_node(kw_id, "keyword", {"name": norm_kw})
            kw_id = keyword_id_map[norm_kw]

            # topic: keyword relates to paper
            if not (pid in seen_keywords and norm_kw in seen_keywords):
                graph.add_edge(kw_id, pid, "topic", weight=1.0)

            # method: paper proposes/uses this technique
            is_method = norm_kw in ("code generation", "automated program repair",
                                    "prompt engineering", "chain-of-thought",
                                    "retrieval-augmented generation", "self-debugging",
                                    "self-repair", "fault localization", "multi-agent system",
                                    "competitive programming", "code evaluation")
            if is_method:
                graph.add_edge(pid, kw_id, "method", weight=1.0)

        seen_keywords.add(pid)

    # === Step 4: Create cite edges ===
    for paper in papers:
        pid = paper["id"]
        for cited_id in paper.get("citations", []):
            if graph.has_node(cited_id):
                # Weight: 1.0 (can normalize by citation count later)
                graph.add_edge(pid, cited_id, "cite", weight=1.0)

    # === Step 5: Create sequence edges ===
    # Papers in same domain sorted by year; consecutive papers in same research line
    known_sequences = [
        ("B5", "B6"),   # DeepSeek-Coder -> DeepSeek-Coder-V2
    ]
    for source_id, target_id in known_sequences:
        if graph.has_node(source_id) and graph.has_node(target_id):
            source_year = graph.get_node(source_id).properties.get("year", 0)
            target_year = graph.get_node(target_id).properties.get("year", 0)
            year_gap = abs(target_year - source_year)
            weight = 1.0 / (year_gap + 1)
            graph.add_edge(source_id, target_id, "sequence", weight=weight)

    # Auto-detect: same domain, consecutive years, similar title keywords
    for domain, papers_list in domain_papers.items():
        sorted_papers = sorted(papers_list, key=lambda p: p["year"])
        for i in range(len(sorted_papers) - 1):
            p1 = sorted_papers[i]
            p2 = sorted_papers[i + 1]
            if p1["year"] == p2["year"] - 1:
                # Check title similarity for research lineage
                words1 = set(p1["title"].lower().split())
                words2 = set(p2["title"].lower().split())
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                if overlap >= 0.3 and not graph.has_edge(p1["id"], p2["id"], "sequence"):
                    weight = 1.0
                    graph.add_edge(p1["id"], p2["id"], "sequence", weight=weight)

    # === Step 6: Add citation edges based on domain knowledge ===
    # Known citation relationships from web search results
    known_citations = [
        # SWE-bench cites HumanEval (D1 cites B1)
        ("D1", "B1"),
        # AlphaCode cites Codex/HumanEval (B2 cites B1)
        ("B2", "B1"),
        # StarCoder2 cites Code Llama (B4 cites B3)
        ("B4", "B3"),
        # DeepSeek-Coder-V2 cites DeepSeek-Coder (B6 cites B5)
        ("B6", "B5"),
        # WizardCoder cites StarCoder, Code Llama (B7 cites B3, B4)
        ("B7", "B4"),
        # Magicoder cites StarCoder (B8 cites B4)
        ("B8", "B4"),
        # Code Llama cites Llama 2 / Codex (B3 cites B1)
        ("B3", "B1"),
        # DeepSeek-Coder cites StarCoder, Code Llama (B5 cites B3, B4)
        ("B5", "B4"),
        # Self-Debug cites Codex (D6 cites B1)
        ("D6", "B1"),
        # Self-Repair cites Self-Debug (D7 cites D6)
        ("D7", "D6"),
        # RepairAgent cites Self-Debug, SWE-bench (D3 cites D6, D1)
        ("D3", "D6"),
        ("D3", "D1"),
        # SWE-agent cites SWE-bench (D2 cites D1)
        ("D2", "D1"),
        # Perry insecure cites HumanEval (C10 cites B1)
        ("C10", "B1"),
        # How Secure ChatGPT cites HumanEval (E3 cites B1)
        ("E3", "B1"),
        # LiveCodeBench cites HumanEval (E7 cites B1)
        ("E7", "B1"),
        # Peng productivity cites Copilot studies (C2 cites C1, C3)
        ("C2", "C3"),
        # LLM4SE cites various code LLMs (A5 cites B1, B3, B4)
        ("A5", "B1"),
        ("A5", "B3"),
        ("A5", "B4"),
    ]

    for source_id, target_id in known_citations:
        if graph.has_node(source_id) and graph.has_node(target_id):
            if not graph.has_edge(source_id, target_id, "cite"):
                graph.add_edge(source_id, target_id, "cite", weight=1.0)

    # Update cited_by (reverse of cite)
    for edge in graph.get_edges_by_type("cite"):
        target = edge.target
        if not any(e.source == edge.source and e.type == "cited_by"
                   for e in graph._adjacency.get(target, [])):
            # cited_by is reverse direction of cite
            graph.add_edge(edge.target, edge.source, "cited_by", weight=edge.weight)

    print(graph.summary())
    return graph


if __name__ == "__main__":
    graph = build_graph()
    print("\nGraph built successfully!")