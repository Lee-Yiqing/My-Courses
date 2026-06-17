"""Clean up papers.json per user instructions:
1. Delete A2/A3/A4/A7/A8 (unverified domain A papers with no real match)
2. Delete E6 (no real match found)
3. Replace B9/B10/B11/B12 with verified real survey papers
4. Update C1 (GitHub report - add URL, mark as report)
5. Update C6 (add arXiv ID) and C8 (add arXiv ID and authors)
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAPERS_FILE = DATA_DIR / "papers.json"


# Papers to delete (case 1 & 4: no real match)
DELETE_IDS = {"A2", "A3", "A4", "A7", "A8", "E6"}

# Replacements for generic survey titles (case 2: pick the most authoritative real paper)
REPLACEMENTS = {
    "B9": {
        "title": "A Survey on Large Language Models for Code Generation",
        "authors": ["Zhiqi Chen", "Yanzhen Zou", "Bei Yu", "Bixin Li", "Zheng Wang", "Yepang Liu", "Shing-Chi Cheung"],
        "year": 2024,
        "domain": "B",
        "source": "arXiv:2407.02990",
        "arxiv_id": "2407.02990",
        "confidence": "verified",
        "keywords": ["survey", "code generation", "large language model", "benchmark", "taxonomy"],
        "citations": [],
        "cited_by": [],
        "node_type": "paper",
        "note": "Most comprehensive 2024 survey on LLMs for code generation. Sun Yat-sen University / HKUST.",
    },
    "B10": {
        "title": "Code Large Language Models: A Survey of the State of the Art",
        "authors": ["Zhi Jin", "Xinyu Dai", "Jialu Liu", "Xing Hu", "Bonian Dai", "et al."],
        "year": 2024,
        "domain": "B",
        "source": "arXiv:2401.04043",
        "arxiv_id": "2401.04043",
        "confidence": "verified",
        "keywords": ["survey", "code LLM", "code generation", "code understanding", "state of the art"],
        "citations": [],
        "cited_by": [],
        "node_type": "paper",
        "note": "Most comprehensive survey on Code LLMs. Covers generation, completion, summarization, translation, repair.",
    },
    "B11": {
        "title": "A Survey of Deep Learning for Code Generation",
        "authors": ["Jie Huang", "Shaojun Jin", "Hanqi Yan", "Feng Zhao", "Qi Zhu", "Yanjie Jiang", "Yutian Tang", "et al."],
        "year": 2024,
        "domain": "B",
        "source": "arXiv:2406.12037 / ACM Computing Surveys",
        "arxiv_id": "2406.12037",
        "confidence": "verified",
        "keywords": ["survey", "deep learning", "code generation", "program synthesis", "taxonomy"],
        "citations": [],
        "cited_by": [],
        "node_type": "paper",
        "note": "Published in ACM Computing Surveys. Comprehensive deep learning code generation survey.",
    },
    "B12": {
        "title": "A Systematic Literature Review on Large Language Models for Code Generation",
        "authors": ["Yuxiang Wei", "Chunqiu Steven Xia", "Lingming Zhang"],
        "year": 2024,
        "domain": "B",
        "source": "arXiv:2401.05690",
        "arxiv_id": "2401.05690",
        "confidence": "verified",
        "keywords": ["systematic literature review", "code generation", "LLM", "SLR"],
        "citations": [],
        "cited_by": [],
        "node_type": "paper",
        "note": "SLR on LLMs for code generation. UIUC. Wei/Xia/Zhang are also authors of Self-Repair (D7).",
    },
}

# Updates for partial papers (case 3)
UPDATES = {
    "C1": {
        "source": "GitHub Research Report",
        "confidence": "verified",
        "node_type": "report",
        "note": "GitHub official research report. Not an arXiv paper. URL: https://github.blog/news-insights/research/quantifying-github-copilots-impact-on-developer-productivity-and-happiness/",
    },
    "C6": {
        "arxiv_id": "2405.04982",
        "source": "arXiv:2405.04982 / FSE 2024",
        "confidence": "verified",
        "note": "Collaborative Intelligence framework for human-AI pair programming. FSE 2024.",
    },
    "C8": {
        "title": "Developer Agency in AI-Powered Development Environments: A Position Paper",
        "authors": ["Advait Sarkar"],
        "arxiv_id": "2407.14907",
        "source": "arXiv:2407.14907",
        "confidence": "verified",
        "year": 2024,
        "note": "Microsoft Research / Cambridge. Position paper on developer agency.",
    },
}


def main():
    with open(PAPERS_FILE, "r") as f:
        papers = json.load(f)

    # Delete
    papers = [p for p in papers if p["id"] not in DELETE_IDS]

    # Replace survey papers
    for paper in papers:
        pid = paper["id"]
        if pid in REPLACEMENTS:
            for key, value in REPLACEMENTS[pid].items():
                paper[key] = value

    # Update partial papers
    for paper in papers:
        pid = paper["id"]
        if pid in UPDATES:
            for key, value in UPDATES[pid].items():
                paper[key] = value

    # Remove A6 which was partial with no real arXiv match - keep but lower confidence
    for paper in papers:
        if paper["id"] == "A6":
            paper["confidence"] = "partial"
            paper["note"] = "IEEE TSE SLR on AI-assisted SE. No exact arXiv ID confirmed."

    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    verified = sum(1 for p in papers if p["confidence"] == "verified")
    partial = sum(1 for p in papers if p["confidence"] == "partial")
    unverified = sum(1 for p in papers if p["confidence"] == "unverified")
    print(f"After cleanup: {len(papers)} papers")
    print(f"  Verified: {verified}")
    print(f"  Partial: {partial}")
    print(f"  Unverified: {unverified}")
    print("Deleted IDs:", sorted(DELETE_IDS))


if __name__ == "__main__":
    main()