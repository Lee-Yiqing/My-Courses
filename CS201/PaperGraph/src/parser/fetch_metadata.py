"""Fetch and verify paper metadata using Semantic Scholar API and arXiv API.

Principle: NEVER fabricate. Only write confirmed data. Unverifiable papers stay at confidence="unverified".
"""

import json
import time
import re
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path


SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_BASE = "http://export.arxiv.org/api/query"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAPERS_FILE = DATA_DIR / "papers.json"
LOG_FILE = DATA_DIR / "search_log.json"

RATE_LIMIT_DELAY = 1.0  # seconds between API calls


def load_papers():
    with open(PAPERS_FILE, "r") as f:
        return json.load(f)


def save_papers(papers):
    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)


def save_log(log_entries):
    with open(LOG_FILE, "w") as f:
        json.dump(log_entries, f, indent=2, ensure_ascii=False)


def title_similarity(t1, t2):
    """Simple word-overlap similarity for title matching."""
    words1 = set(re.findall(r"\w+", t1.lower()))
    words2 = set(re.findall(r"\w+", t2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def search_semantic_scholar(title, fields="title,authors,year,externalIds,references,citations,abstract"):
    """Search Semantic Scholar by paper title."""
    query = title.replace(" ", "+")
    url = f"{SEMANTIC_SCHOLAR_BASE}?query={query}&limit=5&fields={fields}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", [])
    except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
        return {"error": str(e)}


def fetch_arxiv(arxiv_id):
    """Fetch paper details from arXiv API by ID."""
    url = f"{ARXIV_BASE}?id_list={arxiv_id}&max_results=1"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read().decode("utf-8")
        root = ET.fromstring(xml_data)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")
        if entry is None:
            return None

        result = {}
        title_el = entry.find("{http://www.w3.org/2005/Atom}title")
        if title_el is not None:
            result["title"] = title_el.text.strip().replace("\n", " ")

        abstract_el = entry.find("{http://www.w3.org/2005/Atom}summary")
        if abstract_el is not None:
            result["abstract"] = abstract_el.text.strip().replace("\n", " ")

        authors_els = entry.findall("{http://www.w3.org/2005/Atom}author")
        result["authors"] = []
        for a in authors_els:
            name = a.find("{http://www.w3.org/2005/Atom}name")
            if name is not None:
                result["authors"].append(name.text.strip())

        categories_els = entry.findall("{http://www.w3.org/2005/Atom}category")
        result["categories"] = [c.get("term") for c in categories_els]

        published_el = entry.find("{http://www.w3.org/2005/Atom}published")
        if published_el is not None:
            result["published"] = published_el.text.strip()[:10]

        return result
    except Exception as e:
        return {"error": str(e)}


def match_paper_in_ss_results(paper, ss_results):
    """Find best matching Semantic Scholar result for a paper."""
    if isinstance(ss_results, dict) and "error" in ss_results:
        return None, ss_results

    best_match = None
    best_sim = 0.0
    for result in ss_results:
        ss_title = result.get("title", "")
        sim = title_similarity(paper["title"], ss_title)
        if sim > best_sim and sim >= 0.60:
            best_sim = sim
            best_match = result

    return best_match, {"similarity": best_sim, "ss_title": best_match.get("title", "") if best_match else ""}


def extract_citations_in_dataset(ss_result, all_paper_titles):
    """Match SS references/citations against our dataset papers."""
    dataset_ids = {}
    for p_title, p_id in all_paper_titles:
        dataset_ids[p_title.lower()] = p_id

    citations = []
    refs = ss_result.get("references", []) or []
    for ref in refs:
        ref_title = ref.get("title", "").lower()
        for dt_title, dt_id in all_paper_titles:
            if title_similarity(ref_title, dt_title.lower()) >= 0.85:
                citations.append(dt_id)

    cited_by = []
    cites = ss_result.get("citations", []) or []
    for cite in cites:
        cite_title = cite.get("title", "").lower()
        for dt_title, dt_id in all_paper_titles:
            if title_similarity(cite_title, dt_title.lower()) >= 0.85:
                cited_by.append(dt_id)

    return citations, cited_by


def update_paper_from_ss(paper, ss_match, all_paper_titles):
    """Update paper entry with verified data from Semantic Scholar."""
    updates = {}

    # Authors
    ss_authors = ss_match.get("authors", []) or []
    if ss_authors:
        updates["authors"] = [a.get("name", "") for a in ss_authors if a.get("name")]

    # Year
    ss_year = ss_match.get("year")
    if ss_year:
        updates["year"] = ss_year

    # arXiv ID
    ext_ids = ss_match.get("externalIds", {}) or {}
    if ext_ids.get("ArXiv"):
        updates["arxiv_id"] = ext_ids["ArXiv"]

    # Abstract
    abstract = ss_match.get("abstract", "")
    if abstract:
        updates["abstract"] = abstract

    # Source
    if ext_ids.get("ArXiv"):
        updates["source"] = f"arXiv:{ext_ids['ArXiv']}"
    elif ss_match.get("venue"):
        updates["source"] = ss_match["venue"]

    # Citations within dataset
    citations, cited_by = extract_citations_in_dataset(ss_match, all_paper_titles)
    if citations:
        updates["citations"] = citations
    if cited_by:
        updates["cited_by"] = cited_by

    # Confidence
    has_core_fields = bool(updates.get("authors")) and bool(updates.get("year"))
    updates["confidence"] = "verified" if has_core_fields else "partial"

    return updates


def extract_keywords_from_abstract(abstract, categories=None):
    """Simple keyword extraction from abstract using frequency analysis."""
    if not abstract:
        return []

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would",
                  "could", "should", "may", "might", "shall", "can", "this",
                  "that", "these", "those", "we", "our", "their", "its", "in",
                  "on", "at", "to", "for", "of", "with", "by", "from", "as",
                  "into", "through", "during", "before", "after", "above",
                  "below", "between", "and", "or", "not", "but", "also",
                  "which", "where", "when", "how", "what", "who", "why",
                  "than", "then", "more", "most", "such", "only", "just",
                  "very", "often", "however", "therefore", "although",
                  "furthermore", "moreover", "based", "using", "used",
                  "propose", "present", "show", "demonstrate", "approach",
                  "method", "study", "paper", "result", "results", "task",
                  "model", "models", "system", "systems", "problem",
                  "problems", "work", "new", "novel", "existing",
                  "different", "multiple", "various", "several",
                  "first", "second", "third", "one", "two"}

    words = re.findall(r"[a-zA-Z]{3,}", abstract.lower())
    freq = {}
    for w in words:
        if w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    # Add categories as keywords
    if categories:
        for cat in categories:
            if cat.startswith("cs."):
                sub = cat.replace("cs.", "")
                freq[sub] = freq.get(sub, 0) + 5

    # Top 8 keywords
    sorted_kw = sorted(freq.items(), key=lambda x: -x[1])
    return [kw for kw, count in sorted_kw[:8]]


def main():
    papers = load_papers()
    log_entries = []

    # Build title-to-id mapping for citation matching
    all_paper_titles = [(p["title"], p["id"]) for p in papers]

    for paper in papers:
        pid = paper["id"]
        title = paper["title"]

        # Skip already verified papers with full info
        if paper.get("confidence") == "verified" and paper.get("authors") and paper.get("abstract"):
            print(f"[{pid}] Already verified with full info, skipping")
            continue

        print(f"\n[{pid}] Searching: {title}")

        # Step 1: Semantic Scholar search
        ss_results = search_semantic_scholar(title)
        time.sleep(RATE_LIMIT_DELAY)

        ss_match, match_info = match_paper_in_ss_results(paper, ss_results)

        log_entry = {
            "paper_id": pid,
            "title_query": title,
            "ss_search": match_info,
            "arxiv_fetch": None,
            "updates_applied": {}
        }

        if ss_match:
            print(f"  Match found (sim={match_info['similarity']:.2f}): {ss_match.get('title', '')}")
            updates = update_paper_from_ss(paper, ss_match, all_paper_titles)

            # Step 2: If we got an arXiv ID, fetch full details from arXiv
            arxiv_id = updates.get("arxiv_id", paper.get("arxiv_id", ""))
            if arxiv_id:
                print(f"  Fetching arXiv: {arxiv_id}")
                arxiv_data = fetch_arxiv(arxiv_id)
                time.sleep(RATE_LIMIT_DELAY)

                log_entry["arxiv_fetch"] = {"arxiv_id": arxiv_id}

                if arxiv_data and not isinstance(arxiv_data, dict) or (isinstance(arxiv_data, dict) and "error" not in arxiv_data):
                    if isinstance(arxiv_data, dict) and "error" in arxiv_data:
                        log_entry["arxiv_fetch"]["error"] = arxiv_data["error"]
                    else:
                        # Update authors from arXiv (more complete)
                        if arxiv_data.get("authors"):
                            updates["authors"] = arxiv_data["authors"]

                        # Update abstract from arXiv
                        if arxiv_data.get("abstract"):
                            updates["abstract"] = arxiv_data["abstract"]

                        # Extract keywords from abstract + categories
                        if arxiv_data.get("abstract"):
                            updates["keywords"] = extract_keywords_from_abstract(
                                arxiv_data["abstract"],
                                arxiv_data.get("categories")
                            )

                        log_entry["arxiv_fetch"]["status"] = "success"
                        print(f"  arXiv fetch successful: {len(arxiv_data.get('authors', []))} authors")

            # Apply updates
            for key, value in updates.items():
                if value:  # Only apply non-empty values
                    paper[key] = value
            log_entry["updates_applied"] = updates
            print(f"  Updated: confidence={paper.get('confidence')}, authors={len(paper.get('authors', []))}")

        else:
            print(f"  No match found in Semantic Scholar")
            if isinstance(ss_results, dict) and "error" in ss_results:
                log_entry["ss_search"]["error"] = ss_results["error"]

        log_entries.append(log_entry)

    # Save results
    save_papers(papers)
    save_log(log_entries)

    # Summary
    verified = sum(1 for p in papers if p.get("confidence") == "verified")
    partial = sum(1 for p in papers if p.get("confidence") == "partial")
    unverified = sum(1 for p in papers if p.get("confidence") == "unverified")

    print(f"\n=== Summary ===")
    print(f"Verified: {verified}")
    print(f"Partial: {partial}")
    print(f"Unverified: {unverified}")
    print(f"Total: {len(papers)}")


if __name__ == "__main__":
    main()