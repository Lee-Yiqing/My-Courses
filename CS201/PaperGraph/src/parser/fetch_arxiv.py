"""Fetch full metadata for papers with known arXiv IDs via arXiv API."""

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ARXIV_BASE = "http://export.arxiv.org/api/query"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAPERS_FILE = DATA_DIR / "papers.json"

DELAY = 3  # seconds between API calls


def load_papers():
    with open(PAPERS_FILE) as f:
        return json.load(f)


def save_papers(papers):
    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)


def fetch_arxiv(arxiv_id):
    url = f"{ARXIV_BASE}?id_list={arxiv_id}&max_results=1"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_data = resp.read().decode("utf-8")
        root = ET.fromstring(xml_data)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")
        if entry is None:
            print(f"  No entry found for {arxiv_id}")
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

        print(f"  Found: {len(result.get('authors', []))} authors, abstract length={len(result.get('abstract', ''))}")
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    papers = load_papers()

    # Papers with confirmed arXiv IDs that need full metadata
    target_ids = [p["arxiv_id"] for p in papers if p.get("arxiv_id") and p.get("confidence") != "verified" or not p.get("abstract")]

    # Also fetch verified papers that lack abstracts
    for paper in papers:
        pid = paper["id"]
        arxiv_id = paper.get("arxiv_id", "")

        if not arxiv_id:
            continue

        # Skip if already has abstract and full authors
        if paper.get("abstract") and len(paper.get("authors", [])) > 5 and paper.get("confidence") == "verified":
            print(f"[{pid}] Already complete, skipping")
            continue

        print(f"\n[{pid}] Fetching arXiv: {arxiv_id}")
        arxiv_data = fetch_arxiv(arxiv_id)
        time.sleep(DELAY)

        if arxiv_data:
            # Update authors
            if arxiv_data.get("authors") and len(arxiv_data["authors"]) > len(paper.get("authors", [])):
                paper["authors"] = arxiv_data["authors"]

            # Update abstract
            if arxiv_data.get("abstract"):
                paper["abstract"] = arxiv_data["abstract"]

            # Update title (use arXiv canonical title)
            if arxiv_data.get("title"):
                paper["title"] = arxiv_data["title"]

            # Update confidence
            if paper.get("authors") and paper.get("year") and paper.get("abstract"):
                paper["confidence"] = "verified"

            print(f"  Updated {pid}: confidence={paper['confidence']}, authors={len(paper.get('authors', []))}")

    save_papers(papers)

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