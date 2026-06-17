"""Batch fetch abstracts and full author lists from arXiv API for papers with known arXiv IDs."""

import json
import time
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAPERS_FILE = DATA_DIR / "papers.json"
ARXIV_BASE = "http://export.arxiv.org/api/query"
DELAY = 3  # seconds between requests


def fetch_arxiv(arxiv_id):
    url = f"{ARXIV_BASE}?id_list={arxiv_id}&max_results=1"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
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
        print(f"  Error fetching {arxiv_id}: {e}")
        return None


def main():
    with open(PAPERS_FILE, "r") as f:
        papers = json.load(f)

    papers_with_ids = [p for p in papers if p.get("arxiv_id") and not p.get("abstract")]
    print(f"Fetching abstracts for {len(papers_with_ids)} papers with arXiv IDs but no abstract")

    for paper in papers_with_ids:
        pid = paper["id"]
        arxiv_id = paper["arxiv_id"]
        print(f"[{pid}] Fetching arXiv:{arxiv_id}...")

        data = fetch_arxiv(arxiv_id)
        time.sleep(DELAY)

        if data:
            if data.get("abstract"):
                paper["abstract"] = data["abstract"]
            if data.get("authors") and len(data["authors"]) > len(paper.get("authors", [])):
                # Replace authors only if arXiv has more (more complete list)
                paper["authors"] = data["authors"]
            if data.get("title"):
                # Keep the verified title, but note if arXiv title differs slightly
                pass
            if data.get("categories"):
                # Add CS categories as keywords
                cs_cats = [c.replace("cs.", "") for c in data["categories"] if c.startswith("cs.")]
                existing_kw = paper.get("keywords", [])
                for cat in cs_cats:
                    if cat not in existing_kw:
                        existing_kw.append(cat)
                paper["keywords"] = existing_kw
            print(f"  Got {len(data.get('authors', []))} authors, {len(data.get('abstract', ''))} chars abstract")
        else:
            print(f"  No data returned")

    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    with_abstract = sum(1 for p in papers if p.get("abstract"))
    print(f"\nPapers with abstracts: {with_abstract}/{len(papers)}")


if __name__ == "__main__":
    main()