"""Convert downloaded PDFs to markdown format for Obsidian vault."""

import json
import re
import fitz  # pymupdf
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PDF_DIR = Path(__file__).resolve().parent.parent.parent / "pdfs"
VAULT_DIR = Path(__file__).resolve().parent.parent.parent / "obsidian_vault"
PAPERS_FILE = DATA_DIR / "papers.json"

DOMAIN_NAMES = {
    "A": "Vibe Coding / Prompt-Driven Development",
    "B": "LLM Code Generation",
    "C": "Human-AI Collaboration",
    "D": "Automated Code Repair",
    "E": "Prompt Engineering / Security",
}


def load_papers():
    with open(PAPERS_FILE) as f:
        return json.load(f)


def pdf_to_markdown(pdf_path):
    """Convert PDF to markdown using pymupdf."""
    try:
        doc = fitz.open(pdf_path)
        md_content = ""

        for page_num in range(min(len(doc), 20)):  # Only first 20 pages
            page = doc[page_num]
            # Extract text blocks
            blocks = page.get_text("blocks", sort=True)
            for block in blocks:
                if block[6] == 0:  # Text block (not image)
                    text = block[4].strip()
                    if text:
                        md_content += text + "\n\n"

        doc.close()
        # Clean up common PDF artifacts
        md_content = re.sub(r'-\n', '', md_content)  # Merge hyphenated lines
        md_content = re.sub(r'\n{3,}', '\n\n', md_content)  # Remove excessive newlines
        md_content = md_content.strip()
        return md_content
    except Exception as e:
        print(f"  Error converting PDF: {e}")
        return ""


def create_obsidian_note(paper, md_content):
    """Create an Obsidian-compatible markdown note for a paper."""
    pid = paper["id"]
    title = paper["title"]
    domain = paper["domain"]
    authors = paper.get("authors", [])
    year = paper.get("year", "")
    keywords = paper.get("keywords", [])
    abstract = paper.get("abstract", "")
    arxiv_id = paper.get("arxiv_id", "")
    confidence = paper.get("confidence", "")
    source = paper.get("source", "")
    domain_name = DOMAIN_NAMES.get(domain, domain)

    # Build wikilinks for keywords
    keyword_links = ""
    if keywords:
        keyword_links = "\n## Keywords\n" + "\n".join(
            f"- [[kw_{kw}|{kw}]]" for kw in keywords
        )

    # Build wikilinks for authors
    author_links = ""
    if authors:
        author_list = []
        for a in authors:
            if a != "et al." and a:
                author_list.append(f"[[author_{a}|{a}]]")
            else:
                author_list.append(a)
        author_links = "\n## Authors\n" + ", ".join(author_list)

    # Domain link
    domain_link = f"[[domain_{domain}|{domain_name}]]"

    # Confidence indicator
    confidence_icon = {"verified": "[x]", "partial": "[/]", "unverified": "[ ]"}
    conf_mark = confidence_icon.get(confidence, "[ ]")

    # Full note content
    note = f"""---
id: {pid}
title: "{title}"
domain: {domain}
year: {year}
arxiv_id: "{arxiv_id}"
confidence: {confidence}
source: "{source}"
node_type: {paper.get('node_type', 'paper')}
---

# {title}

**Domain**: {domain_link} | **Year**: {year} | **Confidence**: {conf_mark} {confidence}

{author_links}

{keyword_links}

## Abstract

{abstract if abstract else "(Abstract not available - see PDF content below)"}

## Paper Content

{md_content if md_content else "(PDF not available for this paper)"}

---

*Source: {source}*
"""

    return note


def main():
    papers = load_papers()
    VAULT_DIR.mkdir(exist_ok=True)

    stats = {"converted": 0, "no_pdf": 0, "empty_content": 0}

    for paper in papers:
        pid = paper["id"]
        arxiv_id = paper.get("arxiv_id", "")
        title = paper["title"]

        print(f"[{pid}] Processing: {title[:50]}...")

        # Find PDF file
        pdf_content = ""
        if arxiv_id:
            pdf_path = PDF_DIR / f"{pid}_{arxiv_id}.pdf"
            if pdf_path.exists():
                pdf_content = pdf_to_markdown(pdf_path)
                if pdf_content:
                    print(f"  Converted PDF: {len(pdf_content)} chars")
                    stats["converted"] += 1
                else:
                    print(f"  PDF conversion yielded empty content")
                    stats["empty_content"] += 1
            else:
                print(f"  PDF not found: {pdf_path.name}")
                stats["no_pdf"] += 1
        else:
            print(f"  No arXiv ID, creating note from metadata only")
            stats["no_pdf"] += 1

        # Create Obsidian note
        note_content = create_obsidian_note(paper, pdf_content)

        # Sanitize filename (replace special chars)
        safe_title = re.sub(r'[\\/:*?"<>|#^[\]]', '_', title)
        safe_title = safe_title[:80]  # Limit filename length
        note_path = VAULT_DIR / f"{pid}_{safe_title}.md"

        with open(note_path, "w", encoding="utf-8") as f:
            f.write(note_content)

        print(f"  Saved: {note_path.name}")

    # Create domain notes
    for domain_code, domain_name in DOMAIN_NAMES.items():
        domain_papers = [p for p in papers if p["domain"] == domain_code]
        domain_note = f"""---
type: domain
---

# {domain_name}

Papers in this domain ({len(domain_papers)} total):

"""
        for p in domain_papers:
            safe_title = re.sub(r'[\\/:*?"<>|#^[\]]', '_', p["title"])[:80]
            domain_note += f"- [[{p['id']}_{safe_title}|{p['id']}: {p['title']}]]\n"

        domain_path = VAULT_DIR / f"domain_{domain_code}.md"
        with open(domain_path, "w", encoding="utf-8") as f:
            f.write(domain_note)
        print(f"Created domain note: {domain_path.name}")

    print(f"\n=== Conversion Summary ===")
    print(f"Converted with PDF content: {stats['converted']}")
    print(f"No PDF available:          {stats['no_pdf']}")
    print(f"Empty PDF content:         {stats['empty_content']}")
    print(f"Vault saved to:            {VAULT_DIR}")
    print(f"Total notes:               {len(list(VAULT_DIR.glob('*.md')))}")


if __name__ == "__main__":
    main()