"""Download paper PDFs from arXiv for all papers with known arXiv IDs."""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PDF_DIR = Path(__file__).resolve().parent.parent.parent / "pdfs"
PAPERS_FILE = DATA_DIR / "papers.json"

DELAY = 5  # seconds between downloads to avoid rate limiting


def load_papers():
    with open(PAPERS_FILE) as f:
        return json.load(f)


def download_pdf(arxiv_id, output_path):
    """Download PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"  Warning: downloaded file too small ({len(data)} bytes), might not be a valid PDF")
                return False
            with open(output_path, "wb") as f:
                f.write(data)
            print(f"  Downloaded: {len(data)} bytes -> {output_path.name}")
            return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    PDF_DIR.mkdir(exist_ok=True)
    papers = load_papers()

    results = {"success": [], "failed": [], "skipped": []}

    for paper in papers:
        pid = paper["id"]
        arxiv_id = paper.get("arxiv_id", "")

        if not arxiv_id:
            print(f"[{pid}] No arXiv ID, skipping")
            results["skipped"].append(pid)
            continue

        output_path = PDF_DIR / f"{pid}_{arxiv_id}.pdf"

        if output_path.exists() and output_path.stat().st_size > 1000:
            print(f"[{pid}] Already downloaded: {output_path.name}")
            results["success"].append(pid)
            continue

        print(f"[{pid}] Downloading arXiv:{arxiv_id}...")
        success = download_pdf(arxiv_id, output_path)
        time.sleep(DELAY)

        if success:
            results["success"].append(pid)
        else:
            results["failed"].append(pid)

    # Save download log
    with open(DATA_DIR / "download_log.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Download Summary ===")
    print(f"Success: {len(results['success'])} papers")
    print(f"Failed:  {len(results['failed'])} papers")
    print(f"Skipped: {len(results['skipped'])} papers (no arXiv ID)")
    print(f"PDFs saved to: {PDF_DIR}")


if __name__ == "__main__":
    main()