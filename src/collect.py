import json
import os
import time
from datetime import datetime
from pathlib import Path
import arxiv
import pandas as pd
from tqdm import tqdm

# Search scope configuration
KEYWORDS = [
    "flutter",
    "react native",
    "cross-platform",
    "mobile app",
    "mobile application",
    "mobile development",
    "mobile software",
    "dart programming",
    "android",
    "ios",
]

CATEGORIES = ["cs.SE", "cs.HC", "cs.PL", "cs.CY", "cs.CR"]
YEAR_FROM = 2015
YEAR_TO = 2026
TARGET_SIZE = 3000
PAGE_SIZE = 50

# Output paths
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUTPUT_DIR / "arxiv_raw.jsonl"
CORPUS_PATH = OUTPUT_DIR / "corpus.jsonl"

def build_query(keywords, categories):
    kw_part = " OR ".join([f'all:"{k}"' for k in keywords]) if keywords else ""
    cat_part = " OR ".join([f"cat:{c}" for c in categories]) if categories else ""
    parts = [p for p in [f"({kw_part})" if kw_part else "",
                          f"({cat_part})" if cat_part else ""] if p]
    return " AND ".join(parts) if parts else "all:*"

def already_collected_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["arxiv_id"])
            except Exception:
                continue
    return ids

def collect_arxiv(query, target_size, page_size, year_from, year_to, out_path: Path):
    client = arxiv.Client(page_size=page_size, delay_seconds=5, num_retries=8)
    seen = already_collected_ids(out_path)
    print(f"Already collected: {len(seen)} articles.")

    saved_this_run = 0
    offset = 0
    outer_attempt = 0
    max_outer_attempts = 6
    initial_backoff_seconds = 60

    while len(seen) < target_size and outer_attempt < max_outer_attempts:
        try:
            search = arxiv.Search(
                query=query,
                max_results=target_size * 3,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            print(f"\nStarting/resuming search from offset={offset} (target={target_size}).")

            with open(out_path, "a", encoding="utf-8") as f:
                pbar = tqdm(initial=len(seen), total=target_size, desc="Collecting")
                for result in client.results(search, offset=offset):
                    offset += 1
                    year = result.published.year if result.published else None
                    if year_from is not None and (year is None or year < year_from):
                        continue
                    if year_to is not None and (year is None or year > year_to):
                        continue

                    arxiv_id = result.get_short_id().split("v")[0]
                    if arxiv_id in seen:
                        continue

                    record = {
                        "arxiv_id": arxiv_id,
                        "title": (result.title or "").strip(),
                        "abstract": (result.summary or "").strip().replace("\n", " "),
                        "authors": [a.name for a in result.authors],
                        "categories": list(result.categories or []),
                        "primary_category": result.primary_category,
                        "published": result.published.isoformat() if result.published else None,
                        "updated": result.updated.isoformat() if result.updated else None,
                        "doi": result.doi,
                        "pdf_url": result.pdf_url,
                        "entry_id": result.entry_id,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    seen.add(arxiv_id)
                    saved_this_run += 1
                    pbar.update(1)
                    pbar.set_postfix(offset=offset)

                    if len(seen) >= target_size:
                        break
                pbar.close()
            break

        except Exception as e:
            outer_attempt += 1
            wait = min(initial_backoff_seconds * (2 ** (outer_attempt - 1)), 600)
            print(f"\n[Warning] Collection interrupted ({outer_attempt}/{max_outer_attempts}): {type(e).__name__}: {e}")
            print(f"Waiting {wait}s before retry...")
            for _ in range(wait):
                time.sleep(1)

    print(f"\nFinished collection. Total collected in {out_path}: {len(seen)} articles.")
    return len(seen)

def process_and_clean():
    if not RAW_PATH.exists():
        print(f"Error: {RAW_PATH} does not exist.")
        return

    raw = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    
    print("Raw records read:", len(raw))
    df = pd.DataFrame(raw)
    
    # Deduplicate by arxiv_id, keep most recent updated date
    df["updated_dt"] = pd.to_datetime(df["updated"], errors="coerce")
    df = df.sort_values("updated_dt").drop_duplicates("arxiv_id", keep="last")
    
    # Remove records without valid title or abstract
    df = df[df["title"].str.len() > 0]
    df = df[df["abstract"].str.len() > 50]
    
    print("After deduplication and cleaning:", len(df))
    
    cols = ["arxiv_id", "title", "abstract", "authors", "categories",
             "primary_category", "published", "doi", "pdf_url"]
    
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for _, row in df[cols].iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            
    print(f"Cleaned corpus saved to: {CORPUS_PATH} ({len(df)} documents).")

if __name__ == "__main__":
    query = build_query(KEYWORDS, CATEGORIES)
    print("Final Query:", query)
    collect_arxiv(
        query=query,
        target_size=TARGET_SIZE,
        page_size=PAGE_SIZE,
        year_from=YEAR_FROM,
        year_to=YEAR_TO,
        out_path=RAW_PATH
    )
    process_and_clean()
