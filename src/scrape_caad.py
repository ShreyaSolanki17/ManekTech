import argparse
import json
import time
from pathlib import Path
from typing import List, Set

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://caad.org.pt/tributario/decisoes/"
USER_AGENT = "Mozilla/5.0 (compatible; LawRAGBot/1.0)"


def fetch(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def normalize_url(href: str) -> str:
    href = href.strip()
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://caad.org.pt" + href
    return BASE_URL + href


def extract_case_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "decisao.php" in href and "id=" in href:
            links.append(normalize_url(href))
    return links


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", id="content")
        or soup.find("div", class_="content")
    )
    if main:
        return main.get_text("\n", strip=True)
    return soup.get_text("\n", strip=True)


def crawl_listing(max_pages: int) -> List[str]:
    visited: Set[str] = set()
    to_visit = [BASE_URL]
    case_links: Set[str] = set()

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        html = fetch(url)

        for link in extract_case_links(html):
            if link.startswith(BASE_URL):
                case_links.add(link)

        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "index.php" in href or href.startswith("?"):
                next_url = normalize_url(href)
                if next_url not in visited:
                    to_visit.append(next_url)

    return sorted(case_links)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default="data/raw/raw_index.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    links = crawl_listing(args.max_pages)[: args.limit]

    with out_path.open("w", encoding="utf-8") as f:
        for url in tqdm(links, desc="Scraping cases"):
            try:
                html = fetch(url)
                text = extract_main_text(html)
                case_id = url.rstrip("/").split("/")[-1]
                record = {
                    "case_id": case_id,
                    "url": url,
                    "raw_text": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                time.sleep(0.5)
            except Exception as exc:
                print(f"Failed to fetch {url}: {exc}")


if __name__ == "__main__":
    main()
