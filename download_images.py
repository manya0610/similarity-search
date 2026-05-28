import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

INPUT_FILE = Path(__file__).parent / "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
OUTPUT_DIR = Path(__file__).parent / "images_low"
CONCURRENCY = 128
TIMEOUT = 30.0
MAX_RETRIES = 3


def iter_tasks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uniq_id = row.get("uniq_id")
            large = row.get("image_urls__small")
            if not uniq_id or not large:
                continue
            for url in large.split("|"):
                url = url.strip()
                if not url:
                    continue
                image_name = unquote(os.path.basename(urlparse(url).path))
                if not image_name:
                    continue
                if not image_name.lower().endswith(".jpg"):
                    image_name = os.path.splitext(image_name)[0] + ".jpg"
                filename = f"{uniq_id}_{image_name}"
                yield url, OUTPUT_DIR / filename


async def download(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, dest: Path, stats: dict):
    if dest.exists():
        stats["skipped"] += 1
        return
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(resp.content)
                tmp.rename(dest)
                stats["ok"] += 1
                return
            except (httpx.HTTPError, OSError) as e:
                if attempt == MAX_RETRIES:
                    stats["failed"] += 1
                    print(f"FAIL {url}: {e}", file=sys.stderr)
                else:
                    await asyncio.sleep(0.5 * attempt)


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks_iter = list(iter_tasks(INPUT_FILE))
    total = len(tasks_iter)
    print(f"Queued {total} images")

    stats = {"ok": 0, "failed": 0, "skipped": 0}
    sem = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    headers = {"User-Agent": "Mozilla/5.0 (image-fetcher)"}

    async with httpx.AsyncClient(timeout=TIMEOUT, limits=limits, headers=headers, follow_redirects=True, http2=False) as client:
        tasks = [download(client, sem, url, dest, stats) for url, dest in tasks_iter]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 100 == 0:
                print(f"{done}/{total} ok={stats['ok']} fail={stats['failed']} skip={stats['skipped']}")

    print(f"DONE ok={stats['ok']} fail={stats['failed']} skip={stats['skipped']}")


if __name__ == "__main__":
    asyncio.run(main())
