from __future__ import annotations

import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from similarity import get_index

IMAGE_DIR = Path(__file__).parent / "images_medium"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_index()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


def _safe(val):
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _product_images(uniq_id: str) -> list[str]:
    return [f"/images/{p.name}" for p in sorted(IMAGE_DIR.glob(f"{uniq_id}_*.jpg"))]


def _row_to_dict(row, score: Optional[float] = None) -> dict:
    uid = str(row.get("uniq_id", ""))
    d = {
        "uniq_id": uid,
        "product_name": str(row.get("product_name") or ""),
        "brand": str(row.get("brand") or ""),
        "category": str(row.get("child_category_human") or row.get("child_category") or ""),
        "price": _safe(row.get("sales_price")),
        "rating": _safe(row.get("rating")),
        "review_count": _safe(row.get("review_count")),
        "images": _product_images(uid),
    }
    if score is not None:
        d["score"] = round(float(score), 4)
    return d


@app.get("/")
async def root():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1), limit: int = 10):
    idx = get_index()
    df = idx.df
    mask = df["product_name"].str.contains(q, case=False, na=False)
    rows = df[mask].head(limit)
    return [
        {"uniq_id": r["uniq_id"], "product_name": str(r["product_name"]), "brand": str(r["brand"])}
        for _, r in rows.iterrows()
    ]


@app.get("/api/similar")
async def similar(id: str, n: int = 6):
    idx = get_index()
    if id not in idx.id_to_idx:
        raise HTTPException(status_code=404, detail="Product not found")

    df = idx.df.set_index("uniq_id")
    query_row = df.loc[id].to_dict()
    query_row["uniq_id"] = id

    result_ids, scores = idx.find_similar(id, n, return_scores=True)
    results = []
    for uid, score in zip(result_ids, scores):
        row = df.loc[uid].to_dict()
        row["uniq_id"] = uid
        results.append(_row_to_dict(row, score=score))

    return {"query": _row_to_dict(query_row), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
