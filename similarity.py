"""Product similarity search over the Amazon-fashion JSONL dataset.

Two feature spaces:
  - Text:    sentence-transformer embedding of a curated text blob.
  - Numeric: z-scored numeric/categorical vector (price, weight, rating,
             ranks, reviews, flags, brand & child-category one-hots).

Final similarity = w_text * cos(text) + w_num * cos(numeric).
FAISS IndexFlatIP retrieves a text-based candidate pool; the combined
score then re-ranks within the pool.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).parent / "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
CACHE_DIR = Path(__file__).parent / ".sim_cache"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ---------- field parsers ----------

_REVIEW_COUNT_RE = re.compile(r"(\d+)\s+customer\s+review", re.IGNORECASE)
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|_")
_WEIGHT_RE = re.compile(r"([\d.]+)\s*(kg|g|ml|l)?\b", re.IGNORECASE)


def _to_float(x) -> float:
    if x is None:
        return np.nan
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, TypeError):
        return np.nan


def _parse_weight_grams(x, fallback=None) -> float:
    """Parse '86.2 g', '1.2 kg', or the '999999999' sentinel into grams."""
    for candidate in (x, fallback):
        if candidate is None:
            continue
        s = str(candidate).strip()
        if not s:
            continue
        m = _WEIGHT_RE.match(s)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        if unit == "kg" or unit == "l":
            val *= 1000.0
        if val > 1e8:  # sentinel
            continue
        return val
    return np.nan


def _parse_rank(d) -> float:
    """`{cat: '#1,59,062'}` -> 159062.0. Returns NaN if absent/unparseable."""
    if not isinstance(d, dict) or not d:
        return np.nan
    val = next(iter(d.values()), None)
    if not val:
        return np.nan
    digits = str(val).replace("#", "").replace(",", "").strip()
    try:
        return float(digits)
    except ValueError:
        return np.nan


def _category_key(d) -> str:
    if not isinstance(d, dict) or not d:
        return "_unknown"
    return next(iter(d.keys()))


def _human_category(key: str) -> str:
    if not key or key == "_unknown":
        return ""
    return _CAMEL_SPLIT_RE.sub(" ", key).strip()


def _parse_review_count(details, fallback) -> float:
    if isinstance(details, dict):
        text = details.get("Customer_Reviews", "")
        if isinstance(text, str):
            if "be the first" in text.lower():
                return 0.0
            m = _REVIEW_COUNT_RE.search(text)
            if m:
                return float(m.group(1))
    return _to_float(fallback)


def _parse_discount(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).replace("%", "").strip()
    if not s or s.lower() == "nan":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _yn(x) -> float:
    return 1.0 if isinstance(x, str) and x.strip().upper() == "Y" else 0.0


def _build_text(row) -> str:
    parts: list[str] = []
    if row.get("product_name"):
        parts.append(str(row["product_name"]))
    brand = row.get("brand")
    if brand and brand != "unknown":
        parts.append(f"brand: {brand}")
    cat_human = row.get("child_category_human")
    if cat_human:
        parts.append(f"category: {cat_human}")
    colour = row.get("colour")
    if isinstance(colour, str) and colour.strip():
        parts.append(f"colour: {colour.strip().lower()}")
    if row.get("meta_keywords"):
        parts.append(str(row["meta_keywords"]))
    details = row.get("product_details__k_v_pairs")
    if isinstance(details, dict):
        material = details.get("Material")
        if material:
            parts.append(f"material: {material}")
    return " | ".join(parts)


# ---------- data loading ----------

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    df["sales_price"] = df["sales_price"].apply(_to_float)
    item_weight = df["product_details__k_v_pairs"].apply(
        lambda d: d.get("Item_Weight") if isinstance(d, dict) else None
    )
    df["weight"] = [_parse_weight_grams(w, fb) for w, fb in zip(df["weight"], item_weight)]
    df["rating"] = df["rating"].apply(_to_float)
    df["discount_pct"] = df.get("discount_percentage", pd.Series(index=df.index)).apply(_parse_discount)

    df["parent_rank"] = df["sales_rank_in_parent_category"].apply(_parse_rank)
    df["child_rank"] = df["sales_rank_in_child_category"].apply(_parse_rank)
    df["child_category"] = df["sales_rank_in_child_category"].apply(_category_key)
    df["child_category_human"] = df["child_category"].apply(_human_category)

    fallback = df.get("no__of_reviews", pd.Series(index=df.index))
    df["review_count"] = [
        _parse_review_count(d, fb) for d, fb in zip(df["product_details__k_v_pairs"], fallback)
    ]

    df["amazon_prime"] = df["amazon_prime__y_or_n"].apply(_yn)
    df["best_seller"] = df["best_seller_tag__y_or_n"].apply(_yn)

    df["brand"] = df["brand"].fillna("unknown").astype(str).str.lower().str.strip().replace("", "unknown")
    df["colour"] = df.get("colour", pd.Series(index=df.index)).fillna("").astype(str)

    df["text_blob"] = df.apply(_build_text, axis=1)
    return df.reset_index(drop=True)


# ---------- index ----------

class SimilarityIndex:
    NUMERIC_COLS = [
        "sales_price", "weight", "rating", "review_count",
        "parent_rank", "child_rank", "discount_pct",
        "amazon_prime", "best_seller",
    ]
    LOG_COLS = {"sales_price", "weight", "review_count", "parent_rank", "child_rank"}

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = MODEL_NAME,
        text_weight: float = 0.75,
        numeric_weight: float = 0.25,
        top_brands: int = 200,
        top_categories: int = 100,
    ):
        self.df = df
        self.id_to_idx = {uid: i for i, uid in enumerate(df["uniq_id"].tolist())}
        self.text_weight = text_weight
        self.numeric_weight = numeric_weight
        self.model_name = model_name
        self.top_brands = top_brands
        self.top_categories = top_categories
        self._build()

    def _numeric_matrix(self) -> np.ndarray:
        num = pd.DataFrame(index=self.df.index)
        for c in self.NUMERIC_COLS:
            col = self.df[c].astype(float)
            if c in self.LOG_COLS:
                col = np.log1p(col.clip(lower=0))
            num[c] = col
        for c in self.NUMERIC_COLS:
            med = num[c].median()
            num[c] = num[c].fillna(med if pd.notna(med) else 0.0)
        mean = num.mean()
        std = num.std(ddof=0).replace(0, 1.0)
        num = (num - mean) / std

        top_b = self.df["brand"].value_counts().head(self.top_brands).index
        brand = self.df["brand"].where(self.df["brand"].isin(top_b), "_other")
        brand_oh = pd.get_dummies(brand, prefix="b").astype(np.float32)

        top_c = self.df["child_category"].value_counts().head(self.top_categories).index
        cat = self.df["child_category"].where(self.df["child_category"].isin(top_c), "_other")
        cat_oh = pd.get_dummies(cat, prefix="c").astype(np.float32)

        full = pd.concat([num.astype(np.float32), brand_oh, cat_oh], axis=1).to_numpy(np.float32)
        norms = np.linalg.norm(full, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return full / norms

    def _text_matrix(self) -> np.ndarray:
        CACHE_DIR.mkdir(exist_ok=True)
        cache = CACHE_DIR / f"text_emb_{self.model_name.replace('/', '_')}.npy"
        if cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(self.df):
                return emb.astype(np.float32)

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model = SentenceTransformer(self.model_name, device=device)
        emb = model.encode(
            self.df["text_blob"].tolist(),
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(cache, emb)
        return emb

    def _build(self) -> None:
        self.text_emb = self._text_matrix()
        self.num_emb = self._numeric_matrix()
        self.text_index = faiss.IndexFlatIP(self.text_emb.shape[1])
        self.text_index.add(self.text_emb)

    def find_similar(self, product_id: str, num_similar: int = 10, pool_mult: int = 20) -> List[str]:
        if product_id not in self.id_to_idx:
            raise KeyError(product_id)
        idx = self.id_to_idx[product_id]

        pool = max(num_similar * pool_mult, 200)
        text_q = self.text_emb[idx : idx + 1]
        _, cand = self.text_index.search(text_q, pool + 1)
        cand_idx = cand[0]
        cand_idx = cand_idx[cand_idx != idx]

        text_scores = (self.text_emb[cand_idx] @ text_q.T).ravel()
        num_q = self.num_emb[idx : idx + 1]
        num_scores = (self.num_emb[cand_idx] @ num_q.T).ravel()
        combined = self.text_weight * text_scores + self.numeric_weight * num_scores

        rating = self.df["rating"].iloc[cand_idx].fillna(0).to_numpy()
        price = self.df["sales_price"].iloc[cand_idx].fillna(np.inf).to_numpy()
        order = np.lexsort((price, -rating, -combined))
        top = cand_idx[order][:num_similar]
        return self.df["uniq_id"].iloc[top].tolist()


# ---------- module-level convenience ----------

_index: Optional[SimilarityIndex] = None


def get_index() -> SimilarityIndex:
    global _index
    if _index is None:
        _index = SimilarityIndex(load_data())
    return _index


def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    return get_index().find_similar(product_id, num_similar)


if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "26d41bdc1495de290bc8e6062d927729"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    results = find_similar_products(pid, n)
    df = get_index().df.set_index("uniq_id")
    q = df.loc[pid]
    print(f"\nQuery: {q['product_name']}")
    print(f"       brand={q['brand']!r}  cat={q['child_category']!r}  "
          f"price={q['sales_price']}  rating={q['rating']}  reviews={q['review_count']}\n")
    for uid in results:
        r = df.loc[uid]
        print(f"  {uid}  brand={r['brand']!r}  cat={r['child_category']!r}  "
              f"price={r['sales_price']}  rating={r['rating']}  reviews={r['review_count']}")
        print(f"      {str(r['product_name'])[:110]}")
