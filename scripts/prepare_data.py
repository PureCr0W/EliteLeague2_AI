#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare corpus for TDE experiments:
- Load from local JSONL or Hugging Face Datasets (e.g., WikiText)
- Clean & filter lines
- Generate "canaries" (synthetic secrets) and inject them
- Save processed corpus and canary ground-truth files

Outputs:
  data/processed/corpus_clean.jsonl
  data/processed/corpus_with_canary.jsonl
  data/canaries/canary.jsonl
"""

import os
import sys
import json
import uuid
import math
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

# ------------------------------
# I/O helpers
# ------------------------------
def ensure_dir(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True) if Path(p).suffix else Path(p).mkdir(parents=True, exist_ok=True)

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                # allow plain text lines too
                data.append({"text": line})
    return data

def write_jsonl(items: List[Dict], path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# ------------------------------
# Text normalization & filtering
# ------------------------------
def normalize_text(s: str) -> str:
    # minimal normalization to keep semantics
    return " ".join(s.split()).strip()

def clean_corpus(records: List[Dict],
                 min_len: int = 8,
                 drop_headers_like_wikitext: bool = True) -> List[Dict]:
    out = []
    for r in records:
        t = r.get("text", "")
        if not isinstance(t, str):
            t = str(t)
        t = normalize_text(t)
        if not t or len(t) < min_len:
            continue
        if drop_headers_like_wikitext:
            # Drop lines that are typical WikiText section headers (heuristic)
            if t.startswith("=") and t.endswith("="):
                continue
        out.append({"text": t})
    return out

# ------------------------------
# Canary generation & injection
# ------------------------------
def generate_canaries(n: int,
                      template: str = "The secret code is {canary}.",
                      prefix: str = "CANARY",
                      entropy_bits: int = 128,
                      start_id: int = 1) -> List[Dict]:
    """
    Create n synthetic canaries with high-entropy tokens.
    """
    out = []
    for i in range(start_id, start_id + n):
        # UUID4 ~ 122 bits, augment with hex for ~entropy_bits
        token = uuid.uuid4().hex
        while len(token) * 4 < entropy_bits:
            token += uuid.uuid4().hex
        canary_token = f"{prefix}_{token[:entropy_bits // 4]}".upper()
        text = template.format(canary=canary_token)
        out.append({"id": i, "canary": canary_token, "text": text})
    return out

def inject_canaries(corpus: List[Dict],
                    canaries: List[Dict],
                    rate: Optional[float] = None,
                    interleave: bool = True,
                    seed: int = 1337) -> List[Dict]:
    """
    Insert canary lines into the corpus.
    - If 'rate' provided, at most floor(len(corpus) * rate) canaries are interleaved.
      Otherwise, insert all provided canaries.
    - If 'interleave' True, distribute roughly evenly; else append at end.
    """
    random.seed(seed)
    c_out = corpus.copy()

    n_total = len(corpus)
    if rate is not None:
        k = min(len(canaries), max(1, math.floor(n_total * rate)))
        cans = canaries[:k]
    else:
        cans = canaries

    if not cans:
        return c_out

    if not interleave:
        for c in cans:
            c_out.append({"text": c["text"]})
        return c_out

    # Interleave by inserting near uniformly spaced positions
    step = max(1, n_total // (len(cans) + 1))
    positions = list(range(step, step * (len(cans) + 1), step))
    random.shuffle(positions)
    positions = positions[:len(cans)]
    positions.sort()

    # Build new list with insertions
    out = []
    ci = 0
    next_pos = positions[ci] if positions else None
    for idx, rec in enumerate(corpus, start=1):
        out.append(rec)
        if next_pos is not None and idx == next_pos:
            out.append({"text": cans[ci]["text"]})
            ci += 1
            next_pos = positions[ci] if ci < len(positions) else None

    # If anything remains (edge-case), append
    while ci < len(cans):
        out.append({"text": cans[ci]["text"]})
        ci += 1

    return out

# ------------------------------
# Load from Hugging Face Datasets
# ------------------------------
def load_hf_dataset(name: str, subset: Optional[str], split: str, limit: Optional[int]) -> List[Dict]:
    try:
        from datasets import load_dataset
    except Exception as e:
        print("[ERROR] 'datasets' is not installed. Run: pip install datasets", file=sys.stderr)
        raise e

    ds = load_dataset(name, subset) if subset else load_dataset(name)
    if split not in ds:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(ds.keys())}")
    rows = []
    cnt = 0
    for r in ds[split]:
        # Many HF text datasets use 'text' field
        t = r.get("text")
        if t is None:
            # Try common alternates
            for k in ("content", "document", "body", "data"):
                if k in r:
                    t = r[k]
                    break
        if t is None:
            continue
        rows.append({"text": t})
        cnt += 1
        if limit and cnt >= limit:
            break
    return rows

# ------------------------------
# Main
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Prepare corpus and inject canaries.")
    # Input options
    p.add_argument("--in", dest="in_path", type=str, default=None,
                   help="Path to local JSONL (each line JSON with 'text').")
    p.add_argument("--dataset", type=str, default=None,
                   help="HF dataset name, e.g., 'wikitext'")
    p.add_argument("--subset", type=str, default=None,
                   help="HF dataset subset, e.g., 'wikitext-2-raw-v1'")
    p.add_argument("--split", type=str, default="train",
                   help="HF dataset split (train/validation/test)")
    p.add_argument("--max_rows", type=int, default=None,
                   help="Limit number of rows loaded (for quick tests).")

    # Cleaning
    p.add_argument("--min_len", type=int, default=8, help="Min char length to keep a line.")
    p.add_argument("--keep_headers", action="store_true",
                   help="If set, keep WikiText-like header lines (e.g., '= Section =').")

    # Canary generation & injection
    p.add_argument("--generate_canaries", type=int, default=0,
                   help="Number of canaries to generate.")
    p.add_argument("--canary_rate", type=float, default=None,
                   help="Fraction of corpus lines to (at most) interleave with canaries (e.g., 0.005).")
    p.add_argument("--canary_template", type=str, default="The secret code is {canary}.",
                   help="Template with '{canary}' placeholder.")
    p.add_argument("--canary_prefix", type=str, default="CANARY",
                   help="Prefix for canary token.")
    p.add_argument("--entropy_bits", type=int, default=128,
                   help="Approx desired bits of entropy in canary token.")
    p.add_argument("--seed", type=int, default=1337, help="Random seed.")

    # Output paths
    p.add_argument("--out_clean", type=str, default="data/processed/corpus_clean.jsonl",
                   help="Path to save cleaned corpus.")
    p.add_argument("--out", type=str, default="data/processed/corpus_with_canary.jsonl",
                   help="Path to save corpus with canaries.")
    p.add_argument("--canary_out", type=str, default="data/canaries/canary.jsonl",
                   help="Path to save generated canaries ground-truth.")

    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # 1) Load
    if args.in_path:
        print(f"[INFO] Loading local JSONL: {args.in_path}")
        raw = read_jsonl(args.in_path)
    elif args.dataset:
        print(f"[INFO] Loading HF dataset: {args.dataset} ({args.subset}), split={args.split}")
        raw = load_hf_dataset(args.dataset, args.subset, args.split, args.max_rows)
    else:
        print("[ERROR] Provide either --in or --dataset", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(raw)} raw lines")

    # 2) Clean
    cleaned = clean_corpus(
        raw,
        min_len=args.min_len,
        drop_headers_like_wikitext=(not args.keep_headers),
    )
    print(f"[INFO] Cleaned corpus size: {len(cleaned)}")

    # Save clean
    write_jsonl(cleaned, args.out_clean)
    print(f"[OK] Saved clean corpus → {args.out_clean}")

    # 3) Canary generation
    canaries = []
    if args.generate_canaries > 0:
        canaries = generate_canaries(
            n=args.generate_canaries,
            template=args.canary_template,
            prefix=args.canary_prefix,
            entropy_bits=args.entropy_bits,
            start_id=1,
        )
        # Save ground-truth
        # Note: store only id+canary (and text for convenience)
        write_jsonl(canaries, args.canary_out)
        print(f"[OK] Generated {len(canaries)} canaries → {args.canary_out}")

    # 4) Injection
    if canaries:
        processed = inject_canaries(
            cleaned,
            canaries,
            rate=args.canary_rate,
            interleave=True,
            seed=args.seed,
        )
        print(f"[INFO] Corpus after canary injection: {len(processed)}")
    else:
        processed = cleaned

    # 5) Save final
    write_jsonl(processed, args.out)
    print(f"[OK] Saved processed corpus → {args.out}")

    # Summary
    print("========== SUMMARY ==========")
    print(f"Raw lines:         {len(raw)}")
    print(f"Cleaned lines:     {len(cleaned)}")
    print(f"Canaries created:  {len(canaries)}")
    print(f"Final lines:       {len(processed)}")
    if args.canary_rate is not None and canaries:
        print(f"Target inject rate: {args.canary_rate} (max)")
    print("================================")

if __name__ == "__main__":
    main()
