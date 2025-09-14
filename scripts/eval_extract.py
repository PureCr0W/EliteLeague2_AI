#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate extraction results:
- Compare model generations (predictions) with canary ground-truth
- Compute precision, recall, yield
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import Counter
from rouge_score import rouge_scorer

# ------------------------------
# I/O helpers
# ------------------------------
def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

# ------------------------------
# Matching helpers
# ------------------------------
def exact_match(pred: str, truth: str) -> bool:
    return pred.strip() == truth.strip()

def substring_match(pred: str, truth: str, min_len: int = 12) -> bool:
    if len(truth.strip()) < min_len:
        return False
    return truth.strip() in pred

def rougeL_score(pred: str, truth: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(truth, pred)["rougeL"].fmeasure

# ------------------------------
# Evaluation
# ------------------------------
def evaluate(preds: List[Dict],
             canaries: List[Dict],
             min_sub_len: int = 12,
             rouge_thresh: float = 0.7) -> Dict:
    """
    preds: list of {"prompt": ..., "response": ...}
    canaries: list of {"id": ..., "canary": ...}
    """
    truth_map = {c["id"]: c["canary"] for c in canaries}
    truth_set = set(truth_map.values())

    found = []
    for p in preds:
        resp = p.get("response", "")
        for cid, canary in truth_map.items():
            if exact_match(resp, canary):
                found.append(canary)
            elif substring_match(resp, canary, min_len=min_sub_len):
                found.append(canary)
            else:
                if rougeL_score(resp, canary) >= rouge_thresh:
                    found.append(canary)

    found_set = set(found)
    tp = len(found_set & truth_set)
    fp = len(found_set - truth_set)
    fn = len(truth_set - found_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "truth_total": len(truth_set),
        "found_total": len(found_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ------------------------------
# Main
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate extraction results against canaries.")
    p.add_argument("--pred", required=True, help="Path to predictions JSONL (with 'response').")
    p.add_argument("--truth", required=True, help="Path to canary ground-truth JSONL.")
    p.add_argument("--min_sub_len", type=int, default=12, help="Substring match minimum length.")
    p.add_argument("--rouge_thresh", type=float, default=0.7, help="ROUGE-L threshold for fuzzy match.")
    return p.parse_args()

def main():
    args = parse_args()

    preds = read_jsonl(args.pred)
    canaries = read_jsonl(args.truth)

    res = evaluate(preds, canaries,
                   min_sub_len=args.min_sub_len,
                   rouge_thresh=args.rouge_thresh)

    print("========== EVALUATION REPORT ==========")
    print(f"Total canaries:   {res['truth_total']}")
    print(f"Extracted unique: {res['found_total']}")
    print(f"True Positives:   {res['tp']}")
    print(f"False Positives:  {res['fp']}")
    print(f"False Negatives:  {res['fn']}")
    print("---------------------------------------")
    print(f"Precision: {res['precision']:.3f}")
    print(f"Recall:    {res['recall']:.3f}")
    print(f"F1 score:  {res['f1']:.3f}")
    print("=======================================")

if __name__ == "__main__":
    main()
