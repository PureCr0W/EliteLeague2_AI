#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TDE attack against target model (e.g., Gemma-3 270M).
- Loads model config & attack config
- Executes attack
- Saves predictions to JSONL
"""

import argparse
import yaml
import os
import json
from pathlib import Path
from datetime import datetime

from tde_lab.attacks.carlini_tde import CarliniTDE
from tde_lab.models.hf_client import HFClient

def ensure_dir(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", required=True, help="Path to model config YAML")
    parser.add_argument("--attack_cfg", required=True, help="Path to attack config YAML")
    parser.add_argument("--out", type=str, default="runs/gemma3_tde/preds.jsonl",
                        help="Path to save predictions JSONL")
    args = parser.parse_args()

    # Load model
    model = HFClient(args.model_cfg)

    # Load attack config
    with open(args.attack_cfg) as f:
        attack_cfg = yaml.safe_load(f)

    # Init attacker
    attacker = CarliniTDE(model, attack_cfg)

    # Example "hints" for attack
    hints = ["secret", "password", "private", "hidden"]

    print(f"[INFO] Running attack with {len(hints)} hints...")
    results = attacker.run(hints)

    # Ensure output directory exists
    ensure_dir(args.out)

    # Save to JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(results)} predictions → {args.out}")
