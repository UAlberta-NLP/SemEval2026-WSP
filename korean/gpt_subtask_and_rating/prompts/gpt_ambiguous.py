#!/usr/bin/env python3
"""
gpt_ambiguous.py — CLEAR vs AMBIGUOUS classification using GPT-5
Uses *exactly* the official Responses API format:
response = client.responses.create(...); print(response.output_text)
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
import pickle

try:
    with open("caches/ambig.pkl", "rb") as f:
        print('load cache')
            
        CACHE = pickle.load(f)
except FileNotFoundError:
        print('making new cache')
        CACHE = {}
        
        
def save_cache():    
        with open("caches/ambig.pkl", "wb") as f:
            pickle.dump(CACHE, f)
            
client = AsyncOpenAI()

# ================================================================
# PROMPT
# ================================================================

PAIR_PROMPT = """You MUST answer in exactly one line.

Return ONLY one of:
ANSWER: CLEAR
ANSWER: AMBIGUOUS

CLEAR  = One meaning is clearly more plausible.
AMBIGUOUS = Multiple meanings plausible OR insufficient evidence.

Do NOT choose meaning 1 or meaning 2.
Do NOT explain.
Do NOT output anything else.

MEANING 1:
{meaning1}

MEANING 2:
{meaning2}

CONTEXT:
Precontext: {precontext}
Sentence: {sentence}
Ending: {ending}

Is this CLEAR or AMBIGUOUS?
"""

# ================================================================
# Data grouping
# ================================================================

def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_by_context(data):
    groups = defaultdict(list)
    for key, item in data.items():
        groups[(item.get("precontext",""),
                item.get("sentence",""),
                item.get("ending",""))].append((key,item))
    return groups

# ================================================================
# Parse answer
# ================================================================

def parse_answer(s):
    if not s:
        return None
    s = s.strip().upper()
    if "CLEAR" in s:
        return "CLEAR"
    if "AMBIGUOUS" in s:
        return "AMBIGUOUS"
    return None

# ================================================================
# API call — strictly matches OpenAI docs
# ================================================================

async def call_model_async(semaphore, model, prompt):
    async with semaphore:
        resp = await client.responses.create(
            model=model,
            input=prompt,
        )
        # The ONLY reliable field:
        return resp.output_text.strip() if resp.output_text else ""

# ================================================================
# Process pair
# ================================================================

async def process_pair_async(item, semaphore, model):
    prompt = PAIR_PROMPT.format(
        meaning1=item["meaning1"],
        meaning2=item["meaning2"],
        precontext=item["precontext"],
        sentence=item["sentence"],
        ending=item["ending"] or "(none)"
    )

    key = prompt
    if key not in CACHE:
        raw = await call_model_async(semaphore, model, prompt)
        CACHE[key] = parse_answer(raw)
    parsed = CACHE[key]

    save_cache()
    return {
        **item,
        "model_raw_output": raw,
        "model_choice": parsed,
        "ambiguous": (parsed == "AMBIGUOUS"),
        "api_error": None if parsed else "parse_error"
    }

# ================================================================
# Main
# ================================================================

async def main_async(args):
    data = load_samples(Path(args.input_path))
    groups = group_by_context(data)

    pair_items = []
    for (pre,sen,end), items in groups.items():
        if len(items) != 2:
            continue
        (k1,i1),(k2,i2) = items
        pair_items.append({
            "key1": k1,
            "key2": k2,
            "meaning1": i1["judged_meaning"],
            "meaning2": i2["judged_meaning"],
            "precontext": pre,
            "sentence": sen,
            "ending": end,
        })

    print(f"Pairs: {len(pair_items)}")

    semaphore = asyncio.Semaphore(args.concurrency)

    results = await tqdm_asyncio.gather(
        *[
            process_pair_async(item, semaphore, args.model)
            for item in pair_items
        ],
        desc="GPT-5 CLEAR/AMBI"
    )

    # Summary
    parsed = sum(1 for r in results if r["model_choice"])
    ambiguous = sum(1 for r in results if r["model_choice"] == "AMBIGUOUS")
    failed = len(results) - parsed

    print("\n========== SUMMARY ==========")
    print(f"Total pairs processed:   {len(results)}")
    print(f"Parsed successfully:     {parsed}")
    print(f"Ambiguous responses:     {ambiguous}")
    print(f"Failed / empty outputs:  {failed}")
    print("=============================\n")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved:", args.out_json)

# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
