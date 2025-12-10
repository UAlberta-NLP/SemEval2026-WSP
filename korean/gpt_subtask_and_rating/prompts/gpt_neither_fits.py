#!/usr/bin/env python3
"""
gpt_neither_fits.py — Determine whether either judged meaning fits the context.
Outputs:
    ANSWER: OK        → at least one meaning fits → neither_fits = 0
    ANSWER: NEITHER   → neither meaning fits      → neither_fits = 1

Matches your existing structure and Response API usage 1:1.
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ================================================================
# PROMPT
# ================================================================

PAIR_PROMPT = """You MUST answer in exactly one line.

Return ONLY one of:
ANSWER: NEITHER
ANSWER: OK

NEITHER = Neither meaning is plausible in the context.
OK      = At least one meaning is plausible (even if both could fit).

Do NOT explain.
Do NOT choose meaning 1 or meaning 2.
Do NOT output anything else.

MEANING 1:
{meaning1}

MEANING 2:
{meaning2}

CONTEXT:
Precontext: {precontext}
Sentence: {sentence}
Ending: {ending}

Does EITHER meaning fit the context?
"""

# ================================================================
# Load data
# ================================================================

def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# group entries into (pre,sentence,ending) pairs
def group_by_context(data):
    groups = defaultdict(list)
    for key, item in data.items():
        groups[(item.get("precontext",""),
                item.get("sentence",""),
                item.get("ending",""))].append((key,item))
    return groups

# ================================================================
# Parse model output
# ================================================================

def parse_answer(s):
    if not s:
        return None
    s = s.strip().upper()

    if "NEITHER" in s:
        return "NEITHER"

    if "OK" in s:
        return "OK"

    return None

# ================================================================
# Make API call
# ================================================================

async def call_model_async(semaphore, model, prompt):
    async with semaphore:
        resp = await client.responses.create(
            model=model,
            input=prompt,
        )
        return resp.output_text.strip() if resp.output_text else ""

# ================================================================
# Process one pair
# ================================================================

async def process_pair_async(item, semaphore, model):
    prompt = PAIR_PROMPT.format(
        meaning1=item["meaning1"],
        meaning2=item["meaning2"],
        precontext=item["precontext"],
        sentence=item["sentence"],
        ending=item["ending"] or "(none)"
    )

    raw = await call_model_async(semaphore, model, prompt)
    parsed = parse_answer(raw)

    return {
        **item,
        "model_raw_output": raw,
        "model_choice": parsed,
        "neither_fits": 1 if parsed == "NEITHER" else 0,
        "api_error": None if parsed else "parse_error"
    }

# ================================================================
# Main asynchronous driver
# ================================================================

async def main_async(args):
    data = load_samples(Path(args.input_path))
    groups = group_by_context(data)

    # build pair items
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
        desc="GPT-5 NEITHER-FITS"
    )

    # summary
    parsed = sum(1 for r in results if r["model_choice"])
    neither = sum(1 for r in results if r["model_choice"] == "NEITHER")
    failed = len(results) - parsed

    print("\n========== SUMMARY ==========")
    print(f"Total pairs processed:   {len(results)}")
    print(f"Parsed successfully:     {parsed}")
    print(f"NEITHER responses:       {neither}")
    print(f"Failed / empty outputs:  {failed}")
    print("=============================\n")

    # save json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved:", args.out_json)

# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
