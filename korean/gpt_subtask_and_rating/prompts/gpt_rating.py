#!/usr/bin/env python3
"""
ambistory_zero_shot.py — Run GPT zero-shot plausibility scoring
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pickle
try:
    with open("caches/rate.pkl", "rb") as f:
        print('load cache')
            
        CACHE = pickle.load(f)
except FileNotFoundError:
        print('making new cache')
        CACHE = {}
        
def save_cache():    
        with open("caches/rate.pkl", "wb") as f:
            pickle.dump(CACHE, f)
            
client = AsyncOpenAI()

# ================================================================
# ZERO-SHOT PROMPT
# ================================================================

AMBISTORY_PROMPT = """You will see a short text in which one sentence is marked with "**".
That sentence contains a HOMONYM, a word that can have multiple meanings.

Your task: Rate how plausible the GIVEN MEANING of that homonym is IN CONTEXT.

Use this scale:
1 = Not plausible at all
2 = Weak plausibility
3 = Ambiguous / both meanings similarly plausible
4 = Mostly plausible
5 = The only plausible meaning

Return ONLY the number 1, 2, 3, 4, or 5.

HOMONYM:
{homonym}

TEXT:
Precontext: {precontext}
**{sentence}**
Ending: {ending}

Meaning being judged: "{meaning}"

Return ONLY a single digit (1–5). No words.
"""

# ================================================================
# PARSE MODEL OUTPUT
# ================================================================

def parse_score(s):
    if not s:
        return None
    s = s.strip()
    if s in {"1", "2", "3", "4", "5"}:
        return int(s)
    return None

# ================================================================

async def call_model_async(semaphore, model, prompt):
    async with semaphore:
        resp = await client.responses.create(
            model=model,
            input=prompt,
            # temperature=0,
        )
        return (resp.output_text or "").strip()

# ================================================================

async def process_item_async(item_id, item, semaphore, model):

    prompt = AMBISTORY_PROMPT.format(
        precontext=item.get("precontext", ""),
        sentence=item.get("sentence", ""),
        ending=item.get("ending", "") or "",
        homonym=item.get("homonym", "(unknown)"),
        meaning=item.get("judged_meaning", "(unknown meaning)"),
    )

    raw = await call_model_async(semaphore, model, prompt)
    parsed = parse_score(raw)

    return {
        "id": item_id,
        "homonym": item.get("homonym"),
        "judged_meaning": item.get("judged_meaning"),
        "model_raw_output": raw,
        "model_score": parsed,
        "api_error": None if parsed is not None else "parse_error"
    }

# ================================================================

async def main_async(args):
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())
    print("Loaded", len(items), "items.")

    semaphore = asyncio.Semaphore(args.concurrency)

    results = await tqdm_asyncio.gather(
        *[
            process_item_async(item_id, item, semaphore, args.model)
            for item_id, item in items
        ],
        desc="AmbiStory ZERO-SHOT"
    )

    parsed = sum(1 for r in results if r["model_score"] is not None)
    failed = len(results) - parsed

    print("\n======== SUMMARY ========")
    print("Total items:", len(results))
    print("Parsed:", parsed)
    print("Failed:", failed)
    print("=========================\n")

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
