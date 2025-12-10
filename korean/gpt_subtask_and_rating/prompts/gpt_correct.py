#!/usr/bin/env python3
"""
gpt_guaranteed_correct.py — classify whether a judged meaning is GUARANTEED CORRECT
Uses OpenAI Responses API strict output.
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI()

# ================================================================
# STRICT PROMPT
# ================================================================

MEANING_PROMPT = """You MUST answer in exactly one line.

Return ONLY one of:
ANSWER: CORRECT
ANSWER: NOT

Definitions:
CORRECT = The judged meaning is unquestionably correct, guaranteed, fully supported by the context, and has extremely high plausibility (near 100% certainty).
NOT = The judged meaning is only moderately plausible, weakly supported, context-dependent, ambiguous, or NOT guaranteed.

Do NOT output anything except one of the answers above.
Do NOT justify.

CONTEXT:
Precontext: {precontext}
Sentence: {sentence}
Ending: {ending}

JUDGED MEANING:
{meaning}

Is the judged meaning GUARANTEED CORRECT?
"""

# ================================================================
# PARSE ANSWER
# ================================================================

def parse_answer(s):
    if not s:
        return None
    s = s.strip().upper()
    if "ANSWER: CORRECT" in s:
        return "CORRECT"
    if "ANSWER: NOT" in s:
        return "NOT"
    return None

# ================================================================
# API CALL WRAPPER
# ================================================================

async def call_model_async(semaphore, model, prompt):
    async with semaphore:
        resp = await client.responses.create(
            model=model,
            input=prompt,
        )
        out = resp.output_text
        return out.strip() if out else ""

# ================================================================
# PROCESS ONE SAMPLE
# ================================================================

async def process_item_async(item_id, item, semaphore, model):
    prompt = MEANING_PROMPT.format(
        meaning=item["judged_meaning"],
        precontext=item.get("precontext", "(none)"),
        sentence=item.get("sentence", "(none)"),
        ending=item.get("ending", "(none)") or "(none)",
    )

    raw = await call_model_async(semaphore, model, prompt)
    parsed = parse_answer(raw)

    return {
        "id": item_id,
        "judged_meaning": item["judged_meaning"],
        "model_raw_output": raw,
        "model_choice": parsed,
        "guaranteed_correct": (parsed == "CORRECT"),
        "api_error": None if parsed else "parse_error"
    }

# ================================================================
# MAIN
# ================================================================

async def main_async(args):
    # Load dataset
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())
    print(f"Items: {len(items)}")

    semaphore = asyncio.Semaphore(args.concurrency)

    results = await tqdm_asyncio.gather(
        *[
            process_item_async(item_id, item, semaphore, args.model)
            for item_id, item in items
        ],
        desc="GPT-5 GUARANTEED_CORRECT"
    )

    parsed = sum(1 for r in results if r["model_choice"])
    correct = sum(1 for r in results if r["model_choice"] == "CORRECT")
    failed = len(results) - parsed

    print("\n========== SUMMARY ==========")
    print(f"Total items processed:   {len(results)}")
    print(f"Parsed successfully:     {parsed}")
    print(f"Guaranteed correct:      {correct}")
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
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
