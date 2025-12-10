#!/usr/bin/env python3
"""
gpt5_pair_eval_async.py — Fast GPT-5 pairwise evaluation using asyncio
AmbiStory evaluator with concurrency, retry logic, and tie support.
"""

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ================================================================
# PROMPT
# ================================================================

PAIR_PROMPT = """You MUST answer in exactly one line.

Return ONLY:
ANSWER: 1
or
ANSWER: 2

Do NOT explain your reasoning. Do NOT output anything else.

HOMONYM: {homonym}

POSSIBLE MEANINGS:
1. {meaning1}
2. {meaning2}

CONTEXT:
Precontext: {precontext}
Sentence: {sentence}
Ending: {ending}

Which meaning (1 or 2) is more plausible?
"""

# ================================================================
# DATA LOADING
# ================================================================

def load_samples(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def group_by_context(data: Dict[str, dict]):
    groups = defaultdict(list)
    for key, item in data.items():
        context_key = (
            item.get("precontext", ""),
            item.get("sentence", ""),
            item.get("ending", ""),
        )
        groups[context_key].append((key, item))
    return groups


# ================================================================
# PARSE ANSWER
# ================================================================

def parse_answer(text: str) -> Optional[int]:
    import re
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        clean = line.replace("**", "").replace("`", "")
        m = re.search(r"answer[^0-9]*([12])", clean, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m2 = re.search(r"\b([12])\b\.?$", clean)
        if m2:
            return int(m2.group(1))
    return None


# ================================================================
# ASYNC GPT-5 CALL WITH RETRIES
# ================================================================

async def call_model_async(semaphore, model, prompt, max_output_tokens, retries=5):
    async with semaphore:
        delay = 1.0
        for attempt in range(retries):
            try:
                resp = await client.responses.create(
                    model=model,
                    instructions=(
                        "You are an expert linguist. "
                        "Follow the user's instructions EXACTLY. "
                        "Return only: ANSWER: 1 or ANSWER: 2"
                    ),
                    input=prompt,
                    reasoning={"effort": "low"},
                    max_output_tokens=max_output_tokens,
                )

                # Extract message
                text = None
                try:
                    for item in resp.output:
                        if getattr(item, "type", None) == "message":
                            if item.content and len(item.content) > 0:
                                text = item.content[0].text
                            break
                except Exception:
                    pass

                if not text:
                    text = getattr(resp, "output_text", "")

                return text.strip()

            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"API failed after retries: {e}")
                await asyncio.sleep(delay)
                delay *= 2


# ================================================================
# PROCESS ONE PAIR
# ================================================================

async def process_pair_async(item, semaphore, model, max_output_tokens):
    prompt = PAIR_PROMPT.format(
        homonym=item["homonym"],
        meaning1=item["meaning1"],
        meaning2=item["meaning2"],
        precontext=item["precontext"],
        sentence=item["sentence"],
        ending=item["ending"] or "(none)",
    )

    raw = None
    ans = None
    human_pref = item["human_pref"]

    try:
        raw = await call_model_async(
            semaphore,
            model=model,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
        ans = parse_answer(raw)
    except Exception as e:
        return {
            **item,
            "human_preferred": human_pref,
            "model_raw_output": raw,
            "model_choice": None,
            "parse_ok": False,
            "correct": None,
            "api_error": str(e),
        }

    # Tie → record but do not score
    if human_pref is None:
        return {
            **item,
            "human_preferred": None,
            "model_raw_output": raw,
            "model_choice": ans,
            "parse_ok": ans in (1, 2),
            "correct": None,
            "api_error": None,
        }

    # Non-tie: normal scoring
    if ans not in (1, 2):
        return {
            **item,
            "human_preferred": human_pref,
            "model_raw_output": raw,
            "model_choice": ans,
            "parse_ok": False,
            "correct": None,
            "api_error": None,
        }

    is_correct = (ans == human_pref)

    return {
        **item,
        "human_preferred": human_pref,
        "model_raw_output": raw,
        "model_choice": ans,
        "parse_ok": True,
        "correct": is_correct,
        "api_error": None,
    }


# ================================================================
# MAIN ASYNC ROUTINE
# ================================================================

async def main_async(args):
    input_path = Path(args.input_path)
    print(f"Loading: {input_path}")

    data = load_samples(input_path)
    print(f"Loaded {len(data)} samples")

    groups = group_by_context(data)
    print(f"Unique contexts: {len(groups)}")

    pair_items = []
    skipped_non_pairs = 0

    # Build pairs including ties
    for (precontext, sentence, ending), items in groups.items():
        if len(items) != 2:
            skipped_non_pairs += 1
            continue
        (k1, i1), (k2, i2) = items
        avg1 = float(i1.get("average", 0.0))
        avg2 = float(i2.get("average", 0.0))
        hom1 = i1.get("homonym", "")
        hom2 = i2.get("homonym", "")

        if hom1 != hom2:
            print(f"[WARN] mismatch: {k1} vs {k2}")
            continue

        if avg1 > avg2:
            human_pref = 1
        elif avg2 > avg1:
            human_pref = 2
        else:
            human_pref = None

        pair_items.append({
            "homonym": hom1,
            "precontext": precontext,
            "sentence": sentence,
            "ending": ending,
            "key1": k1,
            "key2": k2,
            "meaning1": i1.get("judged_meaning", ""),
            "meaning2": i2.get("judged_meaning", ""),
            "avg1": avg1,
            "avg2": avg2,
            "human_pref": human_pref,
        })

    print(f"Pairs (including ties): {len(pair_items)}")
    print(f"Skipped: {skipped_non_pairs}")
    print(f"Using concurrency={args.concurrency}")
    print(f"Using model: {args.model}")

    semaphore = asyncio.Semaphore(args.concurrency)

    # Parallel processing
    results = await tqdm_asyncio.gather(
        *[
            process_pair_async(
                item,
                semaphore,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
            )
            for item in pair_items
        ],
        desc="GPT-5 pair eval",
    )

    # Summaries
    total = sum(1 for r in results if r["human_preferred"] in (1, 2))
    correct = sum(1 for r in results if r["correct"] is True)
    parse_errors = sum(1 for r in results if not r["parse_ok"])

    print("\n===== GPT-5 Pairwise Accuracy =====")
    print(f"Non-tie evaluated pairs: {total}")
    print(f"Correct: {correct}")
    if total:
        print(f"Accuracy: {correct/total:.4f}")
    print(f"Parse errors: {parse_errors}")
    print(f"Skipped (non-pair): {skipped_non_pairs}")

    # Output file
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path}")


# ================================================================
# ENTRY POINT
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max_output_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=10)  # SAFE DEFAULT
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
