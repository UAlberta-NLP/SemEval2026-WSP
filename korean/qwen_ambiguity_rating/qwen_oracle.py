#!/usr/bin/env python3
"""
qwen_oracle.py — Qwen3 oracle for judging homonym ambiguity

Given a homonym word in context (precontext, sentence, ending), this script:
1. Provides 3 short bullet points supporting the judgment
2. Outputs "CLEAR" or "AMBIGUOUS"
3. If CLEAR, specifies which meaning is intended

Outputs: outputs_oracle/<key>.json
Requires: transformers >= 4.51.0 and bitsandbytes
"""

import argparse, json, os, sys
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

ORACLE_PROMPT = """You are an expert linguist analyzing whether a homonym (word with multiple meanings) has a CLEAR or AMBIGUOUS meaning in a given context.

HOMONYM: {homonym}

POSSIBLE MEANINGS:
{meanings_list}

CONTEXT:
Precontext: {precontext}
Sentence containing the homonym: {sentence}
Ending: {ending}

TASK:
1. First, provide exactly 3 SHORT bullet points (one line each) analyzing the plausibility of each meaning given the context.
2. Then, on a new line, output your judgment:
   - "JUDGMENT: CLEAR - [meaning number]" if ONE meaning is highly plausible and the OTHER has low plausibility
   - "JUDGMENT: AMBIGUOUS" if BOTH meanings are plausible OR both have similar plausibility levels

CLEAR = one meaning fits well, the other does not
AMBIGUOUS = both meanings could work, or neither clearly dominates

Consider:
- Does the precontext make one meaning much more likely than the other?
- Does the sentence usage strongly favor one interpretation?
- Does the ending confirm one meaning while ruling out the other?

Format your response EXACTLY as:
• [bullet point 1]
• [bullet point 2]
• [bullet point 3]
JUDGMENT: [CLEAR - N or AMBIGUOUS]
"""



def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def apply_chat(tokenizer, system_text: str, user_text: str):
    messages = [{"role": "system", "content": system_text},
                {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def group_by_context(data: dict) -> dict:
    """Group items by their context (precontext + sentence + ending) to find pairs with same context."""
    groups = {}
    for key, item in data.items():
        context_key = (item["precontext"], item["sentence"], item.get("ending", ""))
        if context_key not in groups:
            groups[context_key] = []
        groups[context_key].append((key, item))
    return groups

def parse_oracle_response(response: str) -> dict:
    """Parse the oracle response to extract analysis and judgment.
    
    Expected format from prompt:
    Brief explanation
    
    JUDGMENT: [CLEAR - N or AMBIGUOUS]
    
    Note: The response may include the echoed prompt, so we look for the LAST
    occurrence of JUDGMENT: which is the actual model output.
    """
    import re
    
    lines = response.strip().split("\n")
    judgment = "AMBIGUOUS"
    clear_meaning = None
    explanation = ""
    
    # Find the LAST JUDGMENT line (the actual output, not the one in the prompt)
    judgment_idx = -1
    for i in range(len(lines) - 1, -1, -1):  # Iterate backwards
        if "JUDGMENT:" in lines[i].upper():
            judgment_idx = i
            break
    
    # Extract explanation: look for "assistant" marker or use text before last JUDGMENT
    assistant_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "assistant":
            assistant_idx = i
    
    if assistant_idx >= 0 and judgment_idx > assistant_idx:
        # Extract text between "assistant" and the JUDGMENT line
        explanation = "\n".join(lines[assistant_idx + 1:judgment_idx]).strip()
    elif judgment_idx > 0:
        explanation = "\n".join(lines[:judgment_idx]).strip()
    
    # Parse the LAST judgment line
    if judgment_idx >= 0:
        line = lines[judgment_idx].strip()
        upper_line = line.upper()
        if "CLEAR" in upper_line:
            judgment = "CLEAR"
            # Extract meaning number: "CLEAR - 1" or "CLEAR - 2"
            match = re.search(r'CLEAR\s*[-:]\s*(\d+)', upper_line)
            if match:
                clear_meaning = int(match.group(1))
        elif "AMBIGUOUS" in upper_line:
            judgment = "AMBIGUOUS"
    
    return {
        "explanation": explanation,
        "judgment": judgment,
        "clear_meaning": clear_meaning
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True, help="Path to input JSON (e.g., data/train.json)")
    p.add_argument("--out_dir", default="outputs_oracle", help="Output directory for results")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    args = p.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group by context to find pairs with same context but different meanings
    context_groups = group_by_context(data)
    print(f"Found {len(context_groups)} unique contexts from {len(data)} items.", flush=True)
    
    # Create oracle items - one per unique context
    oracle_items = []
    for context_key, items in context_groups.items():
        precontext, sentence, ending = context_key
        homonym = items[0][1]["homonym"]
        
        # Collect all different meanings for this context
        meanings = {}
        for key, item in items:
            meaning = item["judged_meaning"]
            if meaning not in meanings:
                meanings[meaning] = {
                    "meaning": meaning,
                    "example_sentence": item.get("example_sentence", ""),
                    "keys": []
                }
            meanings[meaning]["keys"].append(key)
        
        meanings_list = list(meanings.values())
        
        oracle_items.append({
            "context_id": items[0][0],  # Use first key as context ID
            "homonym": homonym,
            "precontext": precontext,
            "sentence": sentence,
            "ending": ending,
            "meanings": meanings_list,
            "original_keys": [k for k, _ in items]
        })
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.skip_existing:
        before = len(oracle_items)
        oracle_items = [item for item in oracle_items 
                       if not os.path.exists(os.path.join(args.out_dir, f"{item['context_id']}.json"))]
        print(f"Loaded {before}; skipping {before - len(oracle_items)} existing.", flush=True)
    else:
        print(f"Processing {len(oracle_items)} unique contexts.", flush=True)
    
    if not oracle_items:
        print("Nothing to do.")
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})

    # Load model in 4-bit precision
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_config,
    ).eval()

    device = model.device
    print(f"Model on: {device} | dtype: {getattr(model, 'dtype', 'mixed')} | GPU count: {torch.cuda.device_count()}", flush=True)

    with tqdm(total=len(oracle_items), desc="Oracle Judging", unit="ctx") as pbar, torch.inference_mode():
        for batch in chunked(oracle_items, args.batch_size):
            prompts = []
            for item in batch:
                # Format meanings list with numbers
                meanings_formatted = "\n".join([
                    f"{i+1}. {m['meaning']}" + (f" (e.g., \"{m['example_sentence']}\")" if m['example_sentence'] else "")
                    for i, m in enumerate(item["meanings"])
                ])
                
                user_prompt = ORACLE_PROMPT.format(
                    homonym=item["homonym"],
                    meanings_list=meanings_formatted,
                    precontext=item["precontext"],
                    sentence=item["sentence"],
                    ending=item["ending"] or "(none)"
                )
                prompts.append(apply_chat(tok, "You are an expert linguist specializing in lexical semantics and word sense disambiguation.", user_prompt))

            enc = tok(prompts, return_tensors="pt", padding=True, truncation=False)
            enc = to_device(enc, device)

            out = model.generate(
                **enc,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
            
            responses = tok.batch_decode(out, skip_special_tokens=True)

            for i, item in enumerate(batch):
                response = responses[i]
                # Extract only the generated part (after the prompt)
                # Find the last occurrence of the prompt ending markers
                generated = response
                for marker in ["JUDGMENT:", "•"]:
                    if marker in response:
                        # Find the response section
                        break
                
                parsed = parse_oracle_response(response)
                
                # Determine which meaning if CLEAR
                clear_meaning_text = None
                if parsed["judgment"] == "CLEAR" and parsed["clear_meaning"] is not None:
                    idx = parsed["clear_meaning"] - 1
                    if 0 <= idx < len(item["meanings"]):
                        clear_meaning_text = item["meanings"][idx]["meaning"]
                
                result = {
                    "context_id": item["context_id"],
                    "homonym": item["homonym"],
                    "precontext": item["precontext"],
                    "sentence": item["sentence"],
                    "ending": item["ending"],
                    "meanings_considered": [m["meaning"] for m in item["meanings"]],
                    "explanation": parsed["explanation"],
                    "judgment": parsed["judgment"],
                    "clear_meaning_index": parsed["clear_meaning"],
                    "clear_meaning_text": clear_meaning_text,
                    "original_keys": item["original_keys"],
                    "raw_response": response
                }
                
                out_path = os.path.join(args.out_dir, f"{item['context_id']}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                judgment_str = f"{parsed['judgment']}"
                if parsed["judgment"] == "CLEAR" and clear_meaning_text:
                    judgment_str += f" -> {clear_meaning_text[:40]}..."
                tqdm.write(f"[✓] {item['context_id']} ({item['homonym']}): {judgment_str}")

            pbar.update(len(batch))

    print(f"\nDone. Wrote oracle judgments to {args.out_dir}")

if __name__ == "__main__":
    try:
        import transformers as _tx
        from packaging import version
        if version.parse(_tx.__version__) < version.parse("4.51.0"):
            print(f"[!] transformers {_tx.__version__} detected; please upgrade to >= 4.51.0 for Qwen3.", file=sys.stderr)
    except Exception:
        pass
    main()
