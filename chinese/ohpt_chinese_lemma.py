import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

DEV_PATH = "data/dev.json"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# 1. Load Qwen model
# -------------------------
print("Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)


# -------------------------
# 2. Extract assistant reply (Qwen2.5 ChatML)
# -------------------------
def extract_assistant_answer(decoded_text: str) -> str:

    # ChatML 格式：<|im_start|>assistant ... <|im_end|>
    if "<|im_start|>assistant" in decoded_text:
        part = decoded_text.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in part:
            part = part.split("<|im_end|>")[0]
        decoded_text = part.strip()

    # 保留中文
    chinese = re.findall(r"[\u4e00-\u9fff]+", decoded_text)
    if chinese:
        return chinese[0]

    # fallback
    return decoded_text.strip()


# -------------------------
# 3. Ask Qwen2.5 for lemma
# -------------------------
def query_qwen_for_lemma(text, homonym):
    """
    输入英文上下文和多义词，输出最可能的中文含义（只输出单个中文词语）
    """

    prompt = f"""
你是一个英文多义词中文释义判断助手。
给定一段英文内容与一个多义词 homonym，你需要判断它在上下文中的中文含义。

内容:
{text}

目标词:
{homonym}

请输出这个词在句子中的**中文释义**，只输出一个最合适的中文词语。不要输出英文。
"""

    messages = [{"role": "user", "content": prompt}]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    output_ids = model.generate(
        model_inputs,
        max_new_tokens=64,
        temperature=0.1,
        do_sample=False,
    )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # ---- 解析 assistant 输出 ----
    answer = extract_assistant_answer(decoded)
    return answer


# -------------------------
# 4. Main logic
# -------------------------
def main():
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    dev_list = list(dev_data.values())

    results = []

    for item in dev_list:

        pre = item.get("precontext", "")
        sent = item.get("sentence", "")
        ending = item.get("ending", "")
        hom = item.get("homonym", "")

        merged_text = f"{pre} {sent} {ending}".strip()

        zh_lemma = query_qwen_for_lemma(merged_text, hom)

        results.append({
            "sample_id": item.get("sample_id"),
            "homonym": hom,
            "merged_text": merged_text,
            "predicted_chinese_lemma": zh_lemma
        })

        print(f"{item.get('sample_id')}: {hom} -> {zh_lemma}")

    with open("dev_chinese_lemma.jsonl", "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✔ Saved to dev_chinese_lemma.jsonl")


if __name__ == "__main__":
    main()
