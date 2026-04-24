
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import re

model_name = "Qwen/Qwen3-14B"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ExpandNet output against gold standard.")
    parser.add_argument("--input_file", type=str, help="Path to the input json file.", default='test.json')
    parser.add_argument("--output_file", type=str, help="Path to the desired output jsonl file location.", default='qwenpredictions.jsonl')
    return parser.parse_args()

def last_int_in_str(text: str):
    matches = re.findall(r"\d+", text)
    return int(matches[-1]) if matches else 3
        
            
def make_sense_prompt(a_word, precontext, sentence, ending, a_sense):
    return (
        f'HOMONYM:\n{a_word}\nTEXT:Precontext: {precontext}\n**{sentence}**\nEnding: {ending}\n\nMeaning being judged: "{a_sense}"'
    )

def remove_thinking(text):
    if len(text.split('/think>')) == 2:
        reasoning, ans = text.split('/think>')
        try:
            return int(ans)
        except ValueError:
            pass
        
    return last_int_in_str(text)


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
"""


def llm_this_helper(prompt):
    messages = [
    {
        "role": "system",
        "content": AMBISTORY_PROMPT
    },
    {
        "role": "user",
        "content": prompt
    }
    ]
    
    

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    return remove_thinking(content)

CACHE = {}

def llm_cachable(p, s, e, w, g):
    key = (p, s, e, w, g, 'notdecrealthinggg')
    if key not in CACHE:
        CACHE[key] = llm_this_helper(make_sense_prompt(w, p, s, e, g))
    return CACHE[key]
        

def score_llm(p, o_sent, e, o_word, o_gloss):
    return llm_cachable(p, o_sent, e, o_word, o_gloss)
        
def do_llm(path):
    ans = []
    
    df = pd.read_json(path, orient="index").reset_index()
    
    rows_to_process = []
    
    for _, row in df.iterrows():
        rows_to_process.append(row)
        
    for row in tqdm(rows_to_process, desc="Processing", unit="lines"):
            
            precon = row['precontext']
            orig_f_sent = row['sentence']
            ending = row['ending']
            original_lemma = row['homonym']
            sense = row['judged_meaning']
            
            score = score_llm(precon, orig_f_sent, ending, original_lemma, sense)
            ans.append(score)
            
    return ans

if __name__ == "__main__":
    
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    a = 0

    lines_to_write = do_llm(args.input_file)

    a = 0
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for prediciton_value in lines_to_write:
            f.write('{"id": "' + str(a) + '", "prediction": ' + str(prediciton_value) + '}\n')
            a += 1
