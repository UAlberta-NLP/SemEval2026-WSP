import pexpect
import os
import spacy
import pandas as pd
import io
import Levenshtein
import re
import argparse


nlp_models = {
    "en": spacy.load("en_core_web_lg"),
}

def tokenize(sentence: str, lang: str = "en"):
   
    if lang not in nlp_models:
        raise ValueError("Language is not found")
    
    nlp = nlp_models[lang]
    
    if isinstance(sentence, float):
        return [], []
    doc = nlp(sentence.strip())
    return [token.text for token in doc], [token.lemma_ for token in doc]

def step_text(prompt, answer):
    """Step with a single string answer."""
    return (prompt, answer)


def step_list(prompt, answers):
    """Step with a list of answers."""
    return (prompt, answers)


def make_lemma_def_pairs(pairs):
    """
    Convert a list of (lemma, definition) pairs into:
        ["lemma --- definition", ...]
    Automatically appends "" to represent 'Enter to stop'.
    """
    arr = [f"{lemma} --- {definition}" for lemma, definition in pairs]
    arr.append("")  # blank line to stop
    return arr

from rapidfuzz import process, fuzz

def closest_to_index(centre_tok, homo):
    # process.extractOne returns (best_match, score, index)
    best_match = process.extractOne(
        homo,
        centre_tok,
        scorer=fuzz.ratio
    )
    return best_match[2]   # index of the closest token

def make_context_entries(entries):
    """
    Convert context lemma-def-position tuples into:
        ["lemma --- definition --- position", ...]
    Does NOT automatically append a blank-line-stop unless desired.
    """
    if not entries:
        return ""
    return [f"{lemma} --- {definition} --- {pos}" for lemma, definition, pos in entries]


NEXT_PROMPT = "Enter space-separated text:"


def make_session(
    text,
    target_pos,
    lemma_defs,
    context_entries,
):
    session = [
        step_text("Enter space-separated text:", text),
        step_text("Target position:", str(target_pos)),
        step_list(
            'Enter candidate lemma-def pairs. " --- " separated. Enter to stop',
            make_lemma_def_pairs(lemma_defs),
        ),
        step_list(
            'Enter context lemma-def-position tuples. " --- " separated. Position should be token position in space-separated input. Enter to stop',
            make_context_entries(context_entries or []),
        ),
    ]
    return session


def parse_for_probs(output):
    lines_of_output = output.split('\n')
    for x, guy in enumerate(lines_of_output):
        curr_line = guy.strip()
        if curr_line == '# predictions':
            return lines_of_output[x+1] + '\n' + lines_of_output[x+2] + '\n'
    assert False


def run_one_session(child, session_steps):
    for expect_str, send_str in session_steps:
        if expect_str != NEXT_PROMPT:
            child.expect(expect_str)
        else:
            pass

        if isinstance(send_str, list):
            for line in send_str:
                child.sendline(line)
        else:
            child.sendline(send_str)

    child.expect(NEXT_PROMPT)
    
def start_repl(command):
    
    child = pexpect.spawn(command, encoding="utf-8", timeout=None)
    buffer = io.StringIO()
    child.logfile = buffer

    # Wait for initial prompt once
    
    child.expect(NEXT_PROMPT)
    return child, buffer

def similarity(string_one, string_two):
    return - Levenshtein.distance(string_one, string_two)

def scale_to_1_to_5(prob):
    assert isinstance(prob, str)
    simple_str = prob.strip('*').strip()
    fl = float(simple_str)
    return 1 + (4*fl)

def get_output_nums(strin, curr_meaning):
    
    assert isinstance(curr_meaning, str)
    
    
    a, b, _ = strin.split('\n')
    
    prob_one, _, gloss_one = a.strip().split('\t')
    prob_two, _, gloss_two = b.strip().split('\t')
   
    
    if similarity(gloss_one, curr_meaning) >= similarity(gloss_two, curr_meaning):
        return scale_to_1_to_5(prob_one)
    else:
        return scale_to_1_to_5(prob_two)
    

def main(sessions, command, addr, input_f):
    child, buffer = start_repl(command)
    
    df = pd.read_json(input_f, orient="index").reset_index()
    a = 0
    
    with open(addr, 'w', encoding='utf-8') as f:
        for i, steps in enumerate(sessions, start=1):
            print(f"\n=== Running session {i}/{len(sessions)} ===")
            
            # Only parse new output
            start_len = buffer.tell()
            run_one_session(child, steps)
            buffer.seek(start_len)
            output = buffer.read()
            
            output_text = parse_for_probs(output)
            
            out_num = get_output_nums(output_text, df['judged_meaning'].iloc[a])
           
            f.write('{"id": "' + str(a) + '", "prediction": ' + str(out_num) + '}\n')
            a += 1

    child.terminate(force=True)
    child.close()


def get_sense_dict(df):
    ans = {}
    for index, row in df.iterrows():
        meaning_string = row['judged_meaning']  
        if row['precontext'] not in ans:
            ans[row['precontext']] = []
        if meaning_string not in ans[row['precontext']]:
            ans[row['precontext']].append(meaning_string)
    
    return ans

def space_sep_find_ind(pre, centre, post, homo):
    pre_tok, _ = tokenize(pre)
    centre_tok, _ = tokenize(centre)
    post_tok, _ = tokenize(post)
    
   
    
    if homo in centre_tok:
        index = len(pre_tok) + centre_tok.index(homo)
     
    elif homo.lower() in [a.lower() for a in centre_tok]:
        lower_list_guy = [a.lower() for a in centre_tok]
        index = len(pre_tok) + lower_list_guy.index(homo.lower())
      
    else:
        index = len(pre_tok) + closest_to_index(centre_tok, homo)
    
    
    return ' '.join([' '.join(pre_tok), ' '.join(centre_tok), ' '.join(post_tok)]), index

def prepare_sessions(path):
    ans = []
                 
    df = pd.read_json(path, orient="index").reset_index()
    
    sense_dict = get_sense_dict(df)
    
    for _, row in df.iterrows():
        
        senses = sense_dict[row['precontext']]
        
        homonym = row['homonym']
        assert len(senses) == 2
        space_sep_text, ind = space_sep_find_ind(row['precontext'], row['sentence'], row['ending'], homonym)
        
        ans.append(make_session(space_sep_text, ind, [(homonym, s) for s in senses], []))
  
    
    return ans

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ExpandNet output against gold standard.")
    parser.add_argument("--input_file", type=str, help="Path to the input json file.", default='test.json')
    parser.add_argument("--output_file", type=str, help="Path to the desired output jsonl file location.", default='wsdpredictions.jsonl')
    parser.add_argument("--consec_predict_file", type=str, default="src/scripts/model/predict.py", 
                        help="Path to the location of the predict script downloaded from ConSec, relative to the ConSec root folder.")
    parser.add_argument("--consec_location", type=str, 
                        help="Path to the root ConSec directory.")
    parser.add_argument("--ckpt", type=str, default="experiments/released-ckpts/consec_wngt_best.ckpt", 
                        help="Path to the installed location of the desired ckpt file to use, relative to the ConSec root folder.")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    predict_script = os.path.join(args.consec_location, args.consec_predict_file)
    ckpt_path = os.path.join(args.consec_location, args.ckpt)
    
    command = f'bash -c "PYTHONPATH={args.consec_location} python {predict_script} {ckpt_path} -t"'
    
    all_sessions = prepare_sessions(args.input_file)
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    
    main(all_sessions, command, args.output_file, args.input_file)
    
