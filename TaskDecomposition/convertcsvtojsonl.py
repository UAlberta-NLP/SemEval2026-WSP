import argparse

def main(in_file, out_file):
    a = 0
    assert in_file != out_file
    with open(in_file, 'r', encoding='utf-8') as in_f:
        with open(out_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                _, _, p, _ = line.split(',')
               
                if p != 'pred':
                    out_f.write('{"id": "' + str(a) + '", "prediction": ' + str(float(p)) + '}\n')
                    a += 1
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()
    
    main(args.in_csv, args.out_jsonl)