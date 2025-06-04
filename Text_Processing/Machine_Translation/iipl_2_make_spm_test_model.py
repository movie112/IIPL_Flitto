import sentencepiece as spm
import os
import argparse

def tokenize(input_path, model_path, output_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(" ".join(sp.encode(line.strip(), out_type=str)) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--src_lang", default="ko")
    parser.add_argument("--tgt_lang", required=True)
    parser.add_argument("--spm_src", required=True)
    parser.add_argument("--spm_tgt", required=True)
    args = parser.parse_args()

    tokenize(f"{args.base_dir}/test.{args.src_lang}", args.spm_src, f"{args.base_dir}/test_spm.{args.src_lang}")
    tokenize(f"{args.base_dir}/test.{args.tgt_lang}", args.spm_tgt, f"{args.base_dir}/test_spm.{args.tgt_lang}")
    
if __name__ == "__main__":
    main()
