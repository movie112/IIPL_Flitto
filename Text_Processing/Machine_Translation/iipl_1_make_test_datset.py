import pandas as pd
from pathlib import Path
import argparse

def make_test_only(csv_path, output_dir, src_lang, tgt_lang):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['transcript', 'translation'])
    df = df[(df['transcript'].str.strip() != '') & (df['translation'].str.strip() != '')]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"test.{src_lang}", 'w', encoding='utf-8') as f_src, \
         open(output_dir / f"test.{tgt_lang}", 'w', encoding='utf-8') as f_tgt:
        for _, row in df.iterrows():
            f_src.write(row['transcript'].strip() + '\n')
            f_tgt.write(row['translation'].strip() + '\n')

    print(f"[✔] Saved test.{src_lang}, test.{tgt_lang} to {output_dir} ({len(df)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("output_dir")
    parser.add_argument("--src_lang", default="ko")
    parser.add_argument("--tgt_lang", required=True)
    args = parser.parse_args()

    make_test_only(args.csv_path, args.output_dir, args.src_lang, args.tgt_lang)
