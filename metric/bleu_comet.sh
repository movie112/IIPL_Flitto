#!/bin/bash

langs=("en" "cn" "jp")
root="/path/to/your/IIPL_Flitto"
ckpt_dir="/path/to/your/Machine_Translation_ckpt"
data_dir="${root}/Text_Processing/Machine_Translation/data"
result_base="${root}/Text_Processing/Machine_Translation/prediction"

# fairseq-generate + 결과 분리
for lang in "${langs[@]}"; do
  model_dir="${ckpt_dir}/real_ckp_ko_${lang}"
  data_path="${data_dir}/spm_ko_${lang}"
  result_path="${result_base}/${lang}"
  mkdir -p "$result_path"

  echo "Generating translation for ko→${lang}..."
  CUDA_VISIBLE_DEVICES=1 fairseq-generate "$data_path" \
    --path "${model_dir}/checkpoint_best.pt" \
    --source-lang ko --target-lang ${lang} \
    --beam 5 --task translation --gen-subset test \
    --remove-bpe sentencepiece \
    --skip-invalid-size-inputs-valid-test \
    --required-batch-size-multiple 1 \
    --max-sentences 128 --batch-size 1 \
    --results-path "$result_path" \

  echo " Extracting results for ${lang}..."
  grep '^H' "${result_path}/generate-test.txt" | cut -f3- > "${result_path}/gen.out.sys"
  grep '^T' "${result_path}/generate-test.txt" | cut -f2- > "${result_path}/gen.out.ref"
  grep '^S' "${result_path}/generate-test.txt" | cut -f2- > "${result_path}/gen.out.src"
done

#  평가: Python으로 BLEU/COMET 수행
python <<EOF

import os, json, subprocess, re
import sacrebleu
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint

def char_tokenize(inp, out):
    with open(inp, 'r', encoding='utf-8') as fin, open(out, 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(' '.join(list(line.strip())) + '\n')

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

results = {}
def evaluate(lang, tokenize_fn=None):
    prediction=f"{result_base}/{lang}"
    src = open(f"{prediction}/gen.out.src", encoding="utf-8").read().splitlines()
    ref = f"{prediction}/gen.out.ref"
    hyp = f"{prediction}/gen.out.sys"

    ref_tok, hyp_tok = ref, hyp
    if tokenize_fn:
        ref_tok = f"{ref}.tok"
        hyp_tok = f"{hyp}.tok"
        tokenize_fn(ref, ref_tok)
        tokenize_fn(hyp, hyp_tok)

    refs = open(ref_tok, encoding="utf-8").read().splitlines()
    hyps = open(hyp_tok, encoding="utf-8").read().splitlines()
    if lang == "en":
        bleu13a = BLEU({'tokenizer': '13a'})
        bleu = bleu13a.corpus_score(hyps, [refs]).score
    else:
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    # bleu = corpus_bleu(hyps, [refs]).score
    print(f"\n BLEU ({lang}): {bleu:.2f}")

    data = [
        {"src": s, "mt": m, "ref": r}
        for s, m, r in zip(src, hyps, refs)
    ]

    # COMET input 준비
    result_dict = comet_model.predict(
        data,
        batch_size=32
    )
    seg_scores = result_dict["scores"]
    sys_score = result_dict["system_score"]

    if isinstance(sys_score, str):
        sys_score = float(sys_score)
    print(f" COMET ({lang}): {sys_score:.4f}")

    # 결과 저장
    results[lang] = {
        "BLEU": round(bleu, 2),
        "COMET": round(sys_score, 4)
    }

# 각 언어별 평가
evaluate("en")
evaluate("cn", char_tokenize)
evaluate("jp", char_tokenize)

# 결과를 JSON으로 저장
with open(f"{result_base}/scores.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

EOF
