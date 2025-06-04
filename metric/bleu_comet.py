import os, json
import sacrebleu
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
import argparse

def char_tokenize(inp, out):
    with open(inp, 'r', encoding='utf-8') as fin, open(out, 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(' '.join(list(line.strip())) + '\n')

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

results = {}
langs = ["en", "cn", "jp"]

def evaluate(lang, result_base, tokenize_fn=None):
    prediction = f"{result_base}/{lang}"
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
    bleu = bleu / 100.0
    print(f"\n BLEU ({lang}): {bleu:.4f}")

    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, hyps, refs)]
    result_dict = comet_model.predict(data, batch_size=32)
    sys_score = float(result_dict["system_score"])

    print(f" COMET ({lang}): {sys_score:.4f}")

    results[lang] = {
        "BLEU": round(bleu, 4),
        "COMET": round(sys_score, 4)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_base", help="Base directory for results")
    args = parser.parse_args()

    evaluate("en", args.result_base)
    evaluate("cn", args.result_base, char_tokenize)
    evaluate("jp", args.result_base, char_tokenize)
    with open(f"{args.result_base}/scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()



