import argparse
import csv
import os
from collections import defaultdict
import json
import jieba
import MeCab
import pkuseg
import nlptutti as nt
from transformers import WhisperProcessor
from tqdm import tqdm
import re
import numpy as np
from soynlp.hangle import compose, decompose, character_is_korean

seg = pkuseg.pkuseg()
mecab = MeCab.Tagger("-Owakati")

REF_PATTERN = re.compile(r"<NA>\s+<NA>\s*(.*?)\s*\|\|\|")

def load_ref_file(ref_path):
    file_texts = defaultdict(list)
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("SPEAKER"): continue
            parts = line.split(maxsplit=2)
            if len(parts) < 2: continue
            fid = parts[1]
            m = REF_PATTERN.search(line)
            if m:
                text = m.group(1).strip()
            else:
                tokens = line.strip().split(maxsplit=10)
                if len(tokens) < 11: continue
                candidate = tokens[10].strip()
                text = candidate[1:-1] if candidate.startswith("<") and candidate.endswith(">") else candidate
            file_texts[fid].append(text)
    return {fid: " ".join(segs) for fid, segs in file_texts.items()}


def load_pred_file(pred_path):
    file_texts = defaultdict(str)
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for fid, segments in data.items():
            file_texts[fid] = " ".join(seg["text"] for seg in segments)
    return file_texts


def normalize_texts(reference, prediction, processor):
    return {
        "reference_normalized": processor.tokenizer._normalize(reference),
        "prediction_normalized": processor.tokenizer._normalize(prediction),
    }


def segment_chinese(text):
    text = re.sub(r"[，。]", " ", text)
    return " ".join(seg.cut(text))


def segment_japanese(text):
    return mecab.parse(text).strip()


def jamo_split(text):
    decomposed = ''
    for char in text:
        try:
            cho, jung, jong = decompose(char)
            decomposed += (cho or '-') + (jung or '-') + (jong or '-')
        except:
            decomposed += char
    return decomposed


def wer(ref_tokens, hyp_tokens):
    d = np.zeros((len(ref_tokens)+1, len(hyp_tokens)+1), dtype=int)
    for i in range(len(ref_tokens)+1): d[i][0] = i
    for j in range(len(hyp_tokens)+1): d[0][j] = j
    for i in range(1, len(ref_tokens)+1):
        for j in range(1, len(hyp_tokens)+1):
            cost = 0 if ref_tokens[i-1] == hyp_tokens[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref_tokens)][len(hyp_tokens)] / max(len(ref_tokens), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_code", type=str, default="ko")
    parser.add_argument("--ref", type=str, default=".../KR.rttm")
    parser.add_argument("--pred", type=str, default=".../KR_transcriptions.json")
    parser.add_argument("--output", type=str, default=".../output.csv")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    reference_dict = load_ref_file(args.ref)
    prediction_dict = load_pred_file(args.pred)

    ref_ids = set(reference_dict.keys())
    pred_ids = set(prediction_dict.keys())
    common_ids = sorted(ref_ids & pred_ids)
    missing_in_ref = pred_ids - ref_ids
    missing_in_pred = ref_ids - pred_ids

    print(f"Total references: {len(ref_ids)}")
    print(f"Total predictions: {len(pred_ids)}")
    print(f"Matched files: {len(common_ids)}")
    if missing_in_ref: print(f"[Warning] {len(missing_in_ref)} preds without refs: {missing_in_ref}")
    if missing_in_pred: print(f"[Warning] {len(missing_in_pred)} refs without preds: {missing_in_pred}")

    is_new = not os.path.exists(args.output)
    with open(args.output, mode="a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        if is_new:
            writer.writerow(["audio_id", "wer", "cer", "reference", "prediction"])
            fout.flush()

        results = {}
        for fid in tqdm(common_ids, desc="Computing WER/CER"):
            ref_text = reference_dict[fid]
            pred_text = prediction_dict[fid]
            norm = normalize_texts(ref_text, pred_text, processor)
            r = norm["reference_normalized"]
            p = norm["prediction_normalized"]

            lang = args.lang_code.lower()
            if lang == "zh":
                r_seg = segment_chinese(r)
                p_seg = segment_chinese(p)
                wer_val = 100 * nt.get_wer(r_seg, p_seg)["wer"]
            elif lang == "ja":
                r_seg = segment_japanese(r)
                p_seg = segment_japanese(p)
                wer_val = 100 * nt.get_wer(r_seg, p_seg)["wer"]
            elif lang == "ko":
                r_jamo = jamo_split(r)
                p_jamo = jamo_split(p)
                r_units = [r_jamo[i:i+3] for i in range(0, len(r_jamo), 3)]
                p_units = [p_jamo[i:i+3] for i in range(0, len(p_jamo), 3)]
                wer_val = round(100 * wer(r_units, p_units), 4)
            else:
                wer_val = round(100 * nt.get_wer(r, p)["wer"], 4)

            cer_val = round(100 * nt.get_cer(r, p)["cer"], 4)

            results[fid] = {"wer": wer_val, "cer": cer_val}
            writer.writerow([fid, wer_val, cer_val, ref_text, pred_text])
            fout.flush()

        avg_wer = sum(x["wer"] for x in results.values()) / len(results)
        avg_cer = sum(x["cer"] for x in results.values()) / len(results)
        writer.writerow(["average", round(avg_wer,4), round(avg_cer,4), "", ""])
        fout.flush()

    print("Overall Average Scores:")
    print(f"  Average WER: {avg_wer:.4f}")
    print(f"  Average CER: {avg_cer:.4f}")
    print(f"CSV file '{args.output}'에 결과가 실시간 누적 저장되었습니다.")
