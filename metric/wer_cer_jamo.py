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
from soynlp.hangle import decompose, character_is_korean

# 세그멘터 초기화
seg = pkuseg.pkuseg()
mecab = MeCab.Tagger("-Owakati")
REF_PATTERN = re.compile(r"<NA>\s+<NA>\s*(.*?)\s*\|\|\|")

# 자모 분해 함수
def sent2phonemes(sent):
    phonemes = []
    for char in sent:
        if character_is_korean(char):
            d = list(decompose(char))
            if d[2] == ' ':
                d[2] = 'JONG_EMPTY'
            phonemes.extend(d)
        else:
            phonemes.append(char)
    return phonemes

# 자모 토크나이저 및 WER 계산기
class Tokenizer:
    def __init__(self):
        self.special = ['SOS','EOS','PAD']
        self.ja = ['ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄸ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ',
                   'ㅁ','ㅂ','ㅃ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','JONG_EMPTY']
        self.mo = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ',
                   'ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
        self.alpha = list('abcdefghijklmnopqrstuvwxyz')
        self.num = list('0123456789')
        self.punc = [',','.','?','!',' ']
        vocab = self.special + self.ja + self.mo + self.alpha + self.num + self.punc
        self.ph2id = {ph: i for i, ph in enumerate(vocab)}
        self.id2ph = {i: ph for ph, i in self.ph2id.items()}
        self.sos = self.ph2id['SOS']
        self.eos = self.ph2id['EOS']
        self.pad = self.ph2id['PAD']

    def encode(self, text):
        phs = sent2phonemes(text.lower())
        ids = [self.sos] + [self.ph2id.get(p, self.pad) for p in phs] + [self.eos]
        return [i for i in ids if i in self.id2ph]

    def decode(self, ids):
        return [self.id2ph[i] for i in ids if i not in (self.sos, self.eos, self.pad)]

    def compute_wer(self, ref, hyp):
        r_ids = self.decode(self.encode(ref))
        h_ids = self.decode(self.encode(hyp))
        d = np.zeros((len(r_ids)+1, len(h_ids)+1), dtype=int)
        for i in range(len(r_ids)+1): d[i][0] = i
        for j in range(len(h_ids)+1): d[0][j] = j
        for i in range(1, len(r_ids)+1):
            for j in range(1, len(h_ids)+1):
                cost = 0 if r_ids[i-1] == h_ids[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        return d[len(r_ids)][len(h_ids)] / max(len(r_ids), 1)

# 참조 파일 로더
def load_ref_file(path):
    data = defaultdict(list)
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if not line.startswith("SPEAKER"): continue
            parts = line.split(maxsplit=2)
            if len(parts) < 2: continue
            fid = parts[1]
            m = REF_PATTERN.search(line)
            if m:
                txt = m.group(1).strip()
            else:
                toks = line.strip().split(maxsplit=10)
                if len(toks) >= 11 and toks[10].startswith("<") and toks[10].endswith(">"):
                    txt = toks[10][1:-1]
                else:
                    txt = toks[10] if len(toks) >= 11 else ''
            data[fid].append(txt)
    return {k: ' '.join(v) for k, v in data.items()}

# 예측 파일 로더
def load_pred_file(path):
    data = defaultdict(str)
    with open(path, encoding='utf-8-sig') as f:
        js = json.load(f)
        for fid, segs in js.items():
            data[fid] = ' '.join(s['text'] for s in segs)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_code', type=str, default='kr')
    parser.add_argument('--ref', type=str, default="/path/to/your/IIPL_Flitto/test/KR.rttm")
    parser.add_argument('--pred', type=str, default="/path/to/your/IIPL_Flitto/test/KR_transcriptions.json")
    parser.add_argument('--output', type=str, default="/path/to/your/IIPL_Flitto/test/output_with_jamo.csv")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    jamo_tok  = Tokenizer() if args.lang_code.lower() == 'kr' else None

    refs  = load_ref_file(args.ref)
    preds = load_pred_file(args.pred)
    ids   = sorted(set(refs) & set(preds))

    is_new = not os.path.exists(args.output)
    total_wer, total_cer, total_jamo = 0.0, 0.0, 0.0
    cnt, jamo_cnt = 0, 0

    # CSV를 utf-8-sig로 저장하여 한글 깨짐 방지
    with open(args.output, 'a', newline='', encoding='utf-8-sig') as fout:
        writer = csv.writer(fout)
        if is_new:
            writer.writerow([
                'audio_id','wer','cer','jamo_ref','jamo_hyp','reference','prediction'
            ])

        for fid in tqdm(ids, desc="Scoring files"):
            r_raw, p_raw = refs[fid], preds[fid]
            norm_ref = processor.tokenizer.normalize(r_raw)
            norm_prd = processor.tokenizer.normalize(p_raw)

            if args.lang_code.lower() == 'cn':
                ref_norm = ' '.join(seg.cut(norm_ref.replace('，',' ').replace('。',' ')))
                prd_norm = ' '.join(seg.cut(norm_prd.replace('，',' ').replace('。',' ')))
                wer_val = nt.get_wer(ref_norm, prd_norm)['wer'] * 100
            elif args.lang_code.lower() == 'jp':
                ref_norm = mecab.parse(norm_ref).strip()
                prd_norm = mecab.parse(norm_prd).strip()
                wer_val = nt.get_wer(ref_norm, prd_norm)['wer'] * 100
            else:
                ref_norm, prd_norm = norm_ref, norm_prd
                wer_val = nt.get_wer(ref_norm, prd_norm)['wer'] * 100

            wer_val = round(wer_val, 4)
            cer_val = round(nt.get_cer(ref_norm, prd_norm)['cer'] * 100, 4)

            if jamo_tok:
                ref_jamos = jamo_tok.decode(jamo_tok.encode(r_raw))
                prd_jamos = jamo_tok.decode(jamo_tok.encode(p_raw))
                jamo_val = round(jamo_tok.compute_wer(r_raw, p_raw) * 100, 4)
            else:
                ref_jamos, prd_jamos, jamo_val = [], [], ''

            writer.writerow([
                fid, wer_val, cer_val, jamo_val,
                ' '.join(ref_jamos), ' '.join(prd_jamos),
                r_raw, p_raw
            ])

            total_wer += wer_val
            total_cer += cer_val
            cnt += 1
            if jamo_tok:
                total_jamo += jamo_val
                jamo_cnt += 1

        if cnt > 0:
            avg_wer  = round(total_wer  / cnt,      4)
            avg_cer  = round(total_cer  / cnt,      4)
            avg_jamo = round(total_jamo / jamo_cnt, 4) if jamo_cnt > 0 else ''
            writer.writerow(['average', avg_jamo, avg_cer, '', '', '', ''])
    
    print("Overall Average Scores:")
    print(f"  Average WER: {avg_jamo:.4f}")
    print(f"  Average CER: {avg_cer:.4f}")
    print("Done. Results appended to", args.output)
