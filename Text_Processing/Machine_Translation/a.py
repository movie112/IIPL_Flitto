from sacrebleu.metrics import BLEU
import csv

# 파일 경로 설정
ref_file = "/home/byeonggeuk/IIPL_Flitto/Text_Processing/Machine_Translation/prediction/cn/gen.out.ref"
hyp_file = "/home/byeonggeuk/IIPL_Flitto/Text_Processing/Machine_Translation/prediction/cn/gen.out.sys"
source_file = "/home/byeonggeuk/IIPL_Flitto/Text_Processing/Machine_Translation/prediction/cn/gen.out.src"
csv_file = "/home/byeonggeuk/IIPL_Flitto/Text_Processing/Machine_Translation/prediction/cn_bleu_by_sentence.csv"

# 글자 단위 토크나이즈 함수
def char_tokenize(text):
    return ' '.join(list(text.strip()))

# BLEU 객체 생성
bleu = BLEU()

# 파일 읽기
with open(ref_file, encoding="utf-8") as f:
    refs_raw = f.read().splitlines()
with open(hyp_file, encoding="utf-8") as f:
    hyps_raw = f.read().splitlines()
with open(source_file, encoding="utf-8") as f:
    source_raw = f.read().splitlines()

# 길이 검증
assert len(refs_raw) == len(hyps_raw) == len(source_raw), "문장 수가 일치하지 않습니다."

# CSV 저장 (UTF-8 with BOM)
with open(csv_file, "w", newline='', encoding="utf-8-sig") as fout:
    writer = csv.writer(fout)
    writer.writerow(["Segment", "BLEU", "Source", "Reference", "Hypothesis"])
    for i, (src, ref, hyp) in enumerate(zip(source_raw, refs_raw, hyps_raw)):
        src_tok = char_tokenize(src)
        ref_tok = char_tokenize(ref)
        hyp_tok = char_tokenize(hyp)
        score = bleu.sentence_score(hyp_tok, [ref_tok])
        writer.writerow([i, f"{score.score:.2f}", src.strip(), ref.strip(), hyp.strip()])

print(f"저장 완료 (UTF-8 with BOM): {csv_file}")