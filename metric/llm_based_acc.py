import re
import os
import csv
import json
from openai import OpenAI
from tqdm import tqdm

api_key = "api_key"
client = OpenAI(api_key=api_key)

with open('.../transcriptions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

predictions = {}
for audio_id, segments in data.items():
    merged_text = " ".join(segment["text"] for segment in segments)
    predictions[audio_id] = merged_text

def load_ground_truth(rttm_path):
    gt_texts = {}
    with open(rttm_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue

            audio_id = line.split()[1]

            if '|||' in line:
                before_pipe = line.split('|||', 1)[0]
                parts = before_pipe.rsplit('<NA>', 1)
                transcript = parts[1].strip() if len(parts) > 1 else before_pipe.strip()

            else:
                matches = re.findall(r'<([^<>]+)>', line)
                non_na = [m for m in matches if m.strip().upper() != 'NA']
                transcript = non_na[-1].strip() if non_na else ''

            if transcript:
                gt_texts.setdefault(audio_id, []).append(transcript)

    return {aid: " ".join(txts) for aid, txts in gt_texts.items()}

rttm_path = ".../KR.rttm"
ground_truth = load_ground_truth(rttm_path)
output_csv = ".../KR_LLM_based_acc.csv"

is_new = not os.path.exists(output_csv)
fout = open(output_csv, mode="a", newline="", encoding="utf-8")
writer = csv.writer(fout)
if is_new:
    writer.writerow(["audio_id", "score", "reference", "prediction"])
    fout.flush()

scores = []
pbar = tqdm(
    list(ground_truth.items()),
    desc="Evaluating audios",
    total=len(ground_truth),
    unit="file"
)

for audio_id, reference in pbar:
    if audio_id not in predictions:
        print(f"오디오 {audio_id}의 예측 결과가 없습니다. 건너뜁니다.")
        continue
    prediction = predictions[audio_id]

    prompt = f"""다음은 원본 전사 텍스트와 자동 음성 인식(ASR) 시스템이 생성한 텍스트입니다.
ASR 결과가 원본 전사와 얼마나 유사한지, 주어진 원본 전사를 기준으로 0-100 사이의 점수로 평가하세요.
0점은 "의미가 완전히 보존되지 않음", 100점은 "자연스럽고 원래 의미를 유지"함을 의미합니다.
약간의 표현 차이가 있더라도 자연스럽고 이해 가능한 문장이라면 이를 고려하세요.
**응답은 숫자만 출력해주세요.**
원본 전사: {reference}
ASR 시스템 출력: {prediction}
점수 (0-100): """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        resp = completion.choices[0].message.content.strip()
        match = re.search(r'(\d+(\.\d+)?)', resp)
        if not match:
            print(f"Audio ID: {audio_id} 응답에서 점수 추출 실패 → {resp}")
            continue
        score = float(match.group(1))
        scores.append(score)

        writer.writerow([audio_id, score, reference, prediction])
        fout.flush()

    except Exception as e:
        print(f"Audio ID: {audio_id} 평가 중 오류 발생: {e}")

if scores:
    avg_score = sum(scores) / len(scores)
    writer.writerow(["average", round(avg_score, 4), "", ""])
    fout.flush()
    print(f"전체 오디오 평균 평가 점수: {avg_score:.4f}")
else:
    print("평가된 오디오가 없습니다.")

fout.close()