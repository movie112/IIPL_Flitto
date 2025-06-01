import re
import os
import csv
import json
import argparse
from openai import OpenAI
from tqdm import tqdm

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

def load_predictions(pred_path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    predictions = {}
    for audio_id, segments in data.items():
        merged_text = " ".join(segment["text"] for segment in segments)
        predictions[audio_id] = merged_text
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 기반 ASR 정확도 평가 스크립트")
    parser.add_argument(
        "--api_key", type=str, required=True,
        help="OpenAI API 키"
    )
    parser.add_argument(
        "--rttm", type=str, required=True,
        help="Ground truth RTTM 파일 경로"
    )
    parser.add_argument(
        "--pred", type=str, required=True,
        help="ASR 예측 결과 JSON 파일 경로"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="평가 결과를 저장할 CSV 파일 경로"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="사용할 LLM 모델 이름 (기본: gpt-4o)"
    )
    args = parser.parse_args()

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=args.api_key)

    # 예측 결과 로드
    predictions = load_predictions(args.pred)
    # Ground truth 로드
    ground_truth = load_ground_truth(args.rttm)

    # 출력 CSV 설정
    is_new = not os.path.exists(args.output)
    fout = open(args.output, mode="a", newline="", encoding="utf-8")
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
                model=args.model,
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
