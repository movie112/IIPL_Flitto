#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
from collections import defaultdict

import numpy as np
from pydub import AudioSegment
import whisper
import torch
from tqdm import tqdm


def audio_segment_to_numpy(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples = samples.astype(np.float32) / (1 << (8 * audio_segment.sample_width - 1))
    return samples


def load_wav_scp(wav_scp_path):
    mapping = {}
    with open(wav_scp_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            utt_id, wav_path = parts[0], parts[1]
            mapping[utt_id] = wav_path
    return mapping


def load_rttm(rttm_path):
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                utt_id = parts[1]
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({
                    "utt_id": utt_id,
                    "start": start,
                    "duration": duration,
                    "speaker": speaker,
                })
            except Exception as e:
                print(f"RTTM 파싱 오류: {e} in line: {line}")
                continue
    return segments


def group_segments_by_utt(segments):
    grouped = defaultdict(list)
    for seg in segments:
        grouped[seg["utt_id"]].append(seg)
    return grouped


def transcribe_audio_segments(wav_mapping, segments_by_utt, model_name="large-v3", language="ko"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_name).to(device)

    all_transcriptions = {}

    for utt_id, segments in tqdm(segments_by_utt.items(), desc="Utterances", unit="utt"):
        if utt_id not in wav_mapping:
            print(f"[WARN] wav.scp에 {utt_id}가 없습니다. 스킵합니다.")
            continue

        wav_path = wav_mapping[utt_id]
        if not os.path.exists(wav_path):
            print(f"[WARN] 파일이 존재하지 않습니다: {wav_path}")
            continue

        try:
            audio = AudioSegment.from_file(wav_path)
        except Exception as e:
            print(f"[ERROR] {wav_path} 로딩 실패: {e}")
            continue

        transcriptions = []
        for seg in segments:
            segment = audio[int(seg["start"]*1000):int((seg["start"]+seg["duration"])*1000)]
            segment_numpy = audio_segment_to_numpy(segment)

            try:
                result = model.transcribe(segment_numpy, fp16=torch.cuda.is_available(), language=language)
                text = result.get("text", "").strip()
            except Exception as e:
                print(f"[ERROR] STT 실패: {e}")
                text = ""
            transcriptions.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "duration": seg["duration"],
                "text": text
            })

        all_transcriptions[utt_id] = transcriptions

        stream_path = os.path.join(output_dir, "transcriptions_stream.jsonl")
        with open(stream_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps({utt_id: transcriptions}, ensure_ascii=False) + "\n")
            fout.flush()

    return all_transcriptions


def main():
    parser = argparse.ArgumentParser(
        description="wav.scp와 RTTM 파일을 기반으로 Whisper STT 실행"
    )
    parser.add_argument("--wav-scp", type=str, default=".../KR_wav.scp", help="wav.scp 파일 경로")
    parser.add_argument("--rttm-file", type=str, default=".../KR.rttm", help="RTTM 파일 경로")
    parser.add_argument("--model-name", type=str, default="large-v3", help="Whisper 모델 이름")
    parser.add_argument("--language", type=str, default="ko", help="언어 코드 (한국어: ko, 영어: en, 중국어: zh, 일본어: ja)")
    parser.add_argument("--output-dir", type=str, default=".../test", help="STT 결과 저장 디렉토리")
    args = parser.parse_args()

    global output_dir
    output_dir = args.output_dir
    
    wav_mapping = load_wav_scp(args.wav_scp)
    if not wav_mapping:
        print("wav.scp 파일을 읽을 수 없거나 내용이 없습니다.")
        return

    segments = load_rttm(args.rttm_file)
    if not segments:
        print("RTTM 파일을 읽을 수 없거나 내용이 없습니다.")
        return

    segments_by_utt = group_segments_by_utt(segments)
    print(f"총 {len(segments_by_utt)}개의 utterance 처리 시작...")

    os.makedirs(output_dir, exist_ok=True)
    open(os.path.join(output_dir, "transcriptions_stream.jsonl"), "w", encoding="utf-8").close()

    all_transcriptions = transcribe_audio_segments(wav_mapping, segments_by_utt, model_name=args.model_name, language=args.language)

    final_path = os.path.join(output_dir, "transcriptions.json")
    try:
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(all_transcriptions, f, ensure_ascii=False, indent=2)
        print(f"STT 결과가 {final_path} 에 저장되었습니다.")
    except Exception as e:
        print(f"[ERROR] 결과 저장 실패: {e}")


if __name__ == "__main__":
    main()
