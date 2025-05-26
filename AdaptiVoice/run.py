import os
import torch
import traceback
from voice_engine.api import ToneColorConverter
from voice_engine import extractor
from TTS_engine.tts_core.api import TTS
from pydub import AudioSegment

# --- 설정 ---
ckpt_converter = '/home/byeonggeuk/IIPL_Flitto/checkpoints/AdaptiVoice/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 출력 디렉토리
output_dir = '/home/byeonggeuk/IIPL_Flitto/AdaptiVoice/demo'
os.makedirs(output_dir, exist_ok=True)

# ToneColorConverter 초기화
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# reference audio에서 speaker embedding 추출
refs = [
    '/home/byeonggeuk/IIPL_Flitto/AdaptiVoice/resources/demo_0.mp3',
    '/home/byeonggeuk/IIPL_Flitto/AdaptiVoice/resources/demo_1.mp3',
    '/home/byeonggeuk/IIPL_Flitto/AdaptiVoice/resources/demo_2.mp3',
]
available_embeddings = []
for p in refs:
    se, _ = extractor.get_se(p, tone_color_converter, vad=True)
    available_embeddings.append(se)

# base speaker embedding (source voice)
language = 'KR'
source_se = torch.load(
    f'/home/byeonggeuk/IIPL_Flitto/checkpoints/AdaptiVoice/base_speakers/ses/{language.lower()}.pth',
    map_location=device
)

# TTS 모델 초기화
tts_model = TTS(language=language, device=device)
default_speaker_id = list(tts_model.hps.data.spk2id.values())[0]
snr_path = '/home/byeonggeuk/IIPL_Flitto/AdaptiVoice/domain_normalized_snr.json'

# RTTM 파일 경로
rttm_path = '/home/byeonggeuk/IIPL_Flitto/AdaptiVoice/demo/KR.rttm'

# speaker_code → embedding 매핑
speaker_map = {}
speaker_counter = 0

# 세션별 conv 파일 경로 저장용
session_segments = {}

# RTTM 한 줄씩 읽어서 처리
with open(rttm_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line or not line.startswith('SPEAKER'):
            continue

        parts = line.split(maxsplit=10)
        if len(parts) < 11:
            print(f"RTTM 형식 오류, 건너뜀: {line}")
            continue

        session_id = parts[1]
        domain     = session_id[3:5]
        speaker_id = parts[7]
        raw_text   = parts[10].strip()
        text = raw_text[1:-1] if raw_text.startswith('<') and raw_text.endswith('>') else raw_text
        if ';' in text:
            text = text.replace(';', '.')
        # speaker embedding 할당
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = available_embeddings[speaker_counter % len(available_embeddings)]
            speaker_counter += 1
        tgt_se = speaker_map[speaker_id]

        # TTS 생성
        tts_path = os.path.join(f"{output_dir}/split", f"{idx}_{speaker_id}_tts.wav")
        try:
            tts_model.tts_to_file(
                text, default_speaker_id, tts_path,
                speed=1.0, snr_path=snr_path, domain=domain
            )
        except Exception as e:
            print(f"[{idx}] TTS 실패 ({speaker_id}): {e}")
            traceback.print_exc()
            continue

        # 생성된 TTS 오디오 길이 확인
        try:
            audio = AudioSegment.from_wav(tts_path)
            if len(audio) < 100:
                print(f"[{idx}] TTS 너무 짧음 ({len(audio)}ms), skip")
                continue
        except Exception as e:
            print(f"[{idx}] TTS 로드 실패: {e}")
            continue

        # tone color 변환
        conv_path = os.path.join(f"{output_dir}/split", f"{idx}_{speaker_id}_conv.wav")
        try:
            tone_color_converter.convert(
                audio_src_path=tts_path,
                src_se=source_se,
                tgt_se=tgt_se,
                output_path=conv_path,
                message="@MyShell"
            )
        except Exception as e:
            print(f"[{idx}] tone conversion 실패: {e}")
            continue

        print(f"[{idx}] 완료: {conv_path}")

        # 세션별 리스트에 추가
        session_segments.setdefault(session_id, []).append(conv_path)

# --- 세션별로 conv.wav들을 합쳐서 최종 파일 생성 ---
for session_id, seg_files in session_segments.items():
    if not seg_files:
        print(f"세션 {session_id}에 유효한 세그먼트 없음, 건너뜀")
        continue

    combined = None
    for path in seg_files:
        seg = AudioSegment.from_wav(path)
        combined = seg if combined is None else (combined + seg)

    final_path = os.path.join(output_dir, f"{session_id}.wav")
    combined.export(final_path, format="wav")
    print(f"세션 {session_id} 최종 오디오 저장: {final_path}")
