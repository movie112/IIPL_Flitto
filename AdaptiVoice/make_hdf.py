"""
INPUT TTS 결과 WAV, MFA 결과 TextGrid
OUTPUT align.hdf5, feats.hdf5
"""

import os
import argparse
import logging
import numpy as np
import h5py
import librosa
from textgrid import TextGrid
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from konlpy.tag import Okt
okt = Okt()
from opencc import OpenCC
cc = OpenCC('t2s')  # 번체 → 간체

parser = argparse.ArgumentParser()
parser.add_argument('--tts_dir', type=str)
parser.add_argument('--mfa_dir', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--lang', type=str, default='kr', help='kr, en, jp, cn')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--num_workers',type=int, default=1)
parser.add_argument('--min_char_count',type=int, default=1)
parser.add_argument("--morph", type=str, default="False", help="spliting per morph for Korean")
parser.add_argument('--noun', type=str, default="False", help="True이면 명사만 추출 (Korean only)")

args = parser.parse_args()

morph_flag = args.morph.lower() in ("true", "1", "yes", "y")
noun_flag = args.noun.lower() in ("true", "1", "yes", "y")

mfa_dir = args.mfa_dir
tts_dir = args.tts_dir

if morph_flag or noun_flag:
    mfa_dir = f"{mfa_dir}_morph"
    tts_dir = f"{tts_dir}_morph"
    
out_base_dir = args.out_dir
if morph_flag:
    out_base_dir = out_base_dir+'_morph'
if noun_flag:
    out_base_dir = out_base_dir+'_noun'
if args.min_char_count > 1:
    out_base_dir = f'{out_base_dir}_chr{args.min_char_count}'
align_path = os.path.join(out_base_dir, 'align.hdf5')    
feats_path = os.path.join(out_base_dir, 'feats.hdf5')  
os.makedirs(os.path.dirname(align_path), exist_ok=True)
os.makedirs(os.path.dirname(feats_path), exist_ok=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

def extract_features(y, sr=16000):
    """
    Kaldi 스타일로 MFCC(36) + Pitch(3) = 39차원 feature를 생성하고 CMVN 적용.
    """
    hop_length = int(0.010 * sr)
    win_length = int(0.025 * sr)
    n_fft = 2 ** int(np.ceil(np.log2(win_length)))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=36,
        n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, center=True
    )

    pitches, mags = librosa.piptrack(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length
    )
    pitch_vals = pitches.mean(axis=0)
    mag_vals = mags.mean(axis=0)
    voiced_flags = (pitches > 0).astype(float).mean(axis=0)
    pitch_features = np.vstack([pitch_vals, mag_vals, voiced_flags])

    min_frames = min(mfcc.shape[1], pitch_features.shape[1])
    if min_frames <= 2:
        raise ValueError(f"Too short audio: only {min_frames} frames")

    mfcc = mfcc[:, 1:min_frames-1]
    pitch_features = pitch_features[:, 1:min_frames-1]
    features = np.vstack([mfcc, pitch_features]).T

    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (features - mean) / std, hop_length

def normalize_text(text):
    text = cc.convert(text)  # 번체 → 간체 변환
    return text.lower()

def process_and_return(mfa_dir, session_id, wav_file, wav_dir, mode):
    tg_path = os.path.join(mfa_dir, session_id, wav_file.replace('.wav', '.TextGrid'))
    if not os.path.exists(tg_path):
        return None

    tg = TextGrid.fromFile(tg_path)
    word_tier = next((t for t in tg.tiers if 'word' in t.name.lower()), None)
    if word_tier is None:
        print('word_tier')
        return None

    all_words = [(iv.mark.strip(), iv.minTime, iv.maxTime) for iv in word_tier.intervals if iv.mark.strip()]
    if not all_words:
        print('all_words')
        return None

    # WAV 로드 및 feature 추출
    wav_path = os.path.join(wav_dir, wav_file)
    y, sr = librosa.load(wav_path, sr=16000)
    feats, hop = extract_features(y, sr)
    sec_per_frame = hop / sr
    num_frames = feats.shape[0]

    starts, ends, wids, words = [], [], [], []
    prev_end = 0
    seg_info = tg.minTime, tg.maxTime
    seg_start_frame = int(np.floor(seg_info[0] / sec_per_frame))
    seg_end_frame = int(np.ceil(seg_info[1] / sec_per_frame))

    # word 단위 segment 정보 수집
    for word, st, et in all_words:
            word_norm = normalize_text(word)
            if mode !='test':
                if len(word_norm) < args.min_char_count: # test는 vocab에서 정제
                    continue
                if  noun_flag and args.lang == 'kr':
                    pos = okt.pos(word_norm, stem=True)
                    if not (len(pos) == 1 and pos[0][1] == 'Noun'):
                        continue

            sf = max(int(np.floor(st / sec_per_frame)), prev_end + 1)
            ef = min(int(np.ceil(et / sec_per_frame)), num_frames)
            if ef <= sf:
                sf = prev_end + 1
                ef = min(sf + 1, num_frames)
            length = ef - sf
            if (mode=='train' and not (20 <= length <= 500)) or (mode =='valid' and not (50 <= length <= 500)):
                continue # test는 evaluate.py 에서 정제

            starts.append(sf)
            ends.append(ef)
            wids.append(f"{session_id}/{wav_file[:-4]}_{sf:06d}-{ef:06d}")
            words.append(word_norm)  # ✅ 정규화된 단어 저장
            prev_end = ef

    if not words:
        print('no words')
        return None

    # 그룹 이름은 세션/파일명 하나만!
    group_name = f"{session_id}/{wav_file[:-4]}"
    return group_name, seg_start_frame, seg_end_frame, feats, starts, ends, wids, words

# 처리 대상 WAV 목록 수집
total = []
for ses in os.listdir(tts_dir):
    dirp = os.path.join(tts_dir, ses)
    for wf in os.listdir(dirp):
        if wf.endswith('.wav'):
            total.append((ses, wf, dirp))
total_wav_count = len(total)

if total_wav_count == 0:
    raise RuntimeError("처리할 WAV 파일이 없습니다.")

created_groups = 0
with h5py.File(align_path, 'w') as hf_a, h5py.File(feats_path, 'w') as hf_f:
    for ses_id, wf, dirp in tqdm(total, desc='Processing WAVs'): # ?iKRNAC1I001, ?iKRNAC1I001_1.wav, /nas_homes/yeonghwa/flitto/datasets/ap_data_f/tts/test/kr/KRNAC1I001
        res = process_and_return(mfa_dir, ses_id, wf, dirp, args.mode)
        if not res:
            if args.mode != 'test':
                continue
            else:
                raise RuntimeError(f"[오류] 세그먼트 정보가 없어 그룹을 생성할 수 없습니다: {ses_id}/{wf}")
        
        grp, ssf, sef, feats, sts, ens, wids, wds = res
        
        # align.hdf5 그룹 생성
        gpa = hf_a.create_group(grp)
        gpa.create_dataset('starts', data=np.array(sts, int))
        gpa.create_dataset('ends',   data=np.array(ens, int))
        dt = h5py.string_dtype(encoding='utf-8')
        gpa.create_dataset('wids',   data=np.array(wids, object), dtype=dt)
        gpa.create_dataset('words',  data=np.array(wds,  object), dtype=dt)

        # feats.hdf5 그룹 생성
        gpf = hf_f.create_group(grp)
        gpf.create_dataset('feats', data=feats)
        gpf.create_dataset('start', data=ssf)
        gpf.create_dataset('end',   data=sef)

        created_groups += 1

print(f"✅ Total groups saved: {created_groups}")
print("✅ 모든 WAV 파일 처리 완료!")
