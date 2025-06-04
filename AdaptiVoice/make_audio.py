'''
INPUT: RTTM
OUTPUT: WAV, TXT (TTS result)
'''
import os
import torch
import re 
import random
import time
from tqdm import tqdm
import pkuseg
from janome.tokenizer import Tokenizer
import nltk
import argparse
from konlpy.tag import Okt
import sys
from voice_engine.api import ToneColorConverter
from voice_engine import se_extractor
from TTS_engine.tts_core.api import TTS
from pydub import AudioSegment
print("argv received:", sys.argv)
start_time = time.time()
okt = Okt()
nltk.download('averaged_perceptron_tagger_eng') 

parser = argparse.ArgumentParser(
    description="Generate hdf5 files for cross-view AP evaluation from audio and forced alignment transcript."
)
parser.add_argument("--lang", type=str, help="kr, en, jp, cn")
parser.add_argument("--mode", type=str, help="train, valid, test")
parser.add_argument("--device_num", type=int, default=1)
parser.add_argument("--morph", type=str, default="False", help="spliting per morph for Korean")
parser.add_argument("--out_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--refspk_path", type=str)
parser.add_argument("--rttm_path", type=str) 
args = parser.parse_args()

lang = args.lang
mode = args.mode
rttm_path = args.rttm_path
morph_flag = args.morph.lower() in ("true", "1", "yes", "y")
# output dir
output_dir = args.out_dir
os.makedirs(output_dir, exist_ok=True)

# data maximum
if mode == 'train':
    end_length = 32 * 60 * 60 * 1000 # 32H
elif mode == 'valid':
    end_length = 3 * 60 * 60 * 1000 # 3H
else: 
    end_length = 32 * 60 * 60 * 1000 # 32H

# 특수문자 정제 함수
def remove_special_chars(text, lang="en"):
    if lang == "en":
        return re.sub(r"[^\w\s'\-\.]", "", text)
    elif lang == "ko":
        return re.sub(r"[^\w\s\uac00-\ud7a3]", "", text)
    elif lang == "jp":
        return re.sub(r"[^\w\s\u3040-\u30ff\u4e00-\u9fff・ー「」]", "", text)
    elif lang == "cn":
        return re.sub(r"[^\w\s\u4e00-\u9fff、·《》]", "", text)
    else:
        return re.sub(r"[^\w\s]", "", text)
    
if lang == 'kr' and morph_flag:
    output_dir = f'{output_dir}_morph'
    os.makedirs(output_dir, exist_ok=True)

# tokenizer
if lang == "cn":
    tokenizer = pkuseg.pkuseg()
elif lang == 'jp':
    tokenizer = Tokenizer()
elif lang == 'kr':
    tokenzier = Okt()

device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(args.device_num) 
random.seed(42)

# TTS, ToneConverter
source_se = torch.load(f'{args.data_dir}/tone/base_speakers/ses/{lang}.pth', map_location=device)
ckpt_converter = f'{args.data_dir}/tone/converter'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
target_se, _ = se_extractor.get_se(args.refspk_path, tone_color_converter, vad=True)
if lang == 'cn':
    lang = 'zh'  # TTS expects 'zh' for Chinese
tts_model = TTS(language=lang.upper(), device=device)
default_speaker_id = list(tts_model.hps.data.spk2id.values())[0]
if lang == 'zh':
    lang = 'cn'  # TTS expects 'cn' for Chinese
    
# tmp wav: 데이터 동시 생성 시 겹치지 않게 주의
src_path = f"{args.data_dir}/wav/{lang}_{mode}.wav"
if morph_flag:
    src_path = f"{args.data_dir}/wav/{lang}_{mode}_morph.wav"

def extract_text_from_rttm(file_path, lang, mode):
    all_texts = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Extracting text from {lang.upper()} RTTM"):
            parts = line.strip().split()
            session_id = parts[1]

            if session_id not in all_texts:
                all_texts[session_id] = []
            if lang == 'kr':
                text = " ".join(parts[8:]).split("|||")[0]
            else:
                text = " ".join(parts[10:]).replace("<", "").replace(">", "")

            text = text.strip()
            text = re.sub(r"\s+", " ", text)

            all_texts[session_id].append(text)
    return all_texts

result = extract_text_from_rttm(rttm_path, lang, mode)

full_length = 0
for session_id, texts in tqdm(result.items(), total=len(result), desc=f"Processing sessions"):
    print(f"== {session_id} ==")
    os.makedirs(f"{output_dir}/{session_id}", exist_ok=True)

    i = 0
    for text in texts:
        i += 1
        # TTS
        try: 
            tts_model.tts_to_file(text, default_speaker_id, src_path, speed=1.0)
        except Exception as e:
            print(f"Error generating audio for text '{text}': {e}")
            continue
        
        try:
            tts_audio = AudioSegment.from_wav(src_path)
        except:
            print(f"Error loading audio for text '{text}': {e}")
            continue

        output_path = f"{output_dir}/{session_id}/{session_id}_{i}.wav"
        txt_path = f"{output_dir}/{session_id}/{session_id}_{i}.txt"

        # Tone converter (openvoice)
        try:
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
            )
            print(f"**SAVED: {output_path}")
            with open(txt_path, "w", encoding="utf-8") as f:
                if lang == 'cn':
                    words = tokenizer.cut(text)
                    text = " ".join(words)
                elif lang == 'jp':
                    words = [token.surface for token in tokenizer.tokenize(text)]
                    text = " ".join(words)
                elif lang == 'kr' and morph_flag:
                    words = tokenzier.morphs(text)
                    text = " ".join(words)
                cleaned_text = remove_special_chars(text, lang)
                f.write(cleaned_text)

            print(f"**SAVED: {text} to {txt_path}")
        except Exception as e:
            print(f"Error converting audio for text '{text}': {e}")
            continue
        
        full_length += len(tts_audio)
        print(f"Audio length: {full_length} ms / {end_length} ms")

    if full_length > end_length:
        print(f"Full length exceeded {end_length} ms. Stopping.")
        break

end_time = time.time()
elapsed = end_time - start_time
with open("tts_runtime_log.txt", "a", encoding="utf-8") as f:
    f.write('make_audio.py\n')
    f.write(f"[{lang.upper()} {mode}] Runtime: {elapsed:.2f} seconds\n")