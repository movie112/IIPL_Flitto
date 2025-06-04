"""
INPUT: TTS 결과 WAV, TXT
OUTPUT: TextGrid (word, phone의 timestamp)
"""
import subprocess
import os
import time
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default="kr")
parser.add_argument("--mode", type=str, default="test")
parser.add_argument("--morph", type=str, default="False")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--mfa_data_dir", type=str)
parser.add_argument("--tts_dir", type=str)
parser.add_argument("--out_dir", type=str)
args = parser.parse_args()

lang = args.lang
mode = args.mode
morph_flag = args.morph.lower() in ("true", "1", "yes", "y")

src_dir = f"{args.tts_dir}" if not morph_flag else f"{args.tts_dir}_morph"
dst_dir = f"{args.out_dir}" if not morph_flag else f"{args.out_dir}_morph"
os.makedirs(dst_dir, exist_ok=True)

session_list = sorted(os.listdir(src_dir))
total = len(session_list)
print(f"🔧 Total sessions to process: {total}")

def run_alignment(session_id):
    src_session_path = os.path.join(src_dir, session_id)
    dst_session_path = os.path.join(dst_dir, session_id)
    
    if os.path.isdir(dst_session_path) and len(os.listdir(dst_session_path)) > 0:
        return session_id, 0.0, True  # already done

    os.makedirs(dst_session_path, exist_ok=True)

    if lang == 'kr':
        cmd = f"mfa align --clean {src_session_path} {args.mfa_data_dir}/korean_mfa.dict {args.mfa_data_dir}/korean_mfa {dst_session_path}"
    elif lang == 'jp':
        cmd = f"mfa align --clean {src_session_path} {args.mfa_data_dir}/japanese_mfa.dict {args.mfa_data_dir}/japanese_mfa {dst_session_path}"
    elif lang == 'cn':
        cmd = f"mfa align --clean {src_session_path} {args.mfa_data_dir}/mandarin_china_mfa.dict {args.mfa_data_dir}/mandarin_mfa {dst_session_path}"
    elif lang == 'en':
        cmd = f"mfa align --clean {src_session_path} {args.mfa_data_dir}/english_mfa.dict {args.mfa_data_dir}/english_mfa {dst_session_path}"

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    mfa_log_path = os.path.expanduser(f"~/Documents/MFA/{session_id}")
    if os.path.exists(mfa_log_path):
        subprocess.run(f"rm -rf '{mfa_log_path}'", shell=True)
    return session_id, time.time() - start, False

time_sum, count = 0.0, 0
with tqdm(total=total, desc="MFA Align (Sequential)") as pbar:
    for session_id in session_list:
        session_id, elapsed, skipped = run_alignment(session_id)
        if not skipped:
            count += 1
            time_sum += elapsed
            avg_time = time_sum / count
            eta = avg_time * (total - count) / 60
            pbar.set_postfix({"last": f"{elapsed:.1f}s", "avg": f"{avg_time:.1f}s", "ETA": f"{eta:.1f}min"})
        else:
            pbar.write(f"✅ {session_id} skipped.")
        pbar.update(1)

print("🎉 병렬 MFA 정렬 완료")
end_time = time.time()
elapsed = end_time - start_time
with open("tts_runtime_log.txt", "a", encoding="utf-8") as f:
    f.write('make_mfa.py\n')
    f.write(f"[{lang.upper()} {mode}] Runtime: {elapsed:.2f} seconds\n")