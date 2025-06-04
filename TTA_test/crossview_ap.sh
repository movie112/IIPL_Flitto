#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
# ===== 설정 =====
device_num=0
mode="test"

root_dir="/path/to/your/IIPL_Flitto"
lang="kr"

tts_dir="${root_dir}/AdaptiVoice"
save_dir="${root_dir}/TTA_test/crossview_ap_data"
data_dir="${root_dir}/metric/Crossview-AP/data"
eval_path="${root_dir}/metric/Crossview-AP/code/evaluate.py"
ckpt_dir="${root_dir}/checkpoints"

echo "======================================="
echo "▶ Running TTS generation for ${lang^^}"
echo "======================================="
conda activate ap_env
python -u "$tts_dir/make_audio.py" \
  --lang "$lang" \
  --mode "$mode" \
  --device_num "$device_num" \
  --data_dir "$data_dir" \
  --ckpt_dir "$ckpt_dir" \
  --out_dir "$save_dir/tts_$mode/$lang" \
  --refspk_path "$data_dir/wav/reference.wav" \
  --rttm_path "${save_dir}/${lang^^}.rttm" \

echo "Done for ${lang^^}"
echo ""

echo "======================================="
echo "▶ Running MFA generation for ${lang^^}"
echo "======================================="
conda activate mfa_env
python "$tts_dir/make_mfa.py" \
  --lang "$lang" \
  --mode "$mode" \
  --data_dir "$data_dir" \
  --mfa_data_dir "$data_dir/mfa_data" \
  --tts_dir "$save_dir/tts_$mode/$lang" \
  --out_dir "$save_dir/mfa_$mode/$lang" \

echo "Done for ${lang^^}"
echo ""

echo "======================================="
echo "▶ Running Vocab generation for ${lang^^}"
echo "======================================="
conda activate ap_env
python "$tts_dir/make_vocab.py" \
  --lang "$lang" \
  --mode "$mode" \
  --data_dir "$data_dir" \
  --mfa_dir "$save_dir/mfa_$mode/$lang" \
  --out_dir "$save_dir/vocab_$mode/$lang" \

echo "Done for ${lang^^}"
echo ""

echo "======================================="
echo "▶ Running align.hdf5, feats.hdf5 generation for ${lang^^}"
echo "======================================="

python "$tts_dir/make_hdf.py" \
  --lang "$lang" \
  --mode "$mode" \
  --mfa_dir "$save_dir/mfa_$mode/$lang" \
  --tts_dir "$save_dir/tts_$mode/$lang" \
  --out_dir "$save_dir/hdf_$mode/$lang" \

echo "Done for ${lang^^}"
echo ""


conda run -n ap_env python "$eval_path" \
  --config "$data_dir/config/config_$lang.json" \
  --result_dir "$save_dir/result/$lang" \

echo "Done for ${lang^^}"
echo ""