#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

# ===== 설정 =====
device_num=0
mode="test"

root_dir="/path/to/your/IIPL_Flitto"
save_dir="${root_dir}/TTA_test/crossview_ap_data"
data_dir="${root_dir}/metric/Crossview-AP/data"
eval_path="${root_dir}/metric/Crossview-AP/code/evaluate.py"

# ===== 언어 리스트 =====
langs=("en" "cn" "kr" "jp")

for lang in "${langs[@]}"; do
  echo "======================================="
  echo "▶ Evaluation for ${lang^^}"
  echo "======================================="
  conda activate ap_env

  config_path="$data_dir/config/eval_only_config/config_$lang.json"
  temp_config="${config_path}.tmp"
  sed "s|/path/to/your/IIPL_Flitto|$root_dir|g" "$config_path" > "$temp_config"
  python "$eval_path" \
  --config "$temp_config" \
  --result_dir "$save_dir/eval_only_result/$lang"
  rm "$temp_config"

  echo "✅ Done for ${lang^^}"
  echo ""
done