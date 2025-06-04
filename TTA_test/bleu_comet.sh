# bleu_comet.sh
#!/bin/bash

langs=("en" "cn" "jp")
root="/path/to/your/IIPL_Flitto"
dataset_root="${root}/Text_Processing/Machine_Translation/data"
test_csv_dir="${root}/TTA_test/bleu_comet_data"
result_base="${root}/TTA_test/bleu_comet_data/prediction"
ckpt_dir="${root}/checkpoints/Machine_Translation_ckpt"
code_dir="${root}/Text_Processing/Machine_Translation"
metric_dir="${root}/metric"

for lang in "${langs[@]}"; do
  echo -e "\n[1] Making test set for ${lang}"
  python ${code_dir}/iipl_1_make_test_datset.py \
    "${test_csv_dir}/${lang}.csv" \
    "${dataset_root}/spm_ko_${lang}" \
    --src_lang ko --tgt_lang ${lang}

  echo -e "\n[2] Tokenizing test set (SPM) for ${lang}"
  python ${code_dir}/iipl_2_make_spm_test_model.py \
    "${dataset_root}/spm_ko_${lang}" \
    --src_lang ko --tgt_lang ${lang} \
    --spm_src "${dataset_root}/spm/${lang}/spm_ko.model" \
    --spm_tgt "${dataset_root}/spm/${lang}/spm_${lang}.model"

  echo -e "\n[3] Preprocessing fairseq binary input for ${lang}"
  bash ${code_dir}/iip_3_preprocess_fairseq_test.sh \
    ko ${lang} \
    "${dataset_root}/spm_ko_${lang}/test_spm" \
    "${dataset_root}/spm_ko_${lang}" \
    "${dataset_root}/spm/${lang}" \

  echo -e "\n[4] fairseq-generate inference for ko→${lang}"
  model_dir="${ckpt_dir}/real_ckp_ko_${lang}"
  data_path="${dataset_root}/spm_ko_${lang}"
  result_path="${result_base}/${lang}"
  mkdir -p "$result_path"

  CUDA_VISIBLE_DEVICES=1 fairseq-generate "$data_path" \
    --path "${model_dir}/checkpoint_best.pt" \
    --source-lang ko --target-lang ${lang} \
    --beam 5 --task translation --gen-subset test \
    --remove-bpe sentencepiece \
    --skip-invalid-size-inputs-valid-test \
    --required-batch-size-multiple 1 \
    --max-sentences 128 --batch-size 1 \
    --results-path "$result_path"

  echo " Extracting results for ${lang}..."
  grep '^H' "${result_path}/generate-test.txt" | cut -f3- > "${result_path}/gen.out.sys"
  grep '^T' "${result_path}/generate-test.txt" | cut -f2- > "${result_path}/gen.out.ref"
  grep '^S' "${result_path}/generate-test.txt" | cut -f2- > "${result_path}/gen.out.src"
done

echo " Results extracted for ${lang}."
echo -e "\n[5] Calculating BLEU for ${lang}"
python ${metric_dir}/bleu_comet.py \
  "${result_base}"