#!/bin/bash

##############################
# 0. 환경 설정
##############################
export CONDA_ENV_NAME="IIPL_Flitto"

LANG="KR"  # 언어 설정 (KR, EN, CN, JP)
ROOT="/path/to/your/IIPL_Flitto"
OPENAI_API_KEY="your_openai_api_key"
DIARIZENET_CHECKPOINT="${ROOT}/checkpoints/DiarizeNet_ckpt_${LANG}"

DIARIZENET_CONFIG="${ROOT}/DiarizeNet/conf/inference.yaml"
DEEPVOC_CONFIG="${ROOT}/DeepVoc/config/inference.toml"
DEEPVOC_MODEL="${ROOT}/checkpoints/DeepVoc.tar"

INPUT_DIR="${ROOT}/TTA_test/wer_cer_llm_based_acc_data/${LANG}/audio"
OUTPUT_DIR="${ROOT}/TTA_test/wer_cer_llm_based_acc_data/${LANG}"
WAV_SCP_FILE="${ROOT}/TTA_test/wer_cer_llm_based_acc_data/${LANG}/${LANG}_wav.scp"

WHISPER_MODEL="large-v3"

stage=1

######################################
# 1. DeepVoc Inference
######################################
if [ $stage -le 1 ]; then
  echo "=== [STEP 1] DeepVoc Inference ==="
  
  export PYTHONPATH="${ROOT}/DeepVoc:${ROOT}/DeepVoc/se_module:${PYTHONPATH}"

  # DeepVoc Inference
  python -m se_module.tools.inference \
    -C "${DEEPVOC_CONFIG}" \
    -M "${DEEPVOC_MODEL}" \
    -I "${INPUT_DIR}" \
    -O "${OUTPUT_DIR}" | grep "Inference"
  
  unset PYTHONPATH
fi

######################################
# 2. DiarizeNet Inference
######################################
if [ $stage -le 2 ]; then
  echo "=== [STEP 2] DiarizeNet Inference ==="
  
  export PYTHONPATH="${ROOT}/DiarizeNet:${PYTHONPATH}"

  cd "${ROOT}/DiarizeNet"
  python run.py \
    --wav_scp="${WAV_SCP_FILE}" \
    --configs="${DIARIZENET_CONFIG}" \
    --test_from_folder="${DIARIZENET_CHECKPOINT}" \
    --output_rttm="${OUTPUT_DIR}/test.rttm"
fi

######################################
# 3. Whisper STT
######################################
if [ $stage -le 3 ]; then
  echo "=== [STEP 3] Whisper STT ==="
  
  cd "${ROOT}"
  python stt_run.py \
    --wav-scp "${WAV_SCP_FILE}" \
    --rttm-file "${OUTPUT_DIR}/test.rttm" \
    --model-name "${WHISPER_MODEL}" \
    --language "${LANG}" \
    --output-dir "${OUTPUT_DIR}"
fi

######################################
# 4. WER/CER 평가
######################################
if [ $stage -le 4 ]; then
  echo "=== [STEP 4] WER/CER 평가 ==="

  if [ "${LANG}" = "KR" ]; then
    python "${ROOT}/metric/wer_cer_jamo.py" \
      --lang_code "${LANG}" \
      --ref "${OUTPUT_DIR}/${LANG}.rttm" \
      --pred "${OUTPUT_DIR}/transcriptions.json" \
      --output "${OUTPUT_DIR}/wer_cer_results.csv"
  else
    python "${ROOT}/metric/wer_cer.py" \
      --lang_code "${LANG}" \
      --ref "${OUTPUT_DIR}/${LANG}.rttm" \
      --pred "${OUTPUT_DIR}/transcriptions.json" \
      --output "${OUTPUT_DIR}/wer_cer_results.csv"
  fi
fi

######################################
# 5. LLM 기반 정확도 평가
######################################
if [ $stage -le 5 ]; then
  echo "=== [STEP 5] LLM 기반 정확도 평가 ==="

  python "${ROOT}/metric/llm_based_acc.py" \
    --api_key "${OPENAI_API_KEY}" \
    --rttm "${OUTPUT_DIR}/${LANG}.rttm" \
    --pred "${OUTPUT_DIR}/transcriptions.json" \
    --output "${OUTPUT_DIR}/llm_based_acc_results.csv" \
    --model "gpt-4o"
fi
