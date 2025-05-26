#!/bin/bash

##############################
# 0. 환경 설정
##############################
export CONDA_ENV_NAME="IIPL_Flitto"

# 경로 설정
ROOT=".../IIPL_Flitto"
DIARIZENET_CONFIG="${ROOT}/DiarizeNet/conf/inference.yaml"
DIARIZENET_CHECKPOINT="/nas_homes/byeonggeuk/model/DiarizeNet/logs/KR/KR"
DEEPVOC_CONFIG="${ROOT}/DeepVoc/config/inference.toml"
DEEPVOC_MODEL="${ROOT}/checkpoints/DeepVoc.tar"

INPUT_WAV_DIR="${ROOT}/DiarizeNet/data/test_folder/wav"
OUTPUT_WAV_DIR="${ROOT}/DiarizeNet/data/test_folder"
WAV_SCP_FILE="${ROOT}/DiarizeNet/data/test_folder/wav.scp"

WHISPER_MODEL="turbo"

stage=1

######################################
# 1. Conda activate
######################################
if [ $stage -le 1 ]; then
  echo "=== [STEP 1] Conda activate: ${CONDA_ENV_NAME} ==="
  source ~/.bashrc
  conda activate "${CONDA_ENV_NAME}"

fi

######################################
# 2. DeepVoc Inference
######################################
if [ $stage -le 2 ]; then
  echo "=== [STEP 2] DeepVoc Inference ==="
  
  export PYTHONPATH="${ROOT}/DeepVoc:${ROOT}/DeepVoc/se_module:${PYTHONPATH}"

  # DeepVoc Inference
  python -m se_module.tools.inference \
    -C "${DEEPVOC_CONFIG}" \
    -M "${DEEPVOC_MODEL}" \
    -I "${INPUT_WAV_DIR}" \
    -O "${OUTPUT_WAV_DIR}" | grep "Inference"
  
  unset PYTHONPATH
fi

######################################
# 3. DiarizeNet Inference (Speaker Diarization)
######################################
if [ $stage -le 3 ]; then
  echo "=== [STEP 3] DiarizeNet Inference ==="
  
  export PYTHONPATH="${ROOT}/DiarizeNet:${PYTHONPATH}"

  cd "${ROOT}/DiarizeNet"
  python run.py \
    --wav_scp="${WAV_SCP_FILE}" \
    --configs="${DIARIZENET_CONFIG}" \
    --test_from_folder="${DIARIZENET_CHECKPOINT}" \
    --output_rttm="${ROOT}/test_folder/test.rttm"

fi

######################################
# 4. Whisper STT
######################################
if [ $stage -le 4 ]; then
  echo "=== [STEP 4] Whisper STT ==="
  
  cd "${ROOT}"
  python stt_run.py \
    --wav-scp "${WAV_SCP_FILE}" \
    --rttm-file "${ROOT}/DiarizeNet/data/test_folder/test.rttm" \
    --model-name "${WHISPER_MODEL}" \
    --language "ko" \
    --output-dir "${ROOT}/DiarizeNet/data/test_folder"
fi
