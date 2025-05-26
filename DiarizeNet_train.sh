#!/bin/bash

##############################
# 0. 환경 설정
##############################

# 경로 설정
CONDA_ENV_NAME="IIPL_Flitto"
ROOT="/home/byeonggeuk/IIPL_Flitto"
DATASET_ROOT="/nas_homes/byeonggeuk/dataset"

DIARIZENET_CONFIG="${ROOT}/DiarizeNet/conf/train.yaml"
DEEPVOC_CONFIG="${ROOT}/DeepVoc/config/inference.toml"
DEEPVOC_MODEL="${ROOT}/checkpoints/DeepVoc.tar"

INPUT_WAV_DIR="${ROOT}/DiarizeNet/data/demo/audio"
OUTPUT_WAV_DIR="${ROOT}/DiarizeNet/data/demo/audio"

stage=3

######################################
# 1. Conda activate
######################################
if [ $stage -le 1 ]; then
  echo "=== [STEP 1] Conda activate: ${CONDA_ENV_NAME} ==="
  source ~/.bashrc
  conda activate "${CONDA_ENV_NAME}"
fi

######################################
# 2. DeepVoc
######################################
if [ $stage -le 2 ]; then
  echo "=== [STEP 2] DeepVoc ==="
  export PYTHONPATH="${ROOT}/DeepVoc:${PYTHONPATH}"

  python -m se_module.tools.inference \
    -C "${DEEPVOC_CONFIG}" \
    -M "${DEEPVOC_MODEL}" \
    -I "${INPUT_WAV_DIR}" \
    -O "${OUTPUT_WAV_DIR}" | grep "Inference"

  unset PYTHONPATH
fi


######################################
# 3. DiarizeNet Train
######################################
if [ $stage -le 3 ]; then
  echo "=== [STEP 3] DiarizeNet Train ==="
  export PYTHONPATH="${ROOT}/DiarizeNet:${PYTHONPATH}"

  cd "${ROOT}/DiarizeNet"
  python DiarizeNet_train.py \
    --configs="${DIARIZENET_CONFIG}" \
    --gpus="0,1,2" \

  unset PYTHONPATH
  cd - > /dev/null 2>&1
fi