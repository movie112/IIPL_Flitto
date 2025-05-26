# Introduction

This project aims to connect online speaker diarization with Speech-to-Text.

# Install
1. Clone this repository and navigate to IIPL_Flitto folder

```
git clone https://github.com/geuk-hub/IIPL_Flitto.git
cd IIPL_Flitto
```


2. Install Package

```
conda create --name IIPL_Flitto python=3.9
conda activate IIPL_Flitto

pip install --upgrade "pip<24.1"
cd FS-EEND && pip install -r requirements.txt
```


3. Install additional packages

```
pip install Cython librosa pesq pystoi pydub tqdm toml colorful mir_eval torch_complex "numpy<2" "accelerate<1.0.0" ffmpeg --no-deps

pip install -U openai-whisper
```


# Download checkpoints & data

Download the [pre-trained FullSubNet-plus checkpoint](https://drive.google.com/file/d/1UJSt1G0P_aXry-u79LLU_l9tCnNa2u7C/view) and input 'FullSubNet-plus/'

Download the [pre-trained FS-EEND checkpoint](https://drive.google.com/file/d/1SWANfLJldK8BpvCl_iAGmVNOBvuy6Uwd/view) and ensure the path is correctly set in 'test_from_folder' in 'FS-EEND/train_dia_Libri2Mix_infer.py'

Download the [Flitto dataset](https://drive.google.com/file/d/1nnSLZ9P3SPOZ4_w7LCYxePnSnUaX7Eqq/view) and ensure the path is correctly set in 'data_dir' in 'FS-EEND/data/Flitto/test/wav.scp'


# Path Configuration

You need to modify the following paths:
- `data_dir` in `FS-EEND/conf/spk_onl_tfm_enc_dec_nonautoreg_Libri2Mix_infer.yaml`.
- `data_dir` in `FS-EEND/data/Flitto/test/wav.scp`. (ensure it matches the Flitto dataset path)
- `save_dir_parnt` in `FS-EEND/train/oln_tfm_enc_dec.py`.
- `out_root` in `FS-EEND/visualize/gen_h5_output.py`.
- `test_from_folder` in `FS-EEND/train_dia_Libri2Mix_infer.py`. (ensure it matches the FS-EEND checkpoint path)
- `dataset_dir_list` in `FullSubNet-plus/config/inference.toml`.
- `path` in `run.sh`.


# Inference
   
```
bash run.sh
```
