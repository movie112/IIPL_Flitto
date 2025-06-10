## Introduction

IIPL_Flitto is a comprehensive speech and text processing toolkit. This repository provides a collection of modules and scripts for advanced speaker diarization, speech enhancement modeling, speech-to-text (STT), text-to-speech (TTS) and text processing. The toolkit is designed to facilitate research and development in automatic speech recognition (ASR), speaker identification, and natural language processing (NLP) tasks.

## Environment

- **OS**: Ubuntu 24.04.2
- **CUDA Toolkit**: 11.7
- **GPU Driver**: NVIDIA-SMI 570.144 (CUDA Version 12.8)

## Install
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
cd DiarizeNet && pip install -r requirements.txt
```


3. Install additional packages

```
pip install Cython librosa pesq pystoi pydub tqdm toml colorful mir_eval torch_complex "numpy<2" "accelerate<1.0.0" ffmpeg --no-deps jieba Mecab pkuseg

conda install -c conda-forge compilers
pip install pkuseg nlptutti transformers soynlp

pip install -U openai-whisper
pip install openai
```


## Download checkpoints & data

Download the [TTA Test Dataset(wer/cer/llm-based acc)](https://www.dropbox.com/scl/fi/zeps24kl7rgugpjdi9yqd/TTA_test_wer_cer_llm_acc.zip?rlkey=xdjxdvfgye4wjjyix1i4ot5rf&st=34rxda79&dl=0).

Download the [DiarizeNet Model checkpoint](https://www.dropbox.com/scl/fo/uyer0669wfhpvm055v5mf/ACbFAIbVxQbScEPlhhioL0A?rlkey=0hndtmi059oh2r5bh51i0q1op&st=ix16crxu&dl=0).

Download the [AdaptiVoice Model checkpoint](https://www.dropbox.com/scl/fi/90fylcpd5vva1cjsbkxcv/AdaptiVoice_ckpt.zip?rlkey=lg6osh4cf1dbnz7tqv8y733lz&st=rn3gsdxl&dl=0).

Download the [Crossview-AP Model checkpoint](https://www.dropbox.com/scl/fi/nux8aqzrq5db59g8144t3/Crossview_AP_ckpt.zip?rlkey=nink2rrdvssj5p6pbsfijirp2&st=0su6g0a5&dl=0).

Download the [crossview_ap_data](https://www.dropbox.com/scl/fi/t15wnvy008ylmnxk0g8gy/crossviewap_data.zip?rlkey=891f5cwz4aecjtj630ef1d9xh&st=h9or20oa&dl=0).

Download the [Machine Translation Model checkpoint](https://www.dropbox.com/scl/fo/3xle2g3505iydwbw6yqg7/APcyGLXHwL83A2Y3Lu_GaZU?rlkey=i36di9snedlj45vttk6nd0zw9&st=sdhgg06z&dl=0).

Download the [Error Correction Model checkpoint](https://www.dropbox.com/scl/fo/rsl0xailbxcoeiz1ebf5g/AOh-MttVZHLOsO8BH7dc7ZA?rlkey=lta539u6qrqovke5ndodtfsmu&st=3xh1n9xr&dl=0).

## TTA_test: WER/CER/LLM-based acc

Before running the following script, make sure to configure the following environment variables:

- **LANG**: Choose one language from KR (Korean), EN (English), CN (Chinese), or JP (Japanese).
- **ROOT**: Set this to the full path of your `IIPL_Flitto` repository.
- **DIARIZENET_CHECKPOINT**: Put the downloaded DiarizeNet checkpoint into the `IIPL_Flitto/checkpoints` directory.
- **OPENAI_API_KEY**: Provide your OpenAI API key

- **TTA Test Dataset**: Put your TTA Test Dataset files into `TTA_test/wer_cer_llm_based_acc_data` folder.
  - For example:  
    - `TTA_test/wer_cer_llm_based_acc_data/KR`  
    - `TTA_test/wer_cer_llm_based_acc_data/CN`  
    - `TTA_test/wer_cer_llm_based_acc_data/EN`  
    - `TTA_test/wer_cer_llm_based_acc_data/JP`

- **{LANG}_wav.scp**: Ensure that the {LANG}_wav.scp file inside TTA_test/wer_cer_llm_based_acc_data/{LANG} contains correct audio paths.
  - If you need to update or replace the `/path/to/your` prefixes in any `{LANG}_wav.scp` file, you can run the `change_path.py` script located at:
    ```
    IIPL_Flitto/TTA_test/wer_cer_llm_based_acc_data/change_path.py
    ```
    This script will automatically replace `/path/to/your` with your specified `root` path..


```
bash TTA_test/wer_cer_llm_based_acc.sh
```

## TTA_test: crossview-AP

1. Install Pakages 1

1-1. Create `ap_env` environment

```
conda create -n ap_env python=3.9
conda activate ap_env
```

1-2. Install packages

```
conda install -c conda-forge gxx_linux-64
cd IIPL_Flitto/AdaptiVoice/TTS_engine
pip install -e .
cd IIPL_Flitto/AdaptiVoice/voice_engine
pip install -e .
```

1-3. Install additional packages

```
conda install -c conda-forge ffmpeg
pip install mecab-python3
python -m unidic download
pip install pkuseg janome konlpy h5py textgrid tgt opencc librosa
```

2. Install Packages 2

2-1. Create `mfa_env` environment

```
conda create -n mfa_env -c conda-forge montreal-forced-aligner
conda activate mfa_env
```

2-2. Install packages

```
pip install python-mecab-ko jamo spacy-pkuseg dragonmapper hanziconv textgrid tgt
conda install -c conda-forge spacy sudachipy sudachidict-core
```

3. Run

Before running the following script, make sure to configure the following environment variables:
- **root**: Set this to the full path of your `IIPL_Flitto` repository.
- **lang**: Choose one language from kr (Korean), en (English), cn (Chinese), or jp (Japanes).
- **AdaptiVoice_ckpt**: Put the downloaded AdaptiVoice model checkpoint into the `IIPL_Flitto/checkpoints` directory.
- **Crossview_AP_ckpt**: Put the downloaded Crossview-AP model checkpoint into the `IIPL_Flitto/checkpoints` directory.
- **crossview_ap_data**: Put the downloaded crossview_ap_data into the `IIPL_Flitto/TTA_test/crossview_ap_data` directory.
  
3-1. evaluation only
- input: align.hdf5, feats.hdf5, vocab

```
bash TTA_test/crossview_ap_eval_only.sh
```

3-2. evaluation from scratch
- input: rttm
- data generation: rttm -> tts -> timestamp -> vocab -> hdf5

```
bash TTA_test/crossview_ap_from_scratch.sh
```

## TTA_test: BLEU/COMET

1. Install Package

```
conda create -n mt python=3.9
conda activate mt

cd IIPL_Flitto/Text_Processing/Machine_Translation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy regex sacrebleu tensorboard matplotlib pandas cython setuptools pyarrow sacremoses tensorboardX unbabel-comet
pip install pip==23.3.1
conda install -c conda-forge gxx_linux-64
pip install --editable ./
```

2. Run

Before running the following script, make sure to configure the following environment variables:

- **root**: Set this to the full path of your `IIPL_Flitto` repository.
- **Machine_Translation_ckpt**: Put the downloaded Machine Translation checkpoint into the `IIPL_Flitto/checkpoints` directory.
  
```
bash TTA_test/bleu_comet.sh
```

### Error Correction

1. Install Package

```
conda create -n ec python=3.12
conda activate ec

pip install unsloth hgtk
```

2. Run

Before running the following script, make sure to configure the following environment variables:

- **ROOT**: Set this to the full path of your `IIPL_Flitto` repository.
- **MODEL_PATH**: Set this to the full path of your `Error Correction` checkpoints folder.
  
```
python Text_Processing/Error_Correction/LLM_grammer_inference.py
```
