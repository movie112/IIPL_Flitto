# Introduction

IIPL_Flitto is a comprehensive speech and text processing toolkit. This repository provides a collection of modules and scripts for advanced speaker diarization, speech enhancement modeling, speech-to-text (STT), text-to-speech (TTS) and text processing. The toolkit is designed to facilitate research and development in automatic speech recognition (ASR), speaker identification, and natural language processing (NLP) tasks.

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
cd DiarizeNet && pip install -r requirements.txt
```


3. Install additional packages

```
pip install Cython librosa pesq pystoi pydub tqdm toml colorful mir_eval torch_complex "numpy<2" "accelerate<1.0.0" ffmpeg --no-deps

pip install -U openai-whisper
```


# Download checkpoints

Download the [pre-trained DiarizeNet checkpoint](https://www.dropbox.com/scl/fo/uyer0669wfhpvm055v5mf/ACbFAIbVxQbScEPlhhioL0A?rlkey=0hndtmi059oh2r5bh51i0q1op&st=ix16crxu&dl=0).

Download the [Machine Translation Model checkpoint](https://www.dropbox.com/scl/fo/3xle2g3505iydwbw6yqg7/APcyGLXHwL83A2Y3Lu_GaZU?rlkey=i36di9snedlj45vttk6nd0zw9&st=sdhgg06z&dl=0).

# DeepVoc+DiarizeNet+STT
   
```
bash run.sh
```

# Metric

```
python /metric/wer_cer.py
python /metric/llm_based_acc.py
```

# AdaptiVoice

1. Install Package

```
conda create -n adaptivoice python=3.9
conda activate adaptivoice

cd adaptivoice
pip install -e .
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

2. run
```
python /AdaptiVoice/run.py
```

# Machine Translation

1. Install Package

```
conda create -n mt python=3.9
conda activate mt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy regex sacrebleu tensorboard matplotlib pandas cython setuptools
pip install pip==23.3.1
conda install -c conda-forge gxx_linux-64
conda install -c nvidia cuda-toolkit=11.8 cudatoolkit-dev=11.8
```

2. metric
```
bash /metric/bleu_comet.sh
```