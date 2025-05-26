from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizerFast, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
import pandas as pd
import os, torch, random, numpy as np
import hgtk
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CFG = {
    'MODEL_NAME': "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    'LEARNING_RATE': 3e-4,
    'STEPS': 10000,
    'SEED': 42
}

seed_everything(CFG['SEED'])

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["MODEL_NAME"],
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=CFG["SEED"],
    use_rslora=False,
    loftq_config=None,
)

train = pd.read_csv("../IIPL_Flitto/Text_Processing/Error_Correction/train.csv", encoding="utf-8-sig")
test = pd.read_csv("../IIPL_Flitto/Text_Processing/Error_Correction/test.csv", encoding = 'utf-8-sig')
train = train.drop(columns="ID")

def jamo_formatting(examples):
    inputs, outputs = [], []
    for i, o in zip(examples["input"], examples["output"]):
        inputs.append(str(i))
        outputs.append(str(o))
    return {"input": inputs, "output": outputs}

llama_dataset = jamo_formatting(train)
llama_dataset_prompt_map = Dataset.from_dict(llama_dataset)

alpaca_prompt =''' You are a Korean language correction expert.

Your task is to fix ONLY grammar errors, spacing mistakes, and transcription issues caused by spoken Korean.  
DO NOT generate or change sentence endings such as "~요", "~입니다", or any polite endings.  
DO NOT add words or modify meaning.  
Fix only what is wrong in the original sentence. Return the corrected sentence **in the same tone, formality, and length** as the input.

Examples:
- Input: "몸에 죠은 홍삼"
  → Output: "몸에 좋은 홍삼"

- Input: "전공의 선발 등 직무와 관련하여 부당하게 금품수수 행위"
  → Output: "전공의 선발 등 직무와 관련하여 부당하게 금품 수수 행위"

---

### Instruction:
{}

---

### Response:
{}
'''

def formatting_prompts_func(examples):
    return {"text": [alpaca_prompt.format(i, o) + "</s>" for i, o in zip(examples["input"], examples["output"])]}

dataset = llama_dataset_prompt_map.map(formatting_prompts_func, batched=True)

tokenizer.padding_side = "right"
checkpoint_dir = "..."
trainer = SFTTrainer(
    model=model,  
    tokenizer=tokenizer,  
    train_dataset=dataset, 
    dataset_text_field="text", 
    max_seq_length=4096,  
    dataset_num_proc=8,
    packing=False,  
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,  
        max_steps=10000,
        do_eval=False,
        logging_steps=10,  
        learning_rate=2e-4, 
        fp16=not torch.cuda.is_bf16_supported(),  
        bf16=torch.cuda.is_bf16_supported(),  
        optim="adamw_8bit",  
        weight_decay=0.01, 
        lr_scheduler_type="cosine",  
        seed=3407,
        output_dir=checkpoint_dir,  
    ),
)

trainer_stats = trainer.train()

trainer.save_model(checkpoint_dir)
print("EEVE Hybrid 모델 학습 완료 및 저장")
