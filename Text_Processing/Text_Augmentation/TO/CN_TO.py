import os
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)
model.eval()

instruction = """
请将以下句子中与旅游相关的词语（如地名、历史遗迹、公园、景点或活动）替换为其他合适的旅游相关词语。
仅输出替换后的句子，不要输出任何额外的文字、评论或解释。
如果提供的句子中没有旅游相关词语，则原样输出句子。
用中文作答。请不要生成任何额外的符号（例如《、》、`等）。
"""

examples = """
### 示例 1
输入：参观了首尔著名旅游景点景福宫，体验了传统文化活动，并探索了当地美食。
回答：参观了首尔著名地标景福宫，体验了历史遗产活动，并探索了当地餐馆。

### 示例 2
输入：梦村土城和梦村土城站在研究和体验首尔百济历史方面起着核心作用。
回答：罗马广场和斗兽场站在研究和体验古罗马历史方面起着核心作用。

### 示例 3
输入：许多粉丝聚集在蚕室综合运动场观看在蚕室棒球场举行的棒球比赛。
回答：许多粉丝聚集在纽约的洋基体育场观看洋基队举办的棒球比赛。

### 示例 4
输入：纽约城市游带来了一次沉浸式的历史地标探索，游客们享受了地道的本地导览体验并品尝了精致的街头美食。
回答：纽约城市之旅带来了一次沉浸式的历史遗址探索，游客们享受了地道的本地导览体验并品尝了精致的美食车。

### 示例 5
输入：奥林匹克主体育场是蚕室综合运动场中访问量最多的场所。
回答：温布利体育场是伦敦奥林匹克地区访问量最多的场所。

### 示例 6
输入：嗯。
回答：嗯。
"""

def deduplicate_text(text):
    sentences = re.split(r'(?:\n|(?<=[。！？])\s+)', text)
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return " ".join(unique_sentences)

def extract_first_sentence(text):
    match = re.search(r'^(.+?[。！？])(?:\s+|$)', text)
    if match:
        return match.group(1).strip()
    return text.strip()

def clean_output(text):
    text = re.split(r'```', text)[0]
    text = text.split('\n')[0]
    return text.strip()

def contains_unwanted_symbols(text):
    return bool(re.search(r'[《》`]', text))

def is_only_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text)) and not bool(re.search(r'[a-zA-Z]', text))

def augment_text(text, max_attempts=5):
    final_prompt = f"""
在下列句子中，仅将与旅游相关的词语替换为其他旅游相关词语。
只打印答案，不要输出任何解释或额外的句子。
### 测试
输入: {text}
回答:
"""
    full_prompt = f"{instruction}\n{examples}\n{final_prompt}"
    final_result = ""
    for attempt in range(max_attempts):
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text.split("回答:")[-1].strip()
        result = deduplicate_text(result)
        result = extract_first_sentence(result)
        result = clean_output(result)
        final_result = result
        if is_only_chinese(final_result) and not contains_unwanted_symbols(final_result):
            break
        else:
            print(f"Output contains unwanted symbols or English. Retrying... (attempt {attempt+1}/{max_attempts})")
    return final_result

datasets = ["train", "valid", "test"]

for dataset in datasets:
    input_file_path = f".../{dataset}/rttm"
    output_file_path = f".../{dataset}/rttm_augmented"
    
    total_to_lines = 0
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) > 1 and len(tokens[1]) >= 5 and tokens[1][3:5] == "TO":
                total_to_lines += 1
    
    print(f"Processing {dataset} dataset, total TO lines: {total_to_lines}")
    
    with open(input_file_path, "r", encoding="utf-8") as infile, \
         open(output_file_path, "a", encoding="utf-8") as outfile:
        
        pbar = tqdm(total=total_to_lines, desc=f"Processing TO lines in {dataset}")
        for line in infile:
            tokens = line.strip().split()
            if len(tokens) > 1 and len(tokens[1]) >= 5 and tokens[1][3:5] == "TO":
                if line.strip().endswith('>'):
                    last_angle_index = line.rfind('<')
                    transcript = line[last_angle_index+1:].rstrip().rstrip('>')
                    augmented_transcript = augment_text(transcript)
                    new_line = line[:last_angle_index] + f"<{augmented_transcript}>\n"
                    outfile.write(new_line)
                    outfile.flush()
                pbar.update(1)
        pbar.close()
