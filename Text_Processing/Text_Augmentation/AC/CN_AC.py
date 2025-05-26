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
请将以下句子中与医疗相关的词语（如症状、疾病、治疗方法、诊断或医疗机构等）替换为其他合适的医疗相关词语。
仅输出替换后的句子，不要输出任何额外的文字、评论或解释。
如果提供的句子中没有医疗相关词语，则原样输出句子。
用中文作答。请不要生成任何额外的符号（例如《、》、等）。
"""

examples = """
### 示例 1
输入：我从昨天开始就一直头痛得厉害，所以去了内科检查。
回答：我从昨天开始就一直腹痛得厉害，所以去了消化科检查。

### 示例 2
输入：医生诊断为高血压并开始开具降压药。
回答：医生诊断为糖尿病并开始胰岛素治疗。

### 示例 3
输入：在例行的健康检查中，会进行血液检查和心电图。
回答：在年度的全面体检中，会进行尿液检查和胸部X光。

### 示例 4
输入：这家医院的急诊科全天候运行，接收大量危重病人。
回答：这家医疗中心的重症监护室全天候运行，接收大量病情严重的病人。

### 示例 5
输入：手术前通常会使用全身麻醉。
回答：手术前通常会使用局部麻醉。

### 示例 6
输入：我明白了。
回答：我明白了。
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
在下列句子中，仅将与医疗相关的词语替换为其他医疗相关词语。  
只打印答案，不要输出任何解释或额外的句子。
###
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

input_file_path = f".../CN_AC.rttm"
output_file_path = f".../CN_AC_aug.rttm"

total_to_lines = 0
with open(input_file_path, "r", encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()
        if len(tokens) > 1 and len(tokens[1]) >= 5 and tokens[1][3:5] == "AC":
            total_to_lines += 1

print(f"Processing dataset, total AC lines: {total_to_lines}")

with open(input_file_path, "r", encoding="utf-8") as infile, \
        open(output_file_path, "a", encoding="utf-8") as outfile:
    
    pbar = tqdm(total=total_to_lines, desc=f"Processing AC")
    for line in infile:
        tokens = line.strip().split()
        if len(tokens) > 1 and len(tokens[1]) >= 5 and tokens[1][3:5] == "AC":
            if line.strip().endswith('>'):
                last_angle_index = line.rfind('<')
                transcript = line[last_angle_index+1:].rstrip().rstrip('>')
                augmented_transcript = augment_text(transcript)
                new_line = line[:last_angle_index] + f"<{augmented_transcript}>\n"
                outfile.write(new_line)
                outfile.flush()
            pbar.update(1)
    pbar.close()
