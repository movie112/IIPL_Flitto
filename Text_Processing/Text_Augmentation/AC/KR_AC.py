import os
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)
model.eval()

instruction = """
의료 관련 단어(예: 증상, 질병, 치료 방법, 진단명, 의료 기관 등)만 다른 적절한 의료 관련 단어로 변환하세요.
변환된 문장을 정확히 그대로 출력하고 추가적인 텍스트, 설명 또는 주석은 출력하지 마세요.
input 문장의 문맥을 잘 고려하고 존댓말과 반말을 고려하여 문장을 출력하세요.
제공된 문장에 의료 관련 단어가 없다면 문장을 그대로 출력하세요.
한국어로 답변하세요.
"""

examples = """
### 예시 1
input: 어제부터 심한 두통이 있어서 내과에 가서 진료를 받았어요.
output: 어제부터 복통이 심해서 소화기내과에 가서 진료를 받았어요.

### 예시 2
input: 의사 선생님이 고혈압이라고 진단하고, 혈압약 처방을 시작했어요.
output: 의사 선생님이 당뇨병이라고 진단하고, 인슐린 치료를 시작했어요.

### 예시 3
input: 정기 건강검진에서는 혈액 검사와 심전도를 진행해요.
output: 연례 종합검진에서는 소변 검사와 흉부 엑스레이를 진행해요.

### 예시 4
input: 이 병원의 응급실은 24시간 운영되며 위급한 환자를 많이 치료해요.
output: 이 의료센터의 중환자실은 24시간 운영되며 상태가 심각한 환자를 많이 치료해요.

### 예시 5
input: 수술 전에 전신 마취를 사용하는 것이 일반적이에요.
output: 수술 전에 국소 마취를 사용하는 것이 일반적이에요.

### 예시 6
input: 알겠어요.
output: 알겠어요.
"""


def deduplicate_text(text):
    sentences = re.split(r'(?:\n|(?<=[.!?])\s+)', text)
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return " ".join(unique_sentences)

def extract_first_sentence(text):
    match = re.search(r'^(.+?[.!?])(?:\s+|$)', text)
    if match:
        return match.group(1).strip()
    return text.strip()

def is_english(text):
    return not bool(re.search(r'[\u4e00-\u9fff]', text))

def contains_english(text):
    return bool(re.search(r'[A-Za-z]', text))

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def is_korean(text):
    return bool(re.search(r'[\uac00-\ud7af]', text)) and not contains_english(text) and not contains_chinese(text)

def augment_text(text, max_attempts=5):
    final_prompt = f"""
다음 문장에서 의료 관련 단어만 다른 의료 관련 단어로 변경하세요.
한국어로 출력하고 추가 설명이나 문장은 출력하지 마세요.
###
입력: {text}
출력:
"""
    full_prompt = f"{instruction}\n{examples}\n{final_prompt}"
    
    for attempt in range(max_attempts):
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        if is_korean(generated_text):
            break
        else:
            print(f"Non-Korean (or contains English/Chinese) output detected. Retrying... (attempt {attempt+1}/{max_attempts})")
    
    if "출력:" in generated_text:
        output_text = generated_text.rpartition("출력:")[2].strip()
        lines = output_text.splitlines()
        if lines:
            output_text = lines[0].strip()
        else:
            output_text = text
        return output_text
    return generated_text


input_file_path = ".../KR_AC.rttm"
output_file_path = ".../KR_AC_aug.rttm"

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
