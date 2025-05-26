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
관광 관련 단어(예: 장소 이름, 역사적 명소, 공원, 관광 명소 또는 활동)만 다른 적절한 관광 관련 단어로 변환하세요.
변환된 문장을 정확히 그대로 출력하고 추가적인 텍스트, 설명 또는 주석은 출력하지 마세요.
input 문장의 문맥을 잘 고려하고 존댓말과 반말을 고려하여 문장을 출력하세요.
제공된 문장에 관광 관련 단어가 없다면 문장을 그대로 출력하세요.
한국어로 답변하세요.
"""

examples = """
### 예시 1
input: 서울에서 유명한 관광 명소 경복궁을 방문하고, 전통 문화 체험을 즐기며, 지역 식당들을 탐방했다.
output: 서울에서 유명한 랜드마크 경복궁을 방문하고, 역사적 유산 체험을 경험하며, 지역 맛집들을 탐방했다.

### 예시 2
input: 몽촌토성과 몽촌토성역은 서울에서 백제의 역사를 학습하고 체험하는 데 중심적인 역할을 한다.
output: 로마 포룸과 콜로세움역은 고대 로마 역사를 배우고 체험하는 데 중심적인 역할을 한다.

### 예시 3
input: 많은 팬들이 잠실 종합운동장에 모여 잠실 야구 경기장에서 열리는 야구 경기를 관람한다.
output: 많은 팬들이 뉴욕 양키 스타디움에 모여 양키스가 개최하는 야구 경기를 관람한다.

### 예시 4
input: 뉴욕의 시티 투어는 역사적인 랜드마크들을 몰입감 있게 탐험할 수 있도록 제공했으며, 방문객들은 진정한 지역 가이드 체험을 즐기고 정교한 스트리트 푸드를 맛보았다.
output: 뉴욕의 도시 여행은 역사적 명소들을 몰입감 있게 탐험할 수 있도록 제공했으며, 방문객들은 진정한 지역 가이드 체험을 즐기고 정교한 푸드 트럭 음식을 맛보았다.

### 예시 5
input: 올림픽 메인 스타디움은 잠실 스포츠 콤플렉스에서 가장 많이 방문한 장소이다.
output: 웸블리 스타디움은 런던의 올림픽 구역에서 가장 많이 방문한 장소이다.

### 예시 6
input: 음.
output: 음.
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

def contains_english(text):
    return bool(re.search(r'[A-Za-z]', text))

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def is_korean(text):
    return bool(re.search(r'[\uac00-\ud7af]', text)) and not contains_english(text) and not contains_chinese(text)

def augment_text(text, max_attempts=5):
    final_prompt = f"""
다음 문장에서 관광 관련 단어만 다른 관광 관련 단어로 변환하세요.
한국어로 답변만 출력하고 추가 설명이나 문구를 출력하지 마세요.
### 테스트
input: {text}
output:
"""
    full_prompt = f"{instruction}\n{examples}\n{final_prompt}"
    final_result = ""
    for attempt in range(max_attempts):
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text.split("output:")[-1].strip()
        result = deduplicate_text(result)
        final_result = extract_first_sentence(result)
        if is_korean(final_result):
            break
        else:
            print(f"Non-Korean (or contains English/Chinese) output detected. Retrying... (attempt {attempt+1}/{max_attempts})")
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
