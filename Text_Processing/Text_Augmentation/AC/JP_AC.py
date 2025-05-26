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
以下の文章中の医療関連語（例：症状、疾患、治療法、診断名、医療機関など）を他の適切な医療関連語に変換してください。
変換後の文章を正確に出力し、追加のテキスト、コメント、または説明は一切出力しないでください。
もし提供された文章に医療関連語が含まれていない場合は、そのままの文章を出力してください。
日本語で回答してください。
"""

examples = """
### 例 1
入力: 昨日からひどい頭痛がして、内科で診てもらいました。
回答: 昨日から激しい腹痛がして、消化器内科で診てもらいました。

### 例 2
入力: 医師は高血圧と診断し、降圧薬の処方を始めました。
回答: 医師は糖尿病と診断し、インスリン療法を開始しました。

### 例 3
入力: 定期的な健康診断では、血液検査や心電図が行われます。
回答: 年次の人間ドックでは、尿検査や肺のX線撮影が行われます。

### 例 4
入力: この病院の救急外来は24時間体制で、多くの重症患者に対応しています。
回答: この医療センターの集中治療室は24時間体制で、多くの重篤患者に対応しています。

### 例 5
入力: 手術の前に全身麻酔を使用することが一般的です。
回答: 手術の前に局所麻酔を使用することが一般的です。

### 例 6
入力: なるほど。
回答: なるほど。
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
    return bool(re.search(r'[《》`®�]', text))

def is_only_japanese(text):
    contains_kana = bool(re.search(r'[\u3040-\u30FF]', text))
    contains_korean = bool(re.search(r'[\uAC00-\uD7A3]', text))
    return contains_kana and not contains_korean

def augment_text(text, max_attempts=5):
    final_prompt = f"""
入力: {text}
回答:
"""
    full_prompt = f"{instruction}\n{examples}\n{final_prompt}"
    final_result = ""
    for attempt in range(max_attempts):
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
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
        if is_only_japanese(final_result) and not contains_unwanted_symbols(final_result):
            break
        else:
            print(f"Non-Japanese or unwanted symbols detected. Retrying... (attempt {attempt+1}/{max_attempts})")
    return final_result


input_file_path = f".../JP_AC.rttm"
output_file_path = f".../JP_AC_aug.rttm"

total_to_lines = 0
with open(input_file_path, "r", encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()
        if len(tokens) > 1 and len(tokens[1]) >= 5 and tokens[1][3:5] == "AC":
            total_to_lines += 1

print(f"Processing dataset, total AC lines: {total_to_lines}")

with open(input_file_path, "r", encoding="utf-8") as infile, \
        open(output_file_path, "a", encoding="utf-8") as outfile:
    
    pbar = tqdm(total=total_to_lines, desc=f"Processing AC lines")
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
