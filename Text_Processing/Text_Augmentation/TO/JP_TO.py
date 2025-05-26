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
以下の文章中の観光関連語（例：地名、歴史的遺産、公園、観光名所、アクティビティなど）を他の適切な観光関連語に変換してください。
変換後の文章を正確に出力し、追加のテキスト、コメント、または説明は一切出力しないでください。
もし提供された文章に観光関連語が含まれていない場合は、そのままの文章を出力してください。
日本語で回答してください。
"""

examples = """
### 例 1
入力: ソウルの有名な観光名所である景福宮を訪れ、伝統的な文化体験を楽しみ、地元の飲食店を探索しました。
回答: ソウルの名高いランドマークである景福宮を訪れ、歴史的遺産の活動を体験し、地元の食事処を探索しました。

### 例 2
入力: モンチョントソンとモンチョントソン駅は、ソウルにおける百済の歴史の研究と体験において中心的な役割を果たしています。
回答: ローマのフォーラムとコロッセオ駅は、古代ローマの歴史の研究と体験において中心的な役割を果たしています。

### 例 3
入力: 東京の主要アトラクションのいくつかは何ですか？
回答: 東京の重要なランドマークのいくつかは何ですか？

### 例 4
入力: ニューヨークの市内観光ツアーは、歴史的な名所の没入型探検を提供し、訪問者は本格的な地元ガイド体験を楽しみ、絶品のストリートフードを試食しました。
回答: ニューヨークの都市探検は、歴史的な場所の没入型探検を提供し、訪問者は本格的な地元ガイド体験を楽しみ、絶品のフードトラックを試食しました。

### 例 5
入力: オリンピックメインスタジアムは、ジャムシルスポーツコンプレックスで最も訪問される場所です。
回答: ウェンブリー・スタジアムは、ロンドンのオリンピックエリアで最も訪問される場所です。

### 例 6
入力: ふむ。
回答: ふむ。
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
