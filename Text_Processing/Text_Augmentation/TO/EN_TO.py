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
Transform only the tourism-related words (such as place names, historical sites, parks, attractions, or activities) in the sentence below into other appropriate tourism-related words.
Output ONLY the transformed sentence EXACTLY as it should appear, and do NOT output any additional text, commentary, or explanation.
If there are no tourism-related words in the provided sentence, output the sentence as is.
Answer in English.
"""

examples = """
### Example 1
Input: Visited the famous tourist attraction Gyeongbokgung Palace in Seoul, enjoyed traditional cultural experiences, and explored local dining spots.
Answer: Visited the renowned landmark Gyeongbokgung Palace in Seoul, experienced historical heritage activities, and explored local eateries.

### Example 2
Input: Mongchontoseong and Mongchontoseong Station play a central role in studying and experiencing the history of Baekje in Seoul.
Answer: The Roman Forum and Colosseo Station play a central role in studying and experiencing the history of ancient Rome.

### Example 3
Input: Many fans gather at Jamsil Sports Complex to watch baseball games held at Jamsil Baseball Stadium.
Answer: Many fans gather at Yankee Stadium in New York to watch baseball games held by the Yankees.

### Example 4
Input: The city tour of New York provided an immersive exploration of historic landmarks, where visitors enjoyed an authentic local guide experience and sampled exquisite street food.
Answer: The urban excursion of New York provided an immersive exploration of historic sites, where visitors enjoyed an authentic local guide experience and sampled exquisite food trucks.

### Example 5
Input: The Olympic Main Stadium is the most visited site at Jamsil Sports Complex.
Answer: Wembley Stadium is the most visited site in London's Olympic area.

### Example 6
Input: hmm.
Answer: hmm.
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

def augment_text(text, max_attempts=3):
    final_prompt = f"""
In the following sentence, change only the tourism-related words into other tourism-related words.
Print ONLY the answer and do not output any explanations or additional sentences.
### Test
Input: {text}
Answer:
"""
    full_prompt = f"{instruction}\n{examples}\n{final_prompt}"
    final_result = ""
    for attempt in range(max_attempts):
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,  # 문장이 완성되도록 token 수 조정
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text.split("Answer:")[-1].strip()
        result = deduplicate_text(result)
        final_result = extract_first_sentence(result)
        if is_english(final_result):
            break
        else:
            print(f"Non-English output detected. Retrying... (attempt {attempt+1}/{max_attempts})")
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
