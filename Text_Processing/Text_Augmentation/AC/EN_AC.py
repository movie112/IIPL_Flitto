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
Transform only the medical-related terms (such as symptoms, diseases, treatments, diagnoses, or medical institutions) in the sentence below into other appropriate medical-related terms.
Output ONLY the transformed sentence EXACTLY as it should appear, and do NOT output any additional text, commentary, or explanation.
If there are no medical-related terms in the provided sentence, output the sentence as is.
Answer in English.
"""

examples = """
### Example 1
Input: I have had a severe headache since yesterday, so I went to the internal medicine department for a checkup.
Answer: I have had intense abdominal pain since yesterday, so I went to the gastroenterology department for a checkup.

### Example 2
Input: The doctor diagnosed hypertension and started prescribing antihypertensive medication.
Answer: The doctor diagnosed diabetes and started insulin therapy.

### Example 3
Input: During a routine health screening, blood tests and electrocardiograms are performed.
Answer: During an annual full medical exam, urine tests and chest X‑rays are performed.

### Example 4
Input: The emergency department of this hospital operates 24/7 and handles many critically ill patients.
Answer: The intensive care unit of this medical center operates 24/7 and handles many severely ill patients.

### Example 5
Input: It is common to use general anesthesia before surgery.
Answer: It is common to use local anesthesia before surgery.

### Example 6
Input: I see.
Answer: I see.
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
In the following sentence, change only the medical-related words into other medical-related words.
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
                max_new_tokens=60,
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


input_file_path = f".../EN_AC.rttm"
output_file_path = f".../EN_AC_aug.rttm"

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
