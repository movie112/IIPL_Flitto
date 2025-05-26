import json
from tqdm import tqdm
import hgtk
from unsloth import FastLanguageModel
import torch

MODEL_PATH = "..."
INPUT_JSON  = ".../transcription.json"
OUTPUT_JSON = ".../transcription_v2.json"

def decompose(text):
    result = []
    for c in text:
        try:
            result.extend(hgtk.letter.decompose(c))
        except hgtk.exception.NotHangulException:
            result.append(c)
    return [j for j in result if j != '']

def format_input_with_jamo(sentence):
    return sentence

alpaca_prompt = '''당신은 한국어 문법/ 띄어쓰기 오류 교정 전문가입니다. 다음 문장은 STT 인식 오류로 인해 철자나 띄어쓰기가 잘못되었습니다.
문장의 뜻은 바꾸지 말고, 철자와 띄어쓰기만 고쳐주세요. 
입력 문장 마지막 글자에서 절대 생성하지마. 
절대 새로운 문장을 생성하거나 덧붙이지 마세요.
만약 새로운 문장을 생성하거나 덧붙인다면 페널티가 있어.
#입력: {}

#출력:'''

print("Loading model...", MODEL_PATH)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
model.eval()
tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def correct_text(text: str) -> str:
    full_prompt = alpaca_prompt.format(format_input_with_jamo(text))
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=32,
            do_sample=False,
            top_p=1.0,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    if "#출력:" in out:
        out = out.split("#출력:")[1].strip()
    out = out.split("</s>")[0].split("\n")[0].strip()
    if out.startswith("(") and out.endswith(")"):
        out = out[1:-1].strip()
    if out.endswith(","):
        out = out[:-1].strip()
    return out.strip("'\" ")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = {}

for utt_id, segments in tqdm(data.items(), desc="Correcting utterances"):
    corrected_segments = []
    for seg in segments:
        orig = seg.get("text", "")
        try:
            corr = correct_text(orig)
        except Exception as e:
            print(f"[Error] {utt_id}: \"{orig}\" -> {e}")
            corr = orig
        seg["text"] = corr
        corrected_segments.append(seg)
    new_data[utt_id] = corrected_segments

    with open(OUTPUT_JSON, "w", encoding="utf-8") as outf:
        json.dump(new_data, outf, ensure_ascii=False, indent=2)

print("완료. 교정된 결과가 누적으로 저장되었습니다:", OUTPUT_JSON)
