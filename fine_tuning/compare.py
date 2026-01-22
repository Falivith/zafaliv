import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

MODEL_ID = "google/gemma-2b"
FINETUNED_MODEL = "./gemma-2b-autoescola-finetuned"
DATASET_FILE = "dataset.json"
OUTPUT_LOG = "log_avaliacao.json"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def get_prediction(model, tokenizer, instruction, input_text):
    prompt = f"### Instrução:\n{instruction}\n\n### Entrada:\n{input_text}\n\n### Resposta:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=15, 
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.split("### Resposta:\n")[-1].strip()

def check_accuracy(prediction, expected):
    if not prediction or not expected:
        return False
    return prediction.lower().startswith(expected[0].lower())

with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
logs = []

print("\n--- Processando Modelo Base ---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)

for item in tqdm(test_data):
    pred = get_prediction(model, tokenizer, item['instruction'], item['input'])
    logs.append({
        "instruction": item['instruction'],
        "input": item['input'],
        "output_esperado": item['output'],
        "modelo_base": {
            "resposta": pred,
            "acertou": check_accuracy(pred, item['output'])
        }
    })

del model
torch.cuda.empty_cache()

print("\n--- Processando Modelo Treinado ---")
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model_fine = PeftModel.from_pretrained(model_base, FINETUNED_MODEL)

for i, item in enumerate(tqdm(test_data)):
    pred = get_prediction(model_fine, tokenizer, item['instruction'], item['input'])
    logs[i]["modelo_treinado"] = {
        "resposta": pred,
        "acertou": check_accuracy(pred, item['output'])
    }

with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
    json.dump(logs, f, indent=2, ensure_ascii=False)


base_hits = sum(1 for x in logs if x['modelo_base']['acertou'])
fine_hits = sum(1 for x in logs if x['modelo_treinado']['acertou'])

print(f"\nLog gerado com sucesso: {OUTPUT_LOG}")
print(f"Taxa de Acerto Base: {base_hits}/{len(test_data)}")
print(f"Taxa de Acerto Treinado: {fine_hits}/{len(test_data)}")