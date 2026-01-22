import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "google/gemma-2b"
NEW_MODEL_NAME = "gemma-2b-autoescola-finetuned"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def create_prompt(example):
    text = f"### Instrução:\n{example['instruction']}\n\n### Entrada:\n{example['input']}\n\n### Resposta:\n{example['output']}<eos>"
    return {"text": text}

dataset = load_dataset("json", data_files="dataset.json", split="train")
dataset = dataset.map(create_prompt, remove_columns=dataset.column_names)

peft_config = LoraConfig(
    lora_alpha=16, r=8, lora_dropout=0.1,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

training_args = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    fp16=False,
    bf16=False,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer,
)

print("Iniciando treinamento...")
model.config.use_cache = False 

for name, module in model.named_modules():
    if "norm" in name:
        module.to(torch.float32)

trainer.train()

trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print(f"Sucesso! Modelo salvo em {NEW_MODEL_NAME}")