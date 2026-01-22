import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configurações
MODEL_ID = "google/gemma-2b"
FINETUNED_MODEL = "./gemma-2b-autoescola-finetuned"

# Configuração de carregamento leve para a GTX 1660
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def ask_model(model, tokenizer, question):
    prompt = f"### Instrução:\nResponda apenas com a letra da alternativa correta.\n\n### Entrada:\n{question}\n\n### Resposta:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            temperature=0.1, # Baixa temperatura para ser mais direto
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Limpa a resposta para mostrar apenas o que vem depois do marcador de Resposta
    return response.split("### Resposta:\n")[-1].strip()

# Pergunta de teste (use uma que esteja no seu dataset.json)
question_test = "Texto:\nMesmo quem tem experiência ao volante não se esquece do tempo em que as pernas tremiam, as mãos ficavam úmidas e a boca secava na hora de dar marcha a ré num espaço apertado ou quando aparecia um sinal fechado na ladeira. Há motoristas que, mesmo depois de rodar bastante, ainda apresentam estes sintomas.\nBoa parte deles chega a desistir de dirigir. Segundo a psicóloga Neuza Corassa, autora do livro Vença o Medo de Dirigir (Editora Gente), pelo menos 10% dos motoristas precisam de ajuda para vencer a ansiedade ao volante – um problema que pode se transformar em fobia. \"Normalmente, são pessoas que exigem muito de si mesmas e acabam desistindo de dirigir diante dos primeiros erros\", explica.\nConfira algumas recomendações da especialista:\n- Treine direção pelo menos duas vezes por semana.\n- Para algumas pessoas, entrar no carro é mais difícil que o treino propriamente dito. Não invente desculpas.\n- Não peça ajuda ao companheiro, para evitar desentendimentos. É melhor recorrer a um profissional.\n- Quando o medo provoca taquicardia, tremedeira ou falta de ar, é hora de procurar um psicólogo.\n- Há técnicas de relaxamento para baixar o nível de noradrenalina e diminuir a sensação de pânico.\n\nPergunta:\nSegundo o texto, a ansiedade ao volante atinge, sobretudo, os motoristas que\n\nAlternativas:\nA) ainda são iniciantes.\nB) dirigem muito pouco.\nC) sentem grandes fobias.\nD) deixam de treinar.\nE) exigem muito de si mesmos."

print("\n" + "="*50)
print("1. TESTANDO MODELO BASE (Sem treinamento)")
print("="*50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto"
)

res_base = ask_model(model_base, tokenizer, question_test)
print(f"\nPergunta: {question_test}")
print(f"Resposta do Modelo Base: {res_base}")

# Limpar memória para carregar o próximo
del model_base
torch.cuda.empty_cache()

print("\n" + "="*50)
print("2. TESTANDO MODELO FINETUNED (Seu treino)")
print("="*50)

# Carrega o base novamente mas anexa o seu treino (LoRA)
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto"
)
model_finetuned = PeftModel.from_pretrained(model_base, FINETUNED_MODEL)

res_fine = ask_model(model_finetuned, tokenizer, question_test)
print(f"\nPergunta: {question_test}")
print(f"Resposta do Modelo Treinado: {res_fine}")

print("\n" + "="*50)