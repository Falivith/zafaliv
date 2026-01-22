import json
from pathlib import Path

# Caminhos dos arquivos
QUESTOES_PATH = Path("questoes_3_ano_formatadas.json")
GABARITO_PATH = Path("gabarito_questoes_3_ano_formatadas.json")
OUTPUT_PATH = Path("dataset_sft_portugues_3_ano.json")

# Carregar JSONs
with open(QUESTOES_PATH, "r", encoding="utf-8") as f:
    questoes = json.load(f)

with open(GABARITO_PATH, "r", encoding="utf-8") as f:
    gabarito = json.load(f)

# Indexar gabarito por id
gabarito_por_id = {
    item["id"]: item["resposta_correta"]
    for item in gabarito
}

dataset = []

for q in questoes:
    qid = q["id"]

    if qid not in gabarito_por_id:
        print(f"[WARN] Questão {qid} sem gabarito, pulando.")
        continue

    # Montar alternativas como texto
    alternativas_texto = "\n".join(
        f"{letra}) {texto}"
        for letra, texto in q["alternativas"].items()
    )

    # Input do modelo
    input_text = (
        f"Texto:\n{q['texto_base']}\n\n"
        f"Pergunta:\n{q['enunciado']}\n\n"
        f"Alternativas:\n{alternativas_texto}"
    )

    sample = {
        "instruction": "Escolha a alternativa correta com base no texto.",
        "input": input_text,
        "output": gabarito_por_id[qid]
    }

    dataset.append(sample)

# Salvar dataset final
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"[OK] Dataset gerado com {len(dataset)} exemplos em {OUTPUT_PATH}")
