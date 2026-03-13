# Fine-Tuning de LLMs Pequenos

Este diretório agora possui um pipeline reproduzível para:

- dividir o dataset em `train/test` de forma estratificada por alternativa correta;
- medir o desempenho do modelo base antes do ajuste;
- treinar um adapter `LoRA/QLoRA` em cima do modelo escolhido;
- medir novamente o desempenho no teste;
- salvar previsões, métricas e deltas de acurácia por execução.

## Modelos suportados por alias

- `qwen2.5-1.5b`
- `qwen2.5-3b`
- `gemma-3-1b`
- `gemma-3-4b`
- `phi-3.5-mini`
- `llama-3.2-3b`
- `amadeus-verbo-1.5b`

Também é possível passar um `model_id` direto do Hugging Face em `--model`.

## Dataset esperado

O pipeline usa arquivos `.json` ou `.jsonl` com os campos:

```json
{
  "instruction": "Escolha a alternativa correta com base no texto.",
  "input": "Texto... Pergunta... Alternativas...",
  "output": "A"
}
```

Por padrão ele lê:

`experiments/data/datasets/dataset.jsonl`

## Dependências

Você pode reaproveitar o ambiente já usado nos scripts antigos. Se preferir um ambiente isolado:

```bash
pip install -r experiments/requirements.txt
```

## Exemplo de execução

```bash
python3 experiments/run_experiment.py \
  --model qwen2.5-3b \
  --epochs 8 \
  --seed 42
```

O comportamento padrão agora é `80/20` estratificado, o que neste dataset de `49` questões resulta em aproximadamente `39 treino / 10 teste`.

Para rodar todos os modelos recomendados:

```bash
python3 experiments/run_suite.py --seeds 13 21 42
```

O `run_suite.py` agora roda por padrão esta tríade menor, pensada para validação rápida de ganho com fine-tuning:

- `qwen2.5-1.5b`
- `gemma-3-1b`
- `amadeus-verbo-1.5b`

## Saídas

Cada rodada é salva em:

`experiments/artifacts/<modelo>/seed-<seed>/`

Arquivos principais:

- `run_config.json`: configuração completa da execução
- `summary.json`: resumo enxuto com `train_count`, `test_count`, acurácia antes/depois e delta no teste
- `adapter/`: adapter LoRA treinado
- `reports/test_before_predictions.json`
- `reports/test_after_predictions.json`
- `suite_summary.json`: resumo agregado quando usar `run_suite.py`

Exemplo de `summary.json`:

```json
{
  "model_alias": "qwen2.5-3b",
  "model_id": "Qwen/Qwen2.5-3B-Instruct",
  "seed": 42,
  "train_count": 38,
  "test_count": 11,
  "baseline_test_accuracy": 0.72,
  "finetuned_test_accuracy": 0.81,
  "delta_test_accuracy": 0.09,
  "baseline_test_correct": 8,
  "finetuned_test_correct": 9,
  "baseline_test_answer_rate": 1.0,
  "finetuned_test_answer_rate": 1.0,
  "training_summary": {
    "train_metrics": {},
    "eval_metrics": {},
    "best_model_checkpoint": null,
    "trainable_params": {}
  }
}
```

## Observações práticas

- Em `16 GB` de VRAM, este pipeline foi pensado para `LoRA/QLoRA` em `4-bit`.
- O split padrao foi simplificado para `80/20`, priorizando validar rapidamente se o fine-tuning ajuda ou nao.
- O `run_suite.py` foi reduzido para 3 modelos menores por padrao, para baratear tempo de experimento e facilitar comparacao inicial.
- Se depois voce quiser uma metodologia mais forte para pesquisa, vale evoluir para multiplas seeds ou `k-fold`.
- Para `Gemma` e `Llama`, normalmente é preciso aceitar a licença no Hugging Face e definir `HF_TOKEN` ou `HUGGINGFACE_HUB_TOKEN`.
- Como o dataset tem apenas `49` exemplos, vale rodar vários `seeds` para ter uma noção mais estável do ganho real.
