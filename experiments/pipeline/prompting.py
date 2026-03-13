from __future__ import annotations

SYSTEM_PROMPT = (
    "Voce responde questoes de multipla escolha em portugues. "
    "Responda somente com a letra correta entre A e E."
)


def build_user_message(record: dict) -> str:
    return (
        f"Instrucao:\n{record['instruction']}\n\n"
        f"Entrada:\n{record['input']}\n\n"
        "Resposta:"
    )


def _manual_prompt(record: dict, include_answer: bool) -> str:
    prompt = (
        f"Sistema:\n{SYSTEM_PROMPT}\n\n"
        f"{build_user_message(record)}"
    )
    if include_answer:
        return f"{prompt}\n{record['output']}"
    return prompt


def render_training_text(record: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(record)},
        {"role": "assistant", "content": record["output"]},
    ]

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    return _manual_prompt(record, include_answer=True)


def render_inference_prompt(record: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(record)},
    ]

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return _manual_prompt(record, include_answer=False)
