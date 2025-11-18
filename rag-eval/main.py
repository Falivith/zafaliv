from retrieval import Retriever
from generation import Generator, LLMModel

def pretty_print(query, context, answer):
    print("\n==================== QUERY ====================")
    print(query)

    print("\n==================== CONTEXT ==================")
    for c in context:
        print(f"- {c}")

    print("\n==================== ANSWER ====================")
    print(answer)
    print("===============================================\n")


retriever = Retriever()
generator = Generator(model_name=LLMModel.GEMMA3_1B)

query = "Quem foi Marie Curie?"
context = retriever.retrieve(query, k=2)
answer = generator.generate(query, context)

pretty_print(query, context, answer)
