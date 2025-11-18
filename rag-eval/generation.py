from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LLMModel(Enum):
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_3_2_8B = "meta-llama/Llama-3.2-8B"
    LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA_2_13B = "meta-llama/Llama-2-13b-chat-hf"
    LLAMA_3_3_70B = "meta-llama/Llama-3.3-70B"
    GEMMA_2B = "google/gemma-2b"
    GEMMA_7B = "google/gemma-7b"
    GEMMA3_1B = "google/gemma-3-1b-it"


class Generator:
    """
    Generator class responsible for producing model responses given a query and context.
    
    This class loads a HuggingFace transformer model and provides a unified `generate`
    method that formats the RAG-style prompt and returns a text response.
    """

    def __init__(
        self,
        model_name: LLMModel = LLMModel.LLAMA_3_2_1B,
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ):
        """
        Initialize the text generator with a specific LLM.

        Args:
            model_name (LLMModel): Enum specifying the HF model to load.
            max_new_tokens (int): Maximum number of tokens the model can generate.
            temperature (float): Sampling temperature (higher = more creative).
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name.value)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name.value,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def build_prompt(self, query: str, context: list[str]) -> str:
        """
        Build a RAG-style prompt combining context and user query.

        Args:
            query (str): User question.
            context (list[str]): Retrieved documents used as grounding.

        Returns:
            str: A formatted prompt.
        """
        formatted_context = "\n".join([f"- {c}" for c in context])

        return (
            "You are a helpful and concise assistant.\n"
            "Use ONLY the provided context to answer the question.\n"
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{formatted_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def generate(self, query: str, context: list[str]) -> str:
        """
        Generate a response from the LLM using the provided query and context.

        Args:
            query (str): User question.
            context (list[str]): List of retrieved context passages.

        Returns:
            str: Model-generated answer.
        """
        prompt = self.build_prompt(query, context)

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True
        )

        return outputs[0]["generated_text"]
