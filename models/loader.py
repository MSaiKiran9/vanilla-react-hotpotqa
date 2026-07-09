"""
Model loader for local Hugging Face models.

Supports:
- Qwen3-8B
- Llama-3.1-8B
- Gemma-2-9B
- Mistral-7B-Instruct-v0.3

Designed for Google Colab free GPU (T4).
"""

from __future__ import annotations

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import (
    MODELS,
    MAX_NEW_TOKENS,
    DO_SAMPLE,
    REPETITION_PENALTY,
    USE_4BIT,
)


class ModelLoader:
    """
    Loads one Hugging Face model and provides a
    simple generate() interface.
    """

    def __init__(self, model_name: str):

        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODELS.keys())}"
            )

        self.name = model_name
        self.model_id = MODELS[model_name]

        print(f"\nLoading {self.model_id} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None

        if USE_4BIT and torch.cuda.is_available():

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
        )

        self.model.eval()

        print("Model loaded successfully.\n")

    # ---------------------------------------------------------

    def _build_prompt(self, prompt: str) -> str:
        """
        Use chat template when available.
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        except Exception:
            return prompt

    # ---------------------------------------------------------

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """
        Generate model response.
        """

        prompt = self._build_prompt(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

    # ---------------------------------------------------------

    def unload(self):
        """
        Free GPU memory.
        """

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------

    def __repr__(self):

        return f"ModelLoader({self.name})"
