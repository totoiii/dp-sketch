"""LLM + similarity backends."""

import functools
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    """HuggingFace Seq2Seq generator (Flan-T5)."""

    def __init__(self, model_name="google/flan-t5-base", max_tokens=256, device="auto"):
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.max_tokens = max_tokens

    def __call__(self, prompt, temperature=0.7, max_tokens=None, min_tokens=None):
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                max_length=512, truncation=True).to(self.device)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens or self.max_tokens,
            do_sample=True,
            temperature=max(temperature, 0.1),
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )
        if min_tokens:
            gen_kwargs["min_new_tokens"] = min_tokens
        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class Similarity:
    """Sentence-transformers cosine similarity with caching."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    @functools.lru_cache(maxsize=8192)
    def _embed(self, text):
        import numpy as np
        return tuple(self.model.encode(text, convert_to_numpy=True).tolist())

    def __call__(self, a, b):
        import numpy as np
        ea, eb = np.array(self._embed(a)), np.array(self._embed(b))
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))
