from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class SentenceEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Fallback to MiniLM; change to Arabic-friendly model if needed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms
