import json
import os
import faiss
import numpy as np

class VectorStore:
    def __init__(self, encoder, data_path: str):
        self.encoder = encoder
        self.data_path = data_path
        self.items = []
        self.index = None
        self._load_data()
        self._build_index()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

    def _item_text(self, item):
        return f"{item['city']} - {item['name']} - {item['type']} - {item['description']}"

    def _build_index(self):
        texts = [self._item_text(it) for it in self.items]
        embs = self.encoder.encode(texts).astype("float32")
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

    def search(self, query: str, city_filter: str = None, k: int = 8):
        q_emb = self.encoder.encode(query).astype("float32")
        scores, ids = self.index.search(q_emb, k)
        results = []
        for i in ids[0]:
            item = self.items[int(i)]
            if city_filter and item["city"].lower() != city_filter.lower():
                continue
            results.append(item)
        return results
