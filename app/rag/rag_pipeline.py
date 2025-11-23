import os
from app.models.embeddings import SentenceEncoder
from app.rag.vectorstore import VectorStore

class TravelPipeline:
    def __init__(self, data_path: str):
        self.encoder = SentenceEncoder()
        self.vs = VectorStore(self.encoder, data_path)

    def search_places(self, destination: str, budget_egp: float, query: str = ""):
        base_query = query or f"top attractions and local experiences in {destination}"
        items = self.vs.search(base_query, city_filter=destination, k=12)
        # Filter by rough average cost under budget/day window
        filtered = [it for it in items if it.get("avg_cost_egp", 0) <= budget_egp]
        return filtered or items  # fallback if too strict

def init_pipeline():
    data_path = os.getenv("PLACES_DATA", "data/places_sample.json")
    return TravelPipeline(data_path)
