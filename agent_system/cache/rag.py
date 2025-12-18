import os, json, time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MEM_DIR = "memory"
INDEX_PATH = os.path.join(MEM_DIR, "sources.index.faiss")
META_PATH  = os.path.join(MEM_DIR, "sources.meta.json")

class SourceRAG:
    """
    Čuva: (query embedding) -> payload {query, papers, ts}
    papers je lista dict-ova koju prosleđuješ extractor-u.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        os.makedirs(MEM_DIR, exist_ok=True)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)  # list of payloads
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine-ish uz normalizaciju
            self.meta = []

    def _embed(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs).astype("float32")

    def add(self, query: str, papers: list[dict]):
        # Minimalna validacija: papiri moraju imati bar title/url/text
        payload = {"query": query, "papers": papers, "ts": int(time.time())}
        vec = self._embed([query])

        self.index.add(vec)
        self.meta.append(payload)

        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def get(self, query: str, k: int = 1, threshold: float = 0.80):
        if self.index.ntotal == 0:
            return None
        qv = self._embed([query])
        scores, ids = self.index.search(qv, k)

        best_id = int(ids[0][0])
        best_score = float(scores[0][0])

        if best_id == -1 or best_score < threshold:
            return None

        hit = self.meta[best_id]
        return {"score": best_score, **hit}