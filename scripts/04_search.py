import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

load_dotenv()

index_name = "arxiv-papers"
model_name = "allenai/specter2_base"
top_k = 5

pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
index = pc.Index(index_name)
model = SentenceTransformer(model_name)

input_parquet = "data/arxiv_subset.parquet"
input_embeddings = "embeddings/embeddings.npy"
df = pd.read_parquet("data/arxiv_subset.parquet")

def encode_query(query_text: str) -> list:
    embedding = model.encode(query_text, normalize_embeddings = True)
    return embedding.tolist()

query_1 = "teaching machines to recognize objects in pictures"
print(f"Запит: '{query_1}")

query_vector_1 = encode_query(query_1)
results_pure = index.query(
    vector = query_vector_1,
    top_k = top_k,
    include_metadata = True
)

for match in results_pure.get('matches', []):
    meta = match.get('metadata', {})
    print(f"Назва: {meta.get('title')}")
    print(f"Категорія: {meta.get('category')} | Рік: {meta.get('year')}")
    print(f"Абстракт: {meta.get('abstract', '')[:150]}\n")

query_2 = "reinforcement learning"
query_vector_2 = encode_query(query_2)
print("Приклад А: 'reinforcement learning' за останні 5 років та категорією cs.LG")
results_filter_a = index.query(
    vector = query_vector_2,
    top_k = 3,
    include_metadata = True,
    filter = {
        "category": {"$eq": "cs.LG"},
        "year": {"$gte": 2021.0}
    }
)

matches_a = results_filter_a.get('matches', [])

if not matches_a:
    print("  * За даним запитом та фільтрами нічого не знайдено")
else:
    for match in matches_a:
        meta = match.get('metadata', {})
        print(f"  * [{meta.get('year')}] {meta.get('title')} ({meta.get('category')})")

print("Приклад B: старі статті (до 2015 року)")
results_filter_b = index.query(
    vector = query_vector_2,
    top_k = 3,
    include_metadata = True,
    filter = {"year": {"$lt": 2015.0}}
)

for match in results_filter_b.get('matches', []):
    meta = match.get('metadata', {})
    print(f"  * [{meta.get('year')}] {meta.get('title')} ({meta.get('category')})")

print("\nПорівняння метрик на локальних ембеддингах")
embeddings = np.load(input_embeddings)
query_emb_local = model.encode(query_1, normalize_embeddings = False).reshape(1, -1)

cos_sim = cosine_similarity(query_emb_local, embeddings)[0]
top_cos = np.argsort(cos_sim)[::-1][:top_k]

dot_prod = np.dot(embeddings, query_emb_local.T).flatten()
top_dot = np.argsort(dot_prod)[::-1][:top_k]

l2_dist = euclidean_distances(query_emb_local, embeddings)[0]
top_l2 = np.argsort(l2_dist)[:top_k]

print("\nТоп-5 за Cosine Similarity:")
for idx in top_cos:
    print(f"  - [{cos_sim[idx]:.4f}] {df.iloc[idx]['title'][:60]}")

print("\nТоп-5 за Dot Product:")
for idx in top_dot:
    print(f"  - [{dot_prod[idx]:.4f}] {df.iloc[idx]['title'][:60]}")

print("\nТоп-5 за L2-distance:")
for idx in top_l2:
    print(f"  - [{l2_dist[idx]:.4f}] {df.iloc[idx]['title'][:60]}")