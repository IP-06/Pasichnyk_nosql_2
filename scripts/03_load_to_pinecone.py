import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

input_parquet = "data/arxiv_subset.parquet"
input_embeddings = "embeddings/embeddings.npy"
index_name = "arxiv-papers"
vector_dim = 768
batch_size = 200

pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])

existing_indexes = [idx['name'] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Створення індексу '{index_name}' у Pipecone")
    pc.create_index(
        name = index_name,
        dimension = vector_dim,
        metric = "cosine",
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )

index = pc.Index(index_name)

print("Завантаження датасету та ембеддингів")
df = pd.read_parquet(input_parquet)
embeddings = np.load(input_embeddings)
total_records = len(df)
print(f"Загальна кількість записів: {total_records}")

print("Завантаження векторів у Pinecone батчами")
for i in tqdm(range(0, total_records, batch_size), desc = "Завантаження батчів"):
    batch_end = min(i + batch_size, total_records)
    batch_df = df.iloc[i:batch_end]
    batch_embeddings = embeddings[i:batch_end]

    vectors = []
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        global_idx = i + idx
        vector_id = f"paper_{global_idx}"
        vector_values = batch_embeddings[idx].tolist()

        metadata = {
            "arxiv_id": str(row.get("arxiv_id", "")),
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", ""))[:500],
            "authors": str(row.get("authors", ""))[:200],
            "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
            "category": str(row.get("category", ""))
        }

        vectors.append({
            "id": vector_id,
            "values": vector_values,
            "metadata": metadata
        })

    index.upsert(vectors = vectors)

stats = index.describe_index_stats()
print("\nЗавантаження успішно завершено!")
print(f"Загальна кількість векторів в індексі '{index_name}': {stats.get('total_vector_count', 0)}")