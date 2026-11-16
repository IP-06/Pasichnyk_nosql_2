import os
import re
import numpy as np
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

model_name = "allenai/specter2_base"
vector_dim = 768
batch_size = 200

pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(model_name)
df = pd.read_parquet("data/arxiv_subset.parquet")

df['abstract_len'] = df['abstract'].fillna('').apply(len)
df_30 = df.sort_values(by = 'abstract_len', ascending = False).head(30).reset_index(drop = True)
print(f"Вибрано 30 статей із найдовшими анотаціями")

def fixed_size_chunking(text: str, chunk_size: int = 80, overlap: int = 20) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks

def semantic_chunking(text: str, max_words: int = 80) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_count_word = 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current_count_word + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_count_word = len(words)
        else:
            current_chunk.append(sentence)
            current_count_word += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

index_names = {
    "fixed": "arxiv-chunks-fixed",
    "semantic": "arxiv-chunks-semantic"
}

for key, idx_name in index_names.items():
    existing_indexes = [idx['name'] for idx in pc.list_indexes()]
    if idx_name not in existing_indexes:
        print(f"Створення індексу '{idx_name}' у Pinecone")
        pc.create_index(
            name = idx_name,
            dimension = vector_dim,
            metric = "cosine",
            spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
        )

indexes = {
    "fixed": pc.Index(index_names["fixed"]),
    "semantic": pc.Index(index_names["semantic"])
    }

def process_and_upload_strategy(strategy_name: str, chunking_func):
    print(f"\n Обробка та завантаження для стратегії: {strategy_name}")
    all_vectors = []
    
    for idx, row in df_30.iterrows():
        arxiv_id = str(row.get("arxiv_id", f"paper_{idx}"))
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        year = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        category = str(row.get("category", ""))
        
        chunks = chunking_func(abstract)
        for c_idx, chunk_text in enumerate(chunks):
            vector_id = f"{strategy_name}_{idx}_chunk_{c_idx}"
            text_to_embed = f"{title} [SEP] {chunk_text}"
            metadata = {
                "arxiv_id": arxiv_id,
                "title": title[:500],
                "chunk_text": chunk_text[:1000],
                "chunk_index": c_idx,
                "year": year,
                "category": category
            }
            
            all_vectors.append({
                "id": vector_id,
                "text": text_to_embed,
                "metadata": metadata
            })
            
    index = indexes[strategy_name]
    for i in tqdm(range(0, len(all_vectors), batch_size), desc=f"Завантаження {strategy_name}"):
        batch = all_vectors[i:i + batch_size]
        texts = [item["text"] for item in batch]
        
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        vectors_to_upsert = []
        for j, item in enumerate(batch):
            vectors_to_upsert.append({
                "id": item["id"],
                "values": embeddings[j].tolist(),
                "metadata": item["metadata"]
            })
            
        index.upsert(vectors=vectors_to_upsert)
        
    stats = index.describe_index_stats()
    print(f"Загальна кількість векторів у '{index_names[strategy_name]}': {stats.get('total_vector_count', 0)}")

process_and_upload_strategy("fixed", lambda t: fixed_size_chunking(t, chunk_size=80, overlap=20))
process_and_upload_strategy("semantic", lambda t: semantic_chunking(t, max_words=80))

test_queries = [
    "deep learning architectures for image recognition",
    "natural language processing models and transformers"
]

def search_chunks(query_text: str):
    print(f"\nТестовий запит: {query_text}")
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    
    for strategy_name, index in indexes.items():
        print(f"\n-> Стратегія чанкінгу: [{strategy_name.upper()}] (індекс: {index_names[strategy_name]})")
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        
        for idx, match in enumerate(results.get('matches', []), 1):
            meta = match.get('metadata', {})
            print(f"{idx}. [Score: {match.get('score', 0):.4f}] {meta.get('title')}")
            print(f"Категорія: {meta.get('category')} | Рік: {meta.get('year')} | Чанк {meta.get('chunk_index')}")
            print(f"Текст чанка: {meta.get('chunk_text', '')[:180]}\n")

for q in test_queries:
    search_chunks(q)