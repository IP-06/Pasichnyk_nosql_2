import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

def main():
    print("Завантаження датасету")
    file_path = "data/arxiv_subset.parquet"
    df = pd.read_parquet(file_path)

    print("Підготовка та об'єднання текстів")
    texts = (df['title'].astype(str) + " [SEP] " + df['abstract'].astype(str)).tolist()

    print("Ініціалізація моделі")
    model = SentenceTransformer('allenai/specter2_base')

    print("")
    embeddings = model.encode(
        texts,
        batch_size = 64,
        show_progress_bar = True,
        normalize_embeddings = True
    )

    total_texts = len(texts)
    embedding_dim = embeddings.shape[1]
    first_embedding_norm = np.linalg.norm(embeddings[0])

    print("\n Результати")
    print(f"Загальна кількість оброблених текстів: {total_texts}")
    print(f"Розмірність ембеддингів: {embedding_dim}")
    print(f"Норма першого ембеддингу: {first_embedding_norm:.4f}")

    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok = True)

    output_file = os.path.join(output_dir, "embeddings.npy")
    np.save(output_file, embeddings)
    print(f"Ембеддинги успішно збережено у файл: {output_file}")

if __name__ == "__main__":
    main()