import numpy as np
from pathlib import Path

train1_dir = "data/ee488_24_data/train1"
embeddings = []

for img_path in Path(train1_dir).rglob("*.jpg"):
    embedding = extract_embedding(str(img_path), model)  # Use your model
    embeddings.append(embedding)

np.save("train1_embeddings.npy", np.array(embeddings))
