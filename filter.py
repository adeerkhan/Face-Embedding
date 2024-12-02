import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm

# Load pre-trained embeddings from Train1
train1_embeddings = np.load("train1_embeddings.npy")

# Directory paths
train2_dir = "data/ee488_24_data/train2"
filtered_dir = "data/ee488_24_data_filtered/train2"
Path(filtered_dir).mkdir(parents=True, exist_ok=True)

# Confidence threshold for filtering
threshold = 0.7

# Load the model checkpoint (from Train1)
checkpoint_path = "exps/train1/exp01/epoch0015.model"
model = EmbedNet(model="ResNet18", nOut=512).cuda()  # Ensure nOut matches Train1
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Function to extract embeddings
def extract_embedding(img_path, model):
    from PIL import Image
    from torchvision import transforms

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).cuda()

    # Extract embedding
    with torch.no_grad():
        embedding = model(image).cpu().numpy()
    return embedding

# Filter Train2 images
filtered_images = []
for img_path in tqdm(Path(train2_dir).rglob("*.jpg")):
    embedding = extract_embedding(str(img_path), model)
    sim = cosine_similarity(embedding, train1_embeddings)
    if sim.max() > threshold:  # Keep confident samples
        filtered_images.append(str(img_path))

        # Copy or symlink filtered images to filtered_dir
        new_path = str(img_path).replace(train2_dir, filtered_dir)
        Path(new_path).parent.mkdir(parents=True, exist_ok=True)
        Path(new_path).symlink_to(img_path)

# Save filtered image paths
with open("filtered_images.txt", "w") as f:
    for img_path in filtered_images:
        f.write(img_path + "\n")
