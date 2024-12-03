import numpy as np
import torch
from pathlib import Path
from EmbedNet import EmbedNet

# Initialize EmbedNet model with arguments directly passed
model = EmbedNet(
    model="GhostFaceNetsV2",
    optimizer="adopt",
    trainfunc="arcface",
    nPerClass=1,
    nOut=1024,  # Ensure this matches your desired embedding size
    nClasses=2882,
    scale=30.0,
    margin=0.1,
    width=1,
    dropout=0.0,
    image_size=256,
    mixedprec=False
).cuda()
model.eval()  # Set to evaluation mode

# Directory path for train1
train1_dir = "data/ee488_24_data/train1"
embeddings = []

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

# Generate embeddings for train1 images
for img_path in Path(train1_dir).rglob("*.jpg"):
    embedding = extract_embedding(str(img_path), model)
    embeddings.append(embedding)

# Save embeddings to a file
np.save("train1_embeddings.npy", np.array(embeddings))
print("Embeddings for train1 saved to train1_embeddings.npy")
