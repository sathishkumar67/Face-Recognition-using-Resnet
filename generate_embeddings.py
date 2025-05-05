# generate_embeddings.py
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from siamese_resnet.model import SiameseResNet
from mtcnn import MTCNN  # Install with: pip install mtcnn

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "state_dict.pt"  # Use the checkpoint saved by trainer.py
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_PATH = "embeddings_db.npy"
IMG_SIZE = 224

# Load model with proper initialization
model = SiameseResNet(embedding_dim=256).to(DEVICE)

# Load checkpoint (modified from original code)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

# Initialize face detector
detector = MTCNN()

def preprocess_face(image):
    """Professional face preprocessing pipeline"""
    # Detect face using MTCNN
    results = detector.detect_faces(image)
    if not results:
        return None
    
    # Select face with highest confidence
    best_face = max(results, key=lambda x: x['confidence'])
    x, y, w, h = best_face['box']
    
    # Expand box by 20% to include more context
    margin = 0.2
    x = max(0, int(x - margin * w))
    y = max(0, int(y - margin * h))
    w = int(w * (1 + 2 * margin))
    h = int(h * (1 + 2 * margin))
    
    # Crop and resize
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), 
                    interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize using ImageNet stats (matches training)
    face = face.astype(np.float32) / 255.0

    return face

# Generate embeddings
embeddings = {}
for identity in tqdm(os.listdir(KNOWN_FACES_DIR), desc="Processing identities"):
    identity_dir = os.path.join(KNOWN_FACES_DIR, identity)
    if not os.path.isdir(identity_dir):
        continue
    
    identity_embeddings = []
    for img_file in tqdm(os.listdir(identity_dir), 
                    desc=f"{identity[:12]}...", leave=False):
        img_path = os.path.join(identity_dir, img_file)
        
        try:
            # Load and preprocess
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            face = preprocess_face(image)
            if face is None:
                continue
                
            # Convert to tensor
            tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # Generate embedding
            with torch.no_grad():
                embedding = model(tensor).cpu().numpy()
            
            identity_embeddings.append(embedding)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    if identity_embeddings:
        # Use mean of L2-normalized embeddings
        embeddings[identity] = np.mean(
            [e / np.linalg.norm(e) for e in identity_embeddings], 
            axis=0
        )

# Save embeddings
np.save(EMBEDDINGS_PATH, embeddings)
print(f"\n✅ Saved embeddings for {len(embeddings)} identities to {EMBEDDINGS_PATH}")