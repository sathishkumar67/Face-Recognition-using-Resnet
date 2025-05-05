from flask import Flask, request, render_template_string
import os
import cv2
import numpy as np
import torch
from mtcnn import MTCNN
from siamese_resnet.model import SiameseResNet

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'state_dict.pt'
EMBEDDINGS_PATH = 'embeddings_db.npy'
IMG_SIZE = 224
THRESHOLD = 0.6  # Distance threshold for recognition

# Initialize Flask app
app = Flask(__name__)

# Load model
model = SiameseResNet(embedding_dim=256).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return data.item()  # dict: identity -> embedding

embeddings_db = load_embeddings(EMBEDDINGS_PATH)

# Initialize face detector
detector = MTCNN()

def preprocess_face(image):
    results = detector.detect_faces(image)
    if not results:
        return None
    best = max(results, key=lambda x: x['confidence'])
    x, y, w, h = best['box']
    margin = 0.2
    x = max(0, int(x - margin * w))
    y = max(0, int(y - margin * h))
    w = int(w * (1 + 2 * margin))
    h = int(h * (1 + 2 * margin))
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    face = face.astype(np.float32) / 255.0
    return face

@app.route('/', methods=['GET'])
def index():
    return render_template_string('''
    <!doctype html>
    <title>Face Recognition</title>
    <h1>Upload an image for recognition</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <input type="submit" value="Upload">
    </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return 'No file provided', 400
    # Read image from memory
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = preprocess_face(image)
    if face is None:
        return 'No face detected', 200
    # Prepare tensor
    tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    # Compare to database
    distances = {name: np.linalg.norm(emb - db_emb) for name, db_emb in embeddings_db.items()}
    best_match = min(distances, key=distances.get)
    best_distance = distances[best_match]
    if best_distance < THRESHOLD:
        label = f"Matched: {best_match} (distance: {best_distance:.3f})"
    else:
        label = f"Unknown (closest: {best_match}, distance: {best_distance:.3f})"
    return render_template_string('''
    <!doctype html>
    <title>Result</title>
    <h1>{{label}}</h1>
    <a href="/">Try another image</a>
    ''', label=label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
