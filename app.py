from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import torch
from mtcnn import MTCNN
from siamese_resnet.model import SiameseResNet
import threading

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

# Load embeddings database
def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return data.item()  # dict: identity -> embedding

embeddings_db = load_embeddings(EMBEDDINGS_PATH)

# Initialize face detector
detector = MTCNN()

# Video capture
camera = cv2.VideoCapture(0)
lock = threading.Lock()

# Preprocessing helper
def preprocess_face(image):
    results = detector.detect_faces(image)
    if not results:
        return None
    # pick highest confidence
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

# Frame generator for video streaming
def gen_frames():
    while True:
        with lock:
            success, frame = camera.read()
        if not success:
            break
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = preprocess_face(rgb)
        label = 'No face'
        if face is not None:
            tensor = torch.from_numpy(face).permute(2,0,1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model(tensor).cpu().numpy()
            emb = emb / np.linalg.norm(emb)
            # compute distances
            distances = {name: np.linalg.norm(emb - db_emb) for name, db_emb in embeddings_db.items()}
            best_match = min(distances, key=distances.get)
            best_distance = distances[best_match]
            if best_distance < THRESHOLD:
                label = f"{best_match}: {best_distance:.2f}"
            else:
                label = f"Unknown: {best_distance:.2f}"
        # annotate
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <html>
      <head><title>Live Face Recognition</title></head>
      <body>
        <h1>Live Face Recognition</h1>
        <img src="/video_feed" width="720" />
      </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)