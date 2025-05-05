from flask import Flask, Response, render_template_string
import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from siamese_resnet.model import SiameseResNet
import threading

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'state_dict.pt'
EMBEDDINGS_PATH = 'embeddings_db.npy'
IMG_SIZE = 224
THRESHOLD = 5  # Distance threshold for recognition

# Initialize Flask app
app = Flask(__name__)

# Load Siamese model
model = SiameseResNet(embedding_dim=256).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Load embeddings database and convert to torch tensors
def load_embeddings(path):
    data = np.load(path, allow_pickle=True).item()
    return {name: torch.tensor(emb, device=device) for name, emb in data.items()}

embeddings_db = load_embeddings(EMBEDDINGS_PATH)

# Initialize face detector
detector = MTCNN()

# Video capture
camera = cv2.VideoCapture(0)
lock = threading.Lock()

# Preprocess face crop for model input
def preprocess_face(face_img):
    face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    face = face.astype(np.float32) / 255.0
    tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

# Generate MJPEG stream
def gen_frames():
    while True:
        with lock:
            success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        for res in faces:
            x, y, w, h = res['box']
            margin = 0.2
            x1 = max(0, int(x - margin * w))
            y1 = max(0, int(y - margin * h))
            x2 = min(frame.shape[1], int(x + w + margin * w))
            y2 = min(frame.shape[0], int(y + h + margin * h))
            crop = rgb[y1:y2, x1:x2]

            inp = preprocess_face(crop)
            with torch.no_grad():
                emb = model(inp)  # shape [1, dim]

            # Compute distances using raw embeddings
            distances = {name: torch.norm(emb - db_emb) for name, db_emb in embeddings_db.items()}
            best_name, best_dist = min(distances.items(), key=lambda x: x[1])
            dist_val = best_dist.item()
            label = best_name if dist_val < THRESHOLD else 'Unknown'

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({dist_val:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Live Face Recognition</title>
    </head>
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
