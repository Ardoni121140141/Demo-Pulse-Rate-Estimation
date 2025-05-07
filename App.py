from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import scipy.signal as signal
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Global variables for rPPG signal
frames = []
rppg_signal = []
is_recording = False
start_time = None
pulse_rate = None
FPS = 30  # Frames per second
PROCESS_INTERVAL = 2  # Process every 2 second
camera_thread = None

def extract_roi_coordinates(landmarks, indices, image_width, image_height):
    points = [landmarks[idx] for idx in indices]
    coords = [(int(point.x * image_width), int(point.y * image_height)) for point in points]
    min_x, max_x = min(pt[0] for pt in coords), max(pt[0] for pt in coords)
    min_y, max_y = min(pt[1] for pt in coords), max(pt[1] for pt in coords)
    return max(0, min_x), max(0, min_y), min(image_width, max_x), min(image_height, max_y)

def cpu_POS(signal, fps):
    eps = 1e-9
    X = signal
    e, c, f = X.shape
    w = int(1.6 * fps)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(M, Cn)
        S = np.dot(Q, Cn)[0, :, :, :]
        S = np.swapaxes(S, 0, 1)
        S1, S2 = S[:, 0, :], S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
    return H

def generate_frames():
    global frames, is_recording, start_time, pulse_rate
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = extract_roi_coordinates(face_landmarks.landmark, [232, 98, 122, 452, 327, 326], w, h)
                    roi = frame[y1:y2, x1:x2]
                    
                    # Draw ROI bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if is_recording and roi.size > 0:
                        frames.append(roi)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def start_recording():
    global frames, is_recording, start_time, pulse_rate, camera_thread
    frames = []
    is_recording = True
    start_time = time.time()
    pulse_rate = None
    
    # Start background thread for processing
    camera_thread = threading.Thread(target=process_rppg_live, daemon=True)
    camera_thread.start()
    
    return jsonify({"status": "Recording started"})

def stop_recording():
    global is_recording
    is_recording = False
    return jsonify({"status": "Recording stopped"})

def process_rppg_live():
    global frames, rppg_signal, is_recording, pulse_rate
    while is_recording:
        if len(frames) >= FPS * PROCESS_INTERVAL:
            segment = frames[-FPS * PROCESS_INTERVAL:]
            r_signals = [np.mean(f[:, :, 0]) for f in segment]
            g_signals = [np.mean(f[:, :, 1]) for f in segment]
            b_signals = [np.mean(f[:, :, 2]) for f in segment]

            rgb_signals = np.array([r_signals, g_signals, b_signals])
            rgb_signals = rgb_signals.reshape(1, 3, -1)

            if rgb_signals.shape[2] > int(1.6 * FPS):
                raw_rppg = cpu_POS(rgb_signals, fps=FPS).reshape(-1)

                # Bandpass filter
                b, a = signal.butter(3, [0.9, 2.4], btype='band', fs=FPS)
                filtered = signal.filtfilt(b, a, raw_rppg)

                # Z-normalization
                filtered = (filtered - np.mean(filtered)) / (np.std(filtered))

                # Deteksi peak untuk hitung pulse rate
                # peaks, _ = signal.find_peaks(filtered, distance=FPS * 0.5, prominence=0.5)
                peaks, _ = signal.find_peaks(filtered, prominence=0.5)
                if len(peaks) > 1:
                    peak_times = np.array(peaks) / FPS  # Konversi ke detik
                    rr_intervals = np.diff(peak_times)
                    avg_rr = np.mean(rr_intervals)
                    if avg_rr > 0:
                        pulse_rate = round(60 / avg_rr)  # Presisi desimal


                rppg_signal = filtered.tolist()

        time.sleep(PROCESS_INTERVAL)

@app.route('/get_pulse_rate')
def get_pulse_rate():
    global pulse_rate
    return jsonify({"pulse_rate": pulse_rate if pulse_rate else "Processing..."})

@app.route('/rppg_plot')
def rppg_plot():
    global rppg_signal
    return jsonify({"signal": rppg_signal})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start():
    return start_recording()

@app.route('/stop_recording', methods=['POST'])
def stop():
    return stop_recording()

if __name__ == '__main__':
    app.run(debug=True)