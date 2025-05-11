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
    """
    Mengekstrak koordinat ROI (Region of Interest) berbasis landmark wajah.

    Args:
        landmarks (List[Landmark]): Daftar titik landmark dari wajah.
        indices (List[int]): Indeks titik-titik landmark yang digunakan untuk mendefinisikan ROI.
        image_width (int): Lebar gambar.
        image_height (int): Tinggi gambar.

    Returns:
        Tuple[int, int, int, int]: Koordinat bounding box ROI dalam bentuk (x_min, y_min, x_max, y_max).
    """
    # Ambil titik-titik landmark yang relevan
    points = [landmarks[idx] for idx in indices]

    # Konversi ke koordinat piksel
    coords = [(int(point.x * image_width), int(point.y * image_height)) for point in points]

    # Tentukan batas ROI (bounding box)
    min_x, max_x = min(pt[0] for pt in coords), max(pt[0] for pt in coords)
    min_y, max_y = min(pt[1] for pt in coords), max(pt[1] for pt in coords)

    return max(0, min_x), max(0, min_y), min(image_width, max_x), min(image_height, max_y)

def cpu_POS(signal, fps):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':int}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """

    """
    eps: Konstanta kecil (10^-9) untuk mencegah pembagian dengan nol pada proses normalisasi.
    X: Sinyal masukan, berupa array 3 dimensi di mana:
       e: Jumlah estimasi atau ROI (region of interest) di dalam frame (misalnya bagian hidung).
       c: Kanal warna (3 untuk R, G, B).
       f: Jumlah frame.
    w: Panjang jendela (window length), ditentukan berdasarkan frame rate kamera (fps).
       Misalnya pada fps = 30, maka w = 48 frame (~1.6 detik video).
    """
    eps = 1e-9
    X = signal
    e, c, f = X.shape       # Jumlah ROI, kanal warna, dan jumlah frame
    w = int(1.6 * fps)      # Panjang jendela (window length) untuk estimasi sinyal rPPG

    """
    P: Matriks 2x3 tetap yang digunakan untuk proyeksi. Matriks ini mendefinisikan cara mentransformasi warna RGB ke ruang proyeksi baru.
    Q: Stack dari matriks P yang diulang sebanyak jumlah ROI (e). Setiap ROI akan memiliki satu salinan P.
    """
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    """
    H: Matriks hasil untuk menyimpan sinyal denyut nadi (rPPG) dari setiap ROI sepanjang waktu.
    n: Frame saat ini dalam proses sliding window.
    m: Indeks awal dari sliding window, digunakan untuk menentukan frame-frame yang akan diproses saat ini.
    """
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Indeks awal jendela saat ini (4)
        m = n - w + 1

        """
        Normalisasi temporal (Persamaan 5 dari paper):
        Langkah ini bertujuan agar sinyal menjadi tahan terhadap perubahan pencahayaan global dan noise lainnya.
        """
        Cn = X[:, :, m:(n + 1)]                 # Potongan sinyal RGB selama w frame
        M = 1.0 / (np.mean(Cn, axis=2) + eps)   # Hitung rata-rata per kanal warna, lalu dibalik
        M = np.expand_dims(M, axis=2)           # Ubah dimensi agar bisa dikalikan dengan benar [e, c, w]
        Cn = np.multiply(M, Cn)                 # Lakukan normalisasi

        """
        Proyeksi (Persamaan 6):
        Transformasi nilai RGB ke ruang proyeksi agar sinyal aliran darah (denyut nadi) lebih menonjol.
        """
        S = np.dot(Q, Cn)    # Proyeksi warna   
        S = S[0, :, :, :]    # Ambil satu ROI (karena biasanya hanya satu)
        S = np.swapaxes(S, 0, 1) # Ubah urutan dimensi agar sesuai dengan format [w, e, c]

        """
        Penyesuaian (Tuning) (Persamaan 7):
        Menyesuaikan dua komponen proyeksi agar sinyal denyut nadi lebih jelas dan dominan.
        """
        S1, S2 = S[:, 0, :], S[:, 1, :]     # Ambil dua komponen proyeksi
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1)) # Hitung rasio standar deviasi antara dua komponen
        alpha = np.expand_dims(alpha, axis=1)   # Ubah dimensi agar bisa dikalikan dengan benar [e, 1]
        Hn = np.add(S1, alpha * S2)             # Gabungkan dua komponen proyeksi dengan rasio alpha
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)  # Kurangi rata-rata agar sinyal terpusat di nol

        """
        Penambahan overlap (Persamaan 8):
        Gabungkan sinyal yang sudah dituning dari masing-masing window ke dalam sinyal akhir.
        """
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)  # Tambahkan hasil ke segmen window yang sesuai
        
    return H

def generate_frames():
    """
    Generator untuk menghasilkan frame dari kamera secara real-time.
    Jika sedang merekam (`is_recording` True), maka ROI wajah akan disimpan ke dalam list `frames`.
    ROI diambil berdasarkan landmark wajah dan digambarkan dalam kotak hijau.
    Hasil frame dikodekan dalam format JPEG untuk streaming (misal pada aplikasi Flask).
    """
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
                    
                    # Gambar kotak ROI pada wajah
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Simpan ROI jika sedang merekam dan ROI valid
                    if is_recording and roi.size > 0:
                        frames.append(roi)
            
            # Encode frame ke JPEG dan yield untuk streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def start_recording():
    """
    Memulai proses perekaman ROI dari kamera dan memulai thread pemrosesan rPPG secara live.
    Mengatur ulang variabel global dan menjalankan thread latar belakang.
    """
    global frames, is_recording, start_time, pulse_rate, camera_thread

    # Inisialisasi variabel global
    frames = []
    is_recording = True
    start_time = time.time()
    pulse_rate = None
    
    # Memulai thread latar belakang untuk pemrosesan rPPG
    camera_thread = threading.Thread(target=process_rppg_live, daemon=True)
    camera_thread.start()
    
    return jsonify({"status": "Recording started"})

def stop_recording():
    """
    Menghentikan proses perekaman ROI dari kamera.
    Mengubah status is_recording menjadi False.
    """
    global is_recording
    is_recording = False
    return jsonify({"status": "Recording stopped"})

def process_rppg_live():
    """
    Memproses sinyal rPPG secara langsung dari frame yang direkam dalam waktu tertentu,
    menerapkan filter bandpass, normalisasi, dan menghitung pulse rate dari puncak sinyal.
    """
    global frames, rppg_signal, is_recording, pulse_rate

    while is_recording:
        # Periksa apakah jumlah frame cukup untuk diproses
        if len(frames) >= FPS * PROCESS_INTERVAL:
            # Ambil segmen frame terbaru untuk diproses
            segment = frames[-FPS * PROCESS_INTERVAL:]

            # Ekstraksi rata-rata nilai RGB dari setiap frame
            r_signals = [np.mean(f[:, :, 0]) for f in segment]
            g_signals = [np.mean(f[:, :, 1]) for f in segment]
            b_signals = [np.mean(f[:, :, 2]) for f in segment]

            # Bentuk sinyal RGB menjadi array 3D
            rgb_signals = np.array([r_signals, g_signals, b_signals])
            rgb_signals = rgb_signals.reshape(1, 3, -1)

            # Proses hanya jika panjang sinyal cukup
            if rgb_signals.shape[2] > int(1.6 * FPS):
                # Hitung sinyal rPPG menggunakan metode POS
                raw_rppg = cpu_POS(rgb_signals, fps=FPS).reshape(-1)

                # Terapkan filter bandpass untuk menghilangkan noise
                b, a = signal.butter(3, [0.9, 2.4], btype='band', fs=FPS)
                filtered = signal.filtfilt(b, a, raw_rppg)

                # Normalisasi Z-score
                filtered = (filtered - np.mean(filtered)) / (np.std(filtered))
                
                # Deteksi puncak sinyal untuk menghitung pulse rate
                peaks, _ = signal.find_peaks(filtered, prominence=0.5)
                if len(peaks) > 1:
                    peak_times = np.array(peaks) / FPS  # Konversi ke detik
                    rr_intervals = np.diff(peak_times)
                    avg_rr = np.mean(rr_intervals)
                    if avg_rr > 0:
                        pulse_rate = round(60 / avg_rr)  # Presisi desimal

                # Simpan sinyal rPPG yang sudah difilter untuk ditampilkann 
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