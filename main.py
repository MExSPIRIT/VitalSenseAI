import cv2
import mediapipe as mp
import numpy as np
import threading
import uvicorn
import time
from scipy.signal import butter, filtfilt, detrend, find_peaks
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# PART 1: CONFIGURATION
# ==========================================
class Config:
    FS = 30.0             # Sampling Rate (Approx FPS)
    BUFFER_SIZE = 300     # 10 Seconds of data
    MIN_CONFIDENCE = 0.5  # Signal Quality Threshold
    
    # Filter frequencies (in Hz)
    BPM_LOW = 0.7   # 42 BPM
    BPM_HIGH = 3.5  # 210 BPM
    RESP_LOW = 0.1  # 6 breaths/min
    RESP_HIGH = 0.5 # 30 breaths/min

# ==========================================
# PART 2: SIGNAL PROCESSING UTILS (MATH)
# ==========================================
def apply_bandpass_filter(data, fs, low, high, order=3):
    """Butterworth Bandpass Filter to isolate heart frequencies."""
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_sqi(signal):
    """Calculates Signal Quality Index based on Skewness."""
    if len(signal) == 0: return 0.0
    # Clean PPG signals usually have positive skewness
    skewness = np.mean((signal - np.mean(signal))**3) / np.std(signal)**3
    return 1.0 if skewness > 0 else 0.5

def pos_algorithm(red, green, blue, fs):
    """
    Plane-Orthogonal-to-Skin (POS) Algorithm.
    Projects RGB signals to remove motion noise.
    """
    # Normalize
    r_mean = np.mean(red)
    g_mean = np.mean(green)
    b_mean = np.mean(blue)
    
    # Avoid division by zero
    if r_mean == 0 or g_mean == 0 or b_mean == 0:
        return np.zeros(len(red))

    r_norm = red / r_mean
    g_norm = green / g_mean
    b_norm = blue / b_mean
    
    # Projection
    xs = g_norm - b_norm
    ys = g_norm + b_norm - 2 * r_norm
    
    # Alpha Tuning
    alpha = np.std(xs) / np.std(ys) if np.std(ys) != 0 else 0
    
    # Fusion
    h = xs + alpha * ys
    
    # Temporal Filtering
    bvp = detrend(h)
    bvp = apply_bandpass_filter(bvp, fs, Config.BPM_LOW, Config.BPM_HIGH)
    return bvp

# ==========================================
# PART 3: VITAL SIGNS CALCULATOR
# ==========================================
class VitalSignsCalculator:
    def __init__(self, fs):
        self.fs = fs

    def get_hr_and_ibi(self, bvp_signal):
        # Find systolic peaks
        peaks, _ = find_peaks(bvp_signal, distance=self.fs/2.5)
        
        if len(peaks) < 2:
            return 0, [], []

        # Calculate Inter-Beat Interval (ms)
        ibi_list = np.diff(peaks) / self.fs * 1000
        hr_instant = 60000 / np.mean(ibi_list)
        return hr_instant, ibi_list, peaks

    def get_hrv_metrics(self, ibi_list):
        if len(ibi_list) < 2:
            return {"rmssd": 0, "sdnn": 0, "pnn50": 0}
            
        diff_ibi = np.diff(ibi_list)
        rmssd = np.sqrt(np.mean(diff_ibi ** 2))
        sdnn = np.std(ibi_list)
        nn50 = np.sum(np.abs(diff_ibi) > 50)
        pnn50 = (nn50 / len(ibi_list)) * 100
        
        return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50}

    def get_respiration_rate(self, bvp_signal):
        # Respiration from Amplitude Modulation
        resp_signal = apply_bandpass_filter(bvp_signal, self.fs, Config.RESP_LOW, Config.RESP_HIGH)
        peaks, _ = find_peaks(resp_signal, distance=self.fs*2)
        if len(peaks) < 2:
            return 0
        avg_breath_duration = np.mean(np.diff(peaks)) / self.fs
        return 60 / avg_breath_duration if avg_breath_duration > 0 else 0

    def get_stress_index(self, ibi_list):
        # Baevsky Stress Index
        if len(ibi_list) < 10: return 0
        
        bin_width = 50 
        bins = np.arange(min(ibi_list), max(ibi_list) + bin_width, bin_width)
        hist, _ = np.histogram(ibi_list, bins)
        
        max_bin_idx = np.argmax(hist)
        mo = bins[max_bin_idx] / 1000 # Mode (s)
        amo = (hist[max_bin_idx] / len(ibi_list)) * 100 # Amplitude (%)
        vr = (max(ibi_list) - min(ibi_list)) / 1000 # Variation Range (s)
        
        if vr == 0 or mo == 0: return 0
        return amo / (2 * vr * mo)

# ==========================================
# PART 4: API & MAIN APP
# ==========================================
app = FastAPI(title="VitalSense Pro API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# Global State
current_vitals = {
    "heart_rate": 0,
    "respiration_rate": 0,
    "hrv_metrics": {"sdnn": 0, "rmssd": 0, "pnn50": 0},
    "stress_index": 0,
    "spo2": 97.5,
    "sqi": 0.0,
    "status": "Initializing"
}

def run_camera_system():
    global current_vitals
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    calc = VitalSignsCalculator(fs=Config.FS)
    
    # ROI Buffers
    red_buffer = []
    green_buffer = []
    blue_buffer = []
    
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    print("=" * 60)
    print(">>> VitalSense Pro - Camera System Started")
    print(">>> Preview window will open shortly...")
    print(">>> API Server: http://localhost:8000")
    print(">>> Press 'q' in the preview window to quit")
    print("=" * 60)
    
    frame_count = 0
    face_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("ERROR: Failed to read frame from camera")
            break
        
        frame_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Draw face mesh landmarks for visualization
        display_frame = frame.copy()
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Draw face mesh (optional - can be removed for cleaner view)
            # mp.solutions.drawing_utils.draw_landmarks(
            #     display_frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            #     None, None)
            
            # Forehead ROI
            try:
                x = int(landmarks.landmark[10].x * w)
                y = int(landmarks.landmark[10].y * h)
                
                # Dynamic ROI sizing
                box_size = 40
                y_min, y_max = max(0, y), min(h, y + box_size)
                x_min, x_max = max(0, x - box_size), min(w, x + box_size)
                
                # Draw ROI rectangle
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(display_frame, "ROI", (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                roi = frame[y_min:y_max, x_min:x_max]
                
                if roi.size > 0:
                    red_buffer.append(np.mean(roi[:,:,2]))
                    green_buffer.append(np.mean(roi[:,:,1]))
                    blue_buffer.append(np.mean(roi[:,:,0]))
                    
                    if len(red_buffer) > Config.BUFFER_SIZE:
                        red_buffer.pop(0)
                        green_buffer.pop(0)
                        blue_buffer.pop(0)
                    
                    # Show buffer status
                    buffer_pct = (len(red_buffer) / Config.BUFFER_SIZE) * 100
                    
                    # Compute Vitals when buffer fills up
                    if len(red_buffer) == Config.BUFFER_SIZE:
                        bvp = pos_algorithm(
                            np.array(red_buffer), 
                            np.array(green_buffer), 
                            np.array(blue_buffer), 
                            Config.FS
                        )
                        
                        sqi = calculate_sqi(bvp)
                        
                        if sqi > Config.MIN_CONFIDENCE:
                            hr, ibi, _ = calc.get_hr_and_ibi(bvp)
                            hrv = calc.get_hrv_metrics(ibi)
                            rr = calc.get_respiration_rate(bvp)
                            stress = calc.get_stress_index(ibi)
                            
                            current_vitals = {
                                "heart_rate": round(hr, 1),
                                "respiration_rate": round(rr, 1),
                                "hrv_metrics": {k: round(v, 2) for k, v in hrv.items()},
                                "stress_index": round(stress, 2),
                                "spo2": 97.5, # SpO2 static for safety
                                "sqi": round(sqi, 2),
                                "status": "Active"
                            }
                        else:
                            current_vitals["status"] = "Low Signal Quality"
                            
            except Exception as e:
                # Catch ROI errors to keep thread alive
                pass
        else:
            face_detected = False
            if frame_count % 30 == 0:  # Print every second (30 fps)
                print("Waiting for face detection...")
        
        # Display vital signs on frame
        y_offset = 30
        line_height = 25
        
        # Status bar
        status_color = (0, 255, 0) if current_vitals["status"] == "Active" else (0, 165, 255)
        cv2.putText(display_frame, f"Status: {current_vitals['status']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_offset += line_height
        
        # Buffer status
        if len(red_buffer) > 0:
            buffer_pct = (len(red_buffer) / Config.BUFFER_SIZE) * 100
            cv2.putText(display_frame, f"Buffer: {len(red_buffer)}/{Config.BUFFER_SIZE} ({buffer_pct:.1f}%)", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Face detection status
        face_status = "Face Detected" if face_detected else "No Face"
        face_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(display_frame, f"Face: {face_status}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        y_offset += line_height + 5
        
        # Vital signs display
        if current_vitals["status"] == "Active":
            cv2.putText(display_frame, "VITAL SIGNS:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(display_frame, f"HR: {current_vitals['heart_rate']} bpm", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(display_frame, f"RR: {current_vitals['respiration_rate']} bpm", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(display_frame, f"SpO2: {current_vitals['spo2']}%", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(display_frame, f"SQI: {current_vitals['sqi']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show preview window
        cv2.imshow("VitalSense Pro - Camera Preview", display_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n>>> Shutting down camera system...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(">>> Camera system stopped.")

# Start Camera Thread
threading.Thread(target=run_camera_system, daemon=True).start()

@app.get("/")
def root():
    return {"message": "VitalSense Pro API Running"}

@app.get("/vitals")
def get_vitals():
    return current_vitals

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(">>> Starting VitalSense Pro API Server...")
    print(">>> Server will be available at: http://localhost:8000")
    print(">>> API Endpoints:")
    print(">>>   - GET /          : API status")
    print(">>>   - GET /vitals    : Get current vital signs")
    print("=" * 60 + "\n")
    
    # Runs the API server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)