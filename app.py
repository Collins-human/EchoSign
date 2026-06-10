import streamlit as st
import cv2
import av
import mediapipe as mp
import numpy as np
import os
import json
import tempfile
import pandas as pd
import random
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc  

# Try importing WebRTC packages safely to prevent hard failures if not installed yet
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# --- 1. CORE CONFIG & ADVANCED CSS ---
st.set_page_config(page_title="EchoSign Pro", page_icon="🤟", layout="wide")

st.markdown("""
    <style>
    /* Global Font and Header */
    html, body, [class*="css"] { font-size: 19px !important; }
    [data-testid="stHeader"] { background-color: #FF8C00 !important; }
    
    .main-header {
        background-color: #FF8C00; padding: 25px; border-radius: 12px;
        color: white; text-align: center; margin-bottom: 25px;
    }

    /* Sidebar Styling - Solid Black */
    [data-testid="stSidebar"] { 
        background-color: #000000 !important; 
    }
    [data-testid="stSidebar"] * { color: white !important; }
    
    /* Sidebar Button Styling */
    [data-testid="stSidebar"] .stButton button {
        background-color: #1E1E1E !important; 
        color: white !important;
        border: 1px solid #FF8C00 !important;
        font-weight: bold;
    }

    /* Join Community Box */
    .join-community-box {
        background-color: #FFF3E0;
        border: 2px dashed #FF8C00;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 30px;
    }

    /* Result Card */
    .detected-card {
        background: #1E1E1E; color: #FF8C00; padding: 2rem;
        border-radius: 15px; text-align: center; border: 2px solid #FF8C00;
    }
    .word-main { font-size: 60px; font-weight: bold; }
    
    /* Interactive Challenge Card */
    .challenge-card {
        background: linear-gradient(135deg, #FF8C00, #FF5722);
        color: white; padding: 2.5rem; border-radius: 20px;
        text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }
    .challenge-word { font-size: 70px; font-weight: 900; letter-spacing: 2px; }

    /* DYNAMIC FOOTER */
    .dynamic-footer {
        background-color: #000000;
        color: white;
        text-align: center;
        padding: 40px 20px;
        margin-top: 60px;
        border-top: 5px solid #FF8C00;
        line-height: 1.6;
        width: 100%;
    }
    .footer-name { color: #FF8C00; font-size: 22px; font-weight: bold; }

    /* Community Chat Bubbles */
    .chat-bubble {
        background-color: #f1f1f1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF8C00;
        margin-bottom: 10px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

@st.cache_resource
def load_asl_model():
    return load_model('asl_model_final.h5')

model = load_asl_model()
labels = sorted(['AGAIN', 'ANGRY', 'BAD', 'BOOK', 'BROTHER', 'CAR', 'COME', 'COMPUTER', 
                 'FATHER', 'FRIEND', 'GO', 'GOOD', 'HAPPY', 'HELLO', 'HELP', 'HOUSE', 
                 'LIKE', 'MAN', 'MORE', 'MOTHER', 'NAME', 'NO', 'PHONE', 'PLAY', 
                 'PLEASE', 'SAD', 'SISTER', 'SLEEP', 'SORRY', 'STOP', 'WAIT', 'WATER', 
                 'WHEN', 'WHERE', 'WHO', 'WHY', 'WORK', 'YES'])

# --- CHAT PERSISTENCE HELPERS ---
CHAT_FILE = "shared_community_chat.json"

def load_community_messages():
    if not os.path.exists(CHAT_FILE):
        default_posts = [{"user": "Admin", "msg": "Welcome to EchoSign Community Chat!", "time": "10:00 AM"}]
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(default_posts, f, indent=4)
        return default_posts
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return [{"user": "Admin", "msg": "Welcome to EchoSign Community Chat!", "time": "10:00 AM"}]

def save_community_message(user, msg):
    current_posts = load_community_messages()
    new_post = {
        "user": user.strip() if user.strip() else "Anonymous",
        "msg": msg.strip(),
        "time": datetime.now().strftime("%I:%M %p")
    }
    current_posts.append(new_post)
    try:
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(current_posts, f, indent=4)
    except Exception:
        pass


def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        frame_counter += 1
        if frame_counter % 2 == 0: 
            continue
            
        frame = cv2.resize(frame, (320, 240))  
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        frame_coords = np.zeros(126) 
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i < 2: 
                    coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                    frame_coords[i*63 : i*63 + 63] = coords
                    
        # FIXED: Appending is now outside the IF statement so empty frames are properly counted
        sequence.append(frame_coords)
        
    cap.release()
    gc.collect()
    return sequence

def process_video(video_path, filename):
    raw_sequence = extract_landmarks(video_path)
    
    if not raw_sequence or np.all(np.array(raw_sequence) == 0):
        return {"File": filename, "Prediction": "NO HANDS DETECTED", "Confidence": 0.0}

    valid_frames = [frame for frame in raw_sequence if not np.all(frame == 0)]
    hand_ratio = len(valid_frames) / len(raw_sequence) if len(raw_sequence) > 0 else 0
    
    if hand_ratio < 0.15:
        return {"File": filename, "Prediction": "NO HANDS DETECTED", "Confidence": 0.0}

    if len(valid_frames) > 1:
        max_movement = np.max(np.ptp(valid_frames, axis=0))
        if max_movement < 0.02:
            return {"File": filename, "Prediction": "STATIC IMAGE REJECTED", "Confidence": 0.0}

    best_conf, best_idx = 0, 0
    window_predictions = [] 
    
    if len(raw_sequence) < 30:
        input_data = pad_sequences([raw_sequence], maxlen=30, padding='post', dtype='float32')
        res = model(input_data, training=False).numpy()[0]
        best_idx = np.argmax(res)
        best_conf = res[best_idx]
        window_predictions.append(best_idx)
    else:
        for start in range(0, len(raw_sequence) - 30, 8):
            window = raw_sequence[start : start + 30]
            input_data = np.expand_dims(window, axis=0)
            res = model(input_data, training=False).numpy()[0]
            
            current_idx = np.argmax(res)
            window_predictions.append(current_idx) 
            
            if res[current_idx] > best_conf:
                best_idx = current_idx
                best_conf = res[current_idx]

    if len(window_predictions) > 2:
        vote_count = window_predictions.count(best_idx)
        consistency_ratio = vote_count / len(window_predictions)
        if consistency_ratio < 0.30:
            return {"File": filename, "Prediction": "RANDOM MOVEMENT REJECTED", "Confidence": round(best_conf*100, 1)}
                
    CONFIDENCE_LIMIT = 0.60  
    if best_conf < CONFIDENCE_LIMIT:
        final_prediction = "UNRECOGNIZED"
    else:
        final_prediction = labels[best_idx]

    return {"File": filename, "Prediction": final_prediction, "Confidence": round(best_conf*100, 1)}


# --- REAL-TIME WEBCAM PROCESSING ACCUMULATOR CLASS ---
class RealTimeSignProcessor:
    def __init__(self):
        self.frame_buffer = []
        self.current_prediction = "Waiting..."
        self.current_confidence = 0.0
        self.capture_status = "IDLE ❌" 
        self.frame_counter = 0
        self.prediction_history = []
        self.is_locked = False  

    def recv_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1
        
        if not hasattr(self, 'hands_tracker'):
            self.hands_tracker = mp.solutions.hands.Hands(
                static_image_mode=False, 
                max_num_hands=2, 
                min_detection_confidence=0.40,
                min_tracking_confidence=0.50,
                model_complexity=1 
            )
        
        if not self.is_locked:
            if self.frame_counter % 2 == 0:
                small_frame = cv2.resize(img, (320, 240))
                image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = self.hands_tracker.process(image_rgb)
                
                frame_coords = np.zeros(126)
                if results.multi_hand_landmarks:
                    self.capture_status = "SCANNING... 🎥"
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        if i < 2:
                            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                            frame_coords[i*63 : i*63 + 63] = coords
                else:
                    self.capture_status = "IDLE ❌"
                
                self.frame_buffer.append(frame_coords)
                if len(self.frame_buffer) > 30:
                    self.frame_buffer.pop(0) 
                    
                if len(self.frame_buffer) == 30 and self.frame_counter % 10 == 0:
                    valid_frames = [f for f in self.frame_buffer if not np.all(f == 0)]
                    hand_ratio = len(valid_frames) / 30.0
                    
                    if hand_ratio < 0.20:
                        self.current_prediction = "Waiting for hands..."
                        self.current_confidence = 0.0
                        self.capture_status = "IDLE ❌"
                    else:
                        self.capture_status = "CAPTURED ✅"
                        input_data = np.expand_dims(self.frame_buffer, axis=0)
                        
                        res = model(input_data, training=False).numpy()[0]
                        idx = np.argmax(res)
                        conf = res[idx]
                        
                        self.prediction_history.append(idx)
                        if len(self.prediction_history) > 6:
                            self.prediction_history.pop(0)
                            
                        vote_count = self.prediction_history.count(idx)
                        consistency_ratio = vote_count / len(self.prediction_history)
                        
                        if conf < 0.60:
                            self.current_prediction = "UNRECOGNIZED"
                            self.current_confidence = round(conf * 100, 1)
                        elif consistency_ratio < 0.35 and len(self.prediction_history) >= 4:
                            self.current_prediction = "RANDOM MOVEMENT REJECTED"
                            self.current_confidence = round(conf * 100, 1)
                        else:
                            self.current_prediction = labels[idx]
                            self.current_confidence = round(conf * 100, 1)
                        
                        self.is_locked = True  
                        
        cv2.putText(img, f"STATUS: {self.capture_status}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        cv2.putText(img, f"SIGN: {self.current_prediction}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 3. SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = "Translator"
if 'results_data' not in st.session_state: st.session_state.results_data = []
if 'validation_errors' not in st.session_state: st.session_state.validation_errors = [] 
if 'target_word' not in st.session_state:
    st.session_state.target_word = random.choice(labels)
if 'learn_result' not in st.session_state:
    st.session_state.learn_result = None

if 'live_processor_instance' not in st.session_state:
    st.session_state.live_processor_instance = RealTimeSignProcessor()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🤟 EchoSign Menu")
    if st.button("🏠 Translator Studio", use_container_width=True):
        st.session_state.page = "Translator"
        st.rerun()
    if st.button("🎯 Sign-Along Learn Studio", use_container_width=True):
        st.session_state.page = "Learn"
        st.rerun()
    if st.button("👥 Community Chat", use_container_width=True):
        st.session_state.page = "Community"
        st.rerun()
    
    st.divider()
    st.markdown("### Dictionary")
    st.write(", ".join(labels[:12]) + "...")

# --- 5. MAIN PAGE CONTENT ---

# --- TAB 1: TRANSLATOR STUDIO ---
if st.session_state.page == "Translator":
    st.markdown('<div class="main-header"><h1>EchoSign: ASL Sentence Studio</h1></div>', unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 1.4, 1.1], gap="large")

    with col_l:
        st.subheader("📤 1. Upload Sequence")
        st.info("⏱️ Note: Please ensure each video is under 10 seconds for optimal processing.")
        
        files = st.file_uploader("Upload in sentence order", type=["mp4", "mov"], accept_multiple_files=True)
        
        current_file_names = [f.name for f in files] if files else []
        if 'last_uploaded_names' not in st.session_state:
            st.session_state.last_uploaded_names = []
            
        if current_file_names != st.session_state.last_uploaded_names:
            st.session_state.results_data = []  
            st.session_state.validation_errors = [] 
            st.session_state.last_uploaded_names = current_file_names
            st.rerun()
            
        if st.session_state.validation_errors:
            for error_msg in st.session_state.validation_errors:
                if "too long" in error_msg:
                    st.error(error_msg)
                else:
                    st.warning(error_msg)
            
        if files and st.button("🚀 CONSTRUCT SENTENCE", type="primary", use_container_width=True):
            st.session_state.results_data = []
            st.session_state.validation_errors = [] 
            
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t:
                    t.write(f.read())
                    temp_path = t.name
                
                cap = cv2.VideoCapture(temp_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                if duration > 10.0:
                    st.session_state.validation_errors.append(f"⚠️ '{f.name}' is too long ({duration:.1f}s). Skipped.")
                    os.remove(temp_path)
                    continue 
                
                res = process_video(temp_path, f.name)
                
                if res["Prediction"] in ["NO HANDS DETECTED", "UNRECOGNIZED", "STATIC IMAGE REJECTED", "RANDOM MOVEMENT REJECTED"]:
                    st.session_state.validation_errors.append(f"🚫 '{f.name}' was skipped ({res['Prediction']}).")
                else:
                    f.seek(0)
                    res["video_bytes"] = f.read()
                    st.session_state.results_data.append(res)
                
                os.remove(temp_path)
                gc.collect()
            st.rerun()

    with col_c:
        st.subheader("🎥 2. Translation")
        if st.session_state.results_data:
            sentence = " ".join([r['Prediction'] for r in st.session_state.results_data]).lower().capitalize()
            st.info(f"**Full Sentence:** {sentence}.")
            
            names = [r["File"] for r in st.session_state.results_data]
            choice = st.selectbox("Detailed View:", names)
            item = next(x for x in st.session_state.results_data if x["File"] == choice)
            st.markdown(f'<div class="detected-card"><div class="word-main">{item["Prediction"]}</div></div>', unsafe_allow_html=True)
        else:
            st.info("Upload valid ASL videos to see results.")

    with col_r:
        st.subheader("📺 3. Video Window")
        if st.session_state.results_data:
            if "video_bytes" in item:
                st.video(item["video_bytes"])
            else:
                st.info("No video track found for this file.")
        elif files:
            st.write("⏳ Ready to process. Select file preview:")
            file_names = [f.name for f in files]
            preview_choice = st.selectbox("Staged Videos:", file_names, label_visibility="collapsed")
            selected_file = next(f for f in files if f.name == preview_choice)
            st.video(selected_file)
        else:
            st.info("Upload files in Step 1 to visually check your frames here before processing.")

# --- TAB 2: INTERACTIVE LEARN STUDIO ---
elif st.session_state.page == "Learn":
    st.markdown('<div class="main-header"><h1>EchoSign: Interactive Learn Studio</h1></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="challenge-card">
            <p style="margin:0; font-size:18px; text-transform:uppercase; opacity:0.9;">Can you sign this word?</p>
            <div class="challenge-word">{st.session_state.target_word}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🔄 Skip / Give Me Another Word", use_container_width=True):
            st.session_state.target_word = random.choice(labels)
            st.session_state.learn_result = None
            proc = st.session_state.live_processor_instance
            proc.current_prediction = "Waiting..."
            proc.current_confidence = 0.0
            proc.frame_buffer = []
            proc.prediction_history = []
            proc.capture_status = "IDLE ❌"
            proc.is_locked = False
            st.rerun()
            
    st.divider()
    col_l, col_r = st.columns([1.1, 1], gap="large")
    
    with col_l:
        st.subheader("🎥 Active Video Stream Input")
        input_mode = st.radio("Choose Input Device Method:", ["Live Browser Webcam Stream 🎥", "Upload Video File Attachment 📤"])
        st.write("")
        
        if input_mode == "Upload Video File Attachment 📤":
            learn_file = st.file_uploader("Record a quick 3-5 second clip performing the sign:", type=["mp4", "mov"])
            if learn_file and st.button("🚀 VERIFY MY FILE GESTURE", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t:
                    t.write(learn_file.read())
                    temp_path = t.name
                
                evaluation = process_video(temp_path, learn_file.name)
                st.session_state.learn_result = evaluation
                
                os.remove(temp_path)
                gc.collect()
                st.rerun()
                
        else:
            if not HAS_WEBRTC:
                st.error("🚨 Dependencies Missing: Run `pip install streamlit-webrtc av` in your environment terminal console.")
            else:
                webrtc_ctx = webrtc_streamer(
                    key="asl-live-streamer",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                    video_frame_callback=st.session_state.live_processor_instance.recv_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True
                )

    with col_r:
        st.subheader("📺 Real-Time Diagnostics Panel")
        
        if input_mode == "Live Browser Webcam Stream 🎥":
            if 'webrtc_ctx' in locals() and webrtc_ctx.state.playing:
                
                @st.fragment(run_every=1)
                def render_live_hud_monitor():
                    proc = st.session_state.live_processor_instance
                    live_guess = proc.current_prediction
                    live_conf = proc.current_confidence
                    live_status = proc.capture_status
                    
                    st.markdown(f"""
                        <div style="background-color: #000000; border: 3px solid #FF8C00; padding: 25px; border-radius: 12px; color: #FF8C00; font-family: sans-serif; margin-bottom: 25px;">
                            <h3 style="color: #FF8C00; margin-top: 0; font-size: 22px; border-bottom: 1px solid #222; padding-bottom: 10px; letter-spacing: 1px;">📺 LIVE HUD MONITOR</h3>
                            <p style="font-size: 18px; margin: 5px 0;">Status: <span style="color: white; font-weight: bold;">{live_status}</span></p>
                            <p style="font-size: 32px; font-weight: bold; margin: 15px 0 5px 0;">Sign: <span style="color: white;">{live_guess}</span></p>
                            <p style="font-size: 20px; margin: 0;">Match: <span style="color: white;">{live_conf}% Accuracy</span></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if live_guess == st.session_state.target_word and live_conf >= 65.0:
                        st.success(f"🎉 **Perfect Match!** Accurate execution context ratings achieved: **{live_conf}%**")
                        st.balloons()
                        if st.button("🔁 Try Again / Practice More", key="try_again_live_success", use_container_width=True):
                            proc.current_prediction = "Waiting..."
                            proc.current_confidence = 0.0
                            proc.frame_buffer = []
                            proc.prediction_history = []
                            proc.capture_status = "IDLE ❌"
                            proc.is_locked = False
                            st.rerun()
                    elif live_guess in ["UNRECOGNIZED", "RANDOM MOVEMENT REJECTED"]:
                        st.error(f"⚠️ **Rejection Exception Flagged:** {live_guess}")
                        if st.button("🔁 Try Again", key="try_again_live_reject", use_container_width=True):
                            proc.current_prediction = "Waiting..."
                            proc.current_confidence = 0.0
                            proc.frame_buffer = []
                            proc.prediction_history = []
                            proc.capture_status = "IDLE ❌"
                            proc.is_locked = False
                            st.rerun()
                    elif live_guess not in ["Waiting...", "Waiting for hands..."]:
                        st.markdown("### 📊 File Evaluation Metrics")
                        st.markdown("### ❌")
                        st.markdown(f"Syntax Variation: Recognized as '{live_guess}' instead of target word context.")
                        
                        if st.button("🔁 Try Again", key="try_again_live_wrong", use_container_width=True):
                            proc.current_prediction = "Waiting..."
                            proc.current_confidence = 0.0
                            proc.frame_buffer = []
                            proc.prediction_history = []
                            proc.capture_status = "IDLE ❌"
                            proc.is_locked = False
                            st.rerun()
                
                render_live_hud_monitor()
            else:
                st.info("Start the webcam streamer tool to view live metrics directly alongside the camera panel.")
                
        elif input_mode == "Upload Video File Attachment 📤" and learn_file:
            st.write("🎥 **Staged Performance Loop Visualizer Output:**")
            st.video(learn_file)
            
            if st.session_state.learn_result:
                st.divider()
                predicted_word = st.session_state.learn_result["Prediction"]
                confidence_score = st.session_state.learn_result["Confidence"]
                
                st.markdown(f"""
                    <div style="background-color: #000000; border: 3px solid #FF8C00; padding: 25px; border-radius: 12px; color: #FF8C00; font-family: sans-serif; margin-bottom: 25px;">
                        <h3 style="color: #FF8C00; margin-top: 0; font-size: 22px; border-bottom: 1px solid #222; padding-bottom: 10px; letter-spacing: 1px;">📋 STATIC EVALUATION MONITOR</h3>
                        <p style="font-size: 32px; font-weight: bold; margin: 15px 0 5px 0;">Sign: <span style="color: white;">{predicted_word}</span></p>
                        <p style="font-size: 20px; margin: 0;">Match: <span style="color: white;">{confidence_score}% Accuracy</span></p>
                    </div>
                """, unsafe_allow_html=True)
                
                if predicted_word == st.session_state.target_word:
                    st.success(f"🎉 **Perfect Match!** Accurate execution context ratings achieved: **{confidence_score}%**")
                    st.balloons()
                elif predicted_word in ["NO HANDS DETECTED", "STATIC IMAGE REJECTED", "RANDOM MOVEMENT REJECTED", "UNRECOGNIZED"]:
                    st.error(f"⚠️ **Rejection Exception Flagged:** {predicted_word}")
                    if st.button("🔁 Try Again", key="try_again_file_reject", use_container_width=True):
                        st.session_state.learn_result = None
                        st.rerun()
                else:
                    st.markdown("### 📊 File Evaluation Metrics")
                    st.markdown("### ❌")
                    st.markdown(f"Syntax Variation: Recognized as '{predicted_word}' instead of target word context.")
                    
                    if st.button("🔁 Try Again", key="try_again_file_wrong", use_container_width=True):
                        st.session_state.learn_result = None
                        st.rerun()
        else:
            st.info("Diagnostic reporting graphs and accuracy parameters mount onto this frame space dynamically.")

# --- TAB 3: COMMUNITY CHAT ---
elif st.session_state.page == "Community":
    st.markdown('<div class="main-header"><h1>EchoSign Community Chat</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        user_name = st.text_input("Enter your Name:", placeholder="Tumuheki or Nakitende...")
        chat_msg = st.text_area("Your Message:", placeholder="Type here to chat with other signers...")
        if st.button("🚀 Post Message", type="primary"):
            if chat_msg.strip():
                save_community_message(user_name, chat_msg)
                st.rerun()

    st.divider()
    st.subheader("Live Feed")
    
    shared_posts = load_community_messages()
    
    for post in reversed(shared_posts):
        st.markdown(f"""
            <div class="chat-bubble">
                <small style="color:#FF8C00; font-weight:bold;">@{post['user']}</small> 
                <small style="float:right; color:#888;">{post['time']}</small>
                <p style="margin-top:5px;">{post['msg']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- 6. DYNAMIC FOOTER ---
st.markdown(f"""
    <div class="dynamic-footer">
        <p class="footer-name">© 2026 EchoSign Project</p>
        <p>Developed by: <b>Tumuheki Collins & Nakitende Christine</b></p>
        <p>Contacts: <b>0730482053</b> & <b>0763067484</b></p>
        <p><b>Kabale University</b> | Computer Science Final Year Project</p>
    </div>
    """, unsafe_allow_html=True)
