import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc  # <-- ADD THIS NEW LINE

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

    /* Join Community Section in Body */
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

    /* DYNAMIC FOOTER - Locked to Sidebar width and moves with scroll */
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
import mediapipe as mp
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

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame_counter += 1
        # Skip every other frame to save massive amounts of memory
        if frame_counter % 2 == 0: 
            continue
            
        # Shrink the frame resolution to reduce RAM load
        frame = cv2.resize(frame, (320, 240))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        frame_coords = np.zeros(126) 
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i < 2: 
                    coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                    frame_coords[i*63 : i*63 + 63] = coords
        sequence.append(frame_coords)
        
    cap.release()
    gc.collect()  # Force the server to dump the video from memory
    return sequence

def process_video(video_path, filename):
    raw_sequence = extract_landmarks(video_path)
    
    # 1. EMPTY HAND CHECK
    # If the sequence is totally empty or just zeros, MediaPipe saw no hands
    if not raw_sequence or np.all(raw_sequence == 0):
        return {"File": filename, "Prediction": "NO HANDS DETECTED", "Confidence": 0.0}

    best_conf, best_idx = 0, 0
    
    if len(raw_sequence) < 30:
        input_data = pad_sequences([raw_sequence], maxlen=30, padding='post', dtype='float32')
        res = model.predict(input_data, verbose=0)[0]
        best_idx = np.argmax(res)
        best_conf = res[best_idx]
    else:
        for start in range(0, len(raw_sequence) - 30, 8):
            window = raw_sequence[start : start + 30]
            input_data = np.expand_dims(window, axis=0)
            res = model.predict(input_data, verbose=0)[0]
            if res[np.argmax(res)] > best_conf:
                best_idx = np.argmax(res)
                best_conf = res[best_idx]
                
    # 2. THE CONFIDENCE THRESHOLD
    # If the AI is less than 70% sure, reject the translation
    CONFIDENCE_LIMIT = 0.70  
    
    if best_conf < CONFIDENCE_LIMIT:
        final_prediction = "UNRECOGNIZED"
    else:
        final_prediction = labels[best_idx]

    return {"File": filename, "Prediction": final_prediction, "Confidence": round(best_conf*100, 1)}

# --- 3. SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = "Translator"
if 'results_data' not in st.session_state: st.session_state.results_data = []
if 'community_posts' not in st.session_state: 
    st.session_state.community_posts = [{"user": "Admin", "msg": "Welcome to EchoSign Community Chat!", "time": "10:00 AM"}]

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🤟 EchoSign Menu")
    if st.button("🏠 Translator Studio", use_container_width=True):
        st.session_state.page = "Translator"
        st.rerun()
    if st.button("👥 Community Chat", use_container_width=True):
        st.session_state.page = "Community"
        st.rerun()
    
    st.divider()
    st.markdown("### Dictionary")
    st.write(", ".join(labels[:12]) + "...")

# --- 5. MAIN PAGE CONTENT ---
if st.session_state.page == "Translator":
    st.markdown('<div class="main-header"><h1>EchoSign: ASL Sentence Studio</h1></div>', unsafe_allow_html=True)
    
    col_l, col_c, col_r = st.columns([1, 1.4, 1.1], gap="large")

    with col_l:
        st.subheader("📤 1. Upload Sequence")
        files = st.file_uploader("Upload in sentence order", type=["mp4", "mov"], accept_multiple_files=True)
        if files and st.button("🚀 CONSTRUCT SENTENCE", type="primary", use_container_width=True):
            st.session_state.results_data = []
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t:
                    t.write(f.read())
                    temp_path = t.name
                
                # Process the video
                res = process_video(temp_path, f.name)
                st.session_state.results_data.append(res)
                
                # Ruthlessly clean up memory
                os.remove(temp_path)
                gc.collect()

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
            st.info("Upload videos to see results.")

    with col_r:
        st.subheader("📊 3. Session Data")
        if st.session_state.results_data:
            df = pd.DataFrame(st.session_state.results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("Summary table ready.")

    # JOIN COMMUNITY SECTION
    st.markdown("""
        <div class="join-community-box">
            <h3>🌍 Connect with the Community</h3>
            <p>Join other ASL learners and share your EchoSign experience today!</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("💬 Open Community Chat Now", use_container_width=True):
        st.session_state.page = "Community"
        st.rerun()

elif st.session_state.page == "Community":
    st.markdown('<div class="main-header"><h1>EchoSign Community Chat</h1></div>', unsafe_allow_html=True)
    
    # Chat Input Section
    with st.container():
        user_name = st.text_input("Enter your Name:", placeholder="Tumuheki or Nakitende...")
        chat_msg = st.text_area("Your Message:", placeholder="Type here to chat with other signers...")
        if st.button("🚀 Post Message", type="primary"):
            if chat_msg:
                new_post = {
                    "user": user_name if user_name else "Anonymous",
                    "msg": chat_msg,
                    "time": datetime.now().strftime("%I:%M %p")
                }
                st.session_state.community_posts.append(new_post)
                st.rerun()

    st.divider()
    
    # Display Chat Feed
    st.subheader("Live Feed")
    for post in reversed(st.session_state.community_posts):
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
