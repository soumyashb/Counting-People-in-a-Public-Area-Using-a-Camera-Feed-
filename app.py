import streamlit as st
import cv2
import time
import sqlite3
import pandas as pd
import tempfile
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(
    page_title="Crowd Counting System",
    layout="wide"
)

st.markdown("""
<style>

/* ---- BASE APP ---- */
.stApp {
    background: #02040a;
    overflow: hidden;
}

/* ---- NEON AURORA BLOBS ---- */
.aurora {
    pointer-events: none;  /* <<< added */
    position: fixed;
    inset: -50%;
    background:
        radial-gradient(45% 35% at 20% 30%, rgba(0,245,255,0.95), transparent 60%),
        radial-gradient(40% 30% at 80% 40%, rgba(255,78,205,0.9), transparent 60%),
        radial-gradient(35% 25% at 50% 80%, rgba(140,87,255,0.9), transparent 60%);
    filter: blur(110px) saturate(210%);
    animation: auroraFlow 9s cubic-bezier(0.4,0.0,0.2,1) infinite alternate;
    z-index: 0;
}

/* second aurora layer for depth */
.aurora.layer2 {
    pointer-events: none;  /* <<< added */
    animation-duration: 15s;
    opacity: 0.75;
    transform: rotate(180deg);
}

/* ---- BLACK HOLE VORTEX (NO CENTER) ---- */
.vortex {
    pointer-events: none;  /* <<< added */
    position: fixed;
    inset: -50%;
    background: conic-gradient(
        from 0deg,
        rgba(0,245,255,0.45),
        rgba(140,87,255,0.45),
        rgba(255,78,205,0.45),
        rgba(0,245,255,0.45)
    );
    filter: blur(140px) saturate(240%);
    animation: vortexSpin 12s linear infinite;
    z-index: 1;
}

/* ---- AURORA FLOW ---- */
@keyframes auroraFlow {
    0%   { transform: translate(-18%, -12%) scale(1); }
    50%  { transform: translate(0%, 12%) scale(1.3); }
    100% { transform: translate(22%, -18%) scale(1.45); }
}

@keyframes vortexSpin {
    from { transform: rotate(0deg) scale(1.15); }
    to   { transform: rotate(360deg) scale(1.4); }
}

/* ---- KEEP CONTENT ABOVE BACKGROUND ---- */
section[data-testid="stMain"] {
    position: relative;
    z-index: 5;
    color: white;
}

</style>

<!-- BACKGROUND LAYERS -->
<div class="aurora"></div>
<div class="aurora layer2"></div>
<div class="vortex"></div>
""", unsafe_allow_html=True)



st.title("Crowd Counting System")

def db():
    return sqlite3.connect("crowd_counts.db")

def authenticate(username, password):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (username, password)
    )
    user = cur.fetchone()
    conn.close()
    return user

def save_count(count):
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT INTO crowd_counts (count) VALUES (?)", (count,))
    conn.commit()
    conn.close()

def load_counts():
    conn = db()
    df = pd.read_sql_query(
        "SELECT timestamp, count FROM crowd_counts ORDER BY id DESC",
        conn
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")

def get_threshold():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT threshold FROM alerts WHERE id=1")
    t = cur.fetchone()[0]
    conn.close()
    return t

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Admin Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate(u, p)
        if user:
            st.session_state.logged_in = True
            st.session_state.role = user[0]
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

page = st.sidebar.radio("*", ["Live Dashboard", "Admin Panel"])

@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt"), DeepSort(max_age=30)

model, tracker = load_models()

if page == "Live Dashboard":
    st.subheader("Live Crowd Counting")
    source = st.sidebar.radio("Video Source", ["Video File", "Webcam"])

    if source == "Video File":
        uploaded = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = None
    else:
        cap = cv2.VideoCapture(0)

    col1, col2 = st.columns([2, 1])
    video_box = col1.empty()
    metric_box = col2.empty()
    chart_box = col2.empty()

    last_save = 0

    if cap and cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            detections = []

            for box in results.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2-x1, y2-y1], 0.9, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)
            count = len([t for t in tracks if t.is_confirmed()])

            for t in tracks:
                if t.is_confirmed():
                    l, t_, r, b = map(int, t.to_ltrb())
                    cv2.rectangle(frame, (l, t_), (r, b), (0,255,0), 2)

            if time.time() - last_save >= 1:
                save_count(count)
                last_save = time.time()

            threshold = get_threshold()
            if count > threshold:
                st.error(f"ALERT! Crowd limit exceeded ({count} > {threshold})")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_box.image(frame, use_container_width=True)

            df = load_counts()
            if not df.empty:
                metric_box.metric("Current People", count)
                chart_box.line_chart(df.set_index("timestamp")["count"])
    else:
        st.info("Waiting for video input...")

else:
    st.subheader("Crowd Data")
    df = load_counts()
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "crowd_data.csv",
        "text/csv"
    )

    if st.session_state.role == "admin":
        st.subheader("User Management")
        nu = st.text_input("New Username")
        np = st.text_input("New Password")
        nr = st.selectbox("Role", ["admin", "viewer"])

        if st.button("Add User"):
            conn = db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (nu, np, nr)
            )
            conn.commit()
            conn.close()
            st.success("User added")

        st.subheader("Alert Threshold")
        new_t = st.number_input("Max People Allowed", 1, 500, get_threshold())

        if st.button("Update Threshold"):
            conn = db()
            cur = conn.cursor()
            cur.execute("UPDATE alerts SET threshold=? WHERE id=1", (new_t,))
            conn.commit()
            conn.close()
            st.success("Threshold updated")