'''
Streamlit app for real-time sports performance telemetry & action tracking.
Architecture: YOLO inference runs in a background threading.Thread so Streamlit
UI stays responsive (no WebSocket timeout). Frames are saved to a temp JPEG
file on disk and displayed via st.image(path) – no media file store involved.
'''

import base64
import math
import pathlib
import queue
import tempfile
import threading
import time
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_METER_PER_PIXEL: float = 0.01
TRACK_COLOURS = [
    (255,   0,   0),   # Red
    (  0, 200,   0),   # Green
    (  0,   0, 255),   # Blue
    (255, 200,   0),   # Yellow
    (255,   0, 200),   # Pink
    (  0, 200, 255),   # Cyan
    (255, 128,   0),   # Orange
]
MAX_TRAIL_LENGTH: int = 40
VELOCITY_SMOOTH_WINDOW: int = 5
MODEL_WEIGHTS = "yolov8n.pt"
DEMO_VIDEO_PATH = pathlib.Path(__file__).parent / "WhatsApp Video 2026-06-07 at 4.29.53 PM.mp4"

# Shared state between background thread and Streamlit
_worker_thread: threading.Thread | None = None
_result_queue: queue.Queue = queue.Queue(maxsize=5)   # (jpeg_bytes, telemetry_dict)
_stop_event = threading.Event()

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading YOLOv8n weights…")
def load_yolo_model(weights: str = MODEL_WEIGHTS) -> YOLO:
    model = YOLO(weights)
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pure Math Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def compute_velocity(prev, curr, delta_t, m_per_px):
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    return math.sqrt(dx * dx + dy * dy) * m_per_px / delta_t


def smooth(series, window=VELOCITY_SMOOTH_WINDOW):
    tail = series[-window:] if series else [0.0]
    return sum(tail) / len(tail)


# ─────────────────────────────────────────────────────────────────────────────
# Frame Annotation
# ─────────────────────────────────────────────────────────────────────────────
def annotate_frame(frame, bboxes, ids, speeds, track_history):
    for bbox, tid, spd in zip(bboxes, ids, speeds):
        color = TRACK_COLOURS[int(tid) % len(TRACK_COLOURS)]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        cx, cy = compute_centroid(x1, y1, x2, y2)
        cv2.circle(frame, (int(cx), int(cy)), 3, (255, 255, 255), -1)
        label = f"ID:{tid}  {spd:.1f}m/s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        chip_y = y1 - th - 6 if y1 - th - 6 > 0 else y1 + th + 6
        cv2.rectangle(frame, (x1, chip_y - 1), (x1 + tw + 8, chip_y + th + 4), color, -1)
        cv2.putText(frame, label, (x1 + 4, chip_y + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        trail = track_history[int(tid)][-MAX_TRAIL_LENGTH:]
        for i in range(1, len(trail)):
            pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            thickness = max(1, int(3 * i / len(trail)))
            cv2.line(frame, pt1, pt2, color, thickness)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Background Worker Thread
# ─────────────────────────────────────────────────────────────────────────────
def _worker(video_path: str, cfg: dict, model: YOLO, stop_evt: threading.Event,
            out_q: queue.Queue):
    """Run YOLO tracking in a background thread; push results to out_q."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        out_q.put({"error": f"Cannot open video: {video_path}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delta_t = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_skip = cfg.get("frame_skip", 2)
    resize_w  = cfg.get("resize_width", 640)
    max_f     = cfg.get("max_frames", 0)
    m_per_px  = cfg.get("meter_per_pixel", DEFAULT_METER_PER_PIXEL)

    track_history   = defaultdict(list)
    velocity_history = defaultdict(list)
    timestamps       = []
    frame_count      = 0
    processed        = 0

    while not stop_evt.is_set():
        if max_f and processed >= max_f:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        processed += 1

        # Resize for speed
        if resize_w and frame.shape[1] != resize_w:
            h = int(frame.shape[0] * resize_w / frame.shape[1])
            frame = cv2.resize(frame, (resize_w, h))

        results = model.track(source=frame, persist=True, verbose=False)
        boxes = results[0].boxes
        annotated = frame.copy()

        if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
            bboxes = boxes.xyxy.cpu().numpy()
            ids    = boxes.id.cpu().numpy().astype(int)
            speeds = []
            for bbox, tid in zip(bboxes, ids):
                cx, cy = compute_centroid(*bbox)
                hist = track_history[tid]
                v_raw = compute_velocity(hist[-1], (cx, cy), delta_t, m_per_px) if hist else 0.0
                velocity_history[tid].append(v_raw)
                speeds.append(smooth(velocity_history[tid]))
                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > MAX_TRAIL_LENGTH:
                    track_history[tid] = track_history[tid][-MAX_TRAIL_LENGTH:]
                if len(velocity_history[tid]) > MAX_TRAIL_LENGTH:
                    velocity_history[tid] = velocity_history[tid][-MAX_TRAIL_LENGTH:]
            timestamps.append(processed * delta_t * frame_skip)
            annotated = annotate_frame(annotated, bboxes, ids, speeds, track_history)

        # Encode frame as JPEG bytes
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = buf.tobytes() if ok else None

        # Build velocity_data snapshot (copy so it's thread-safe)
        vel_data = {tid: list(series) for tid, series in velocity_history.items()}
        ts_copy  = list(timestamps)
        progress = min(processed / max(max_f or total_frames, 1), 1.0)

        payload = {
            "jpeg": jpeg_bytes,
            "velocity_history": vel_data,
            "timestamps": ts_copy,
            "frame_count": processed,
            "total_frames": total_frames,
            "progress": progress,
            "done": False,
        }
        # Drop old frames if queue is full (keep UI fresh)
        if out_q.full():
            try:
                out_q.get_nowait()
            except queue.Empty:
                pass
        out_q.put(payload)

    cap.release()
    # Signal done – carry the final real data so the UI keeps displaying results
    final_vel  = {tid: list(series) for tid, series in velocity_history.items()}
    final_ts   = list(timestamps)
    # Compute all-time peak speed across every track and every frame
    all_speeds = [v for series in velocity_history.values() for v in series]
    peak_ever  = max(all_speeds, default=0.0)
    try:
        out_q.put({
            "jpeg": jpeg_bytes,       # keep last annotated frame visible
            "done": True,
            "frame_count": processed,
            "total_frames": total_frames,
            "velocity_history": final_vel,
            "timestamps": final_ts,
            "peak_speed_ever": peak_ever,
            "progress": 1.0,
        })
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Velocity Chart
# ─────────────────────────────────────────────────────────────────────────────
def build_velocity_chart(velocity_history: dict, timestamps: list):
    fig = go.Figure()
    for tid, series in velocity_history.items():
        pad_len = len(timestamps) - len(series)
        padded = ([None] * pad_len) + series if pad_len > 0 else series
        fig.add_trace(go.Scatter(
            x=timestamps, y=padded,
            mode="lines+markers",
            name=f"Track {tid}",
            connectgaps=False,
            line=dict(width=2),
        ))
    fig.update_layout(
        title="Velocity (m/s) over Time",
        xaxis_title="Time (s)",
        yaxis_title="Velocity (m/s)",
        legend_title="Tracks",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,30,0.6)",
        font=dict(color="#e0e0e0"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page Config & Styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Telemetry & Action Tracker",
    page_icon="🏃",
    layout="wide",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%); }
.main-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.25rem;
}
.subtitle { color: #8899aa; font-size: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #00d2ff; }
.metric-lbl { font-size: 0.75rem; color: #8899aa; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stSidebar"] { background: rgba(10,10,26,0.9) !important; border-right: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────────────────────
def init_ss():
    defaults = {
        "running": False,
        "last_payload": None,
        "use_demo": False,
        "worker_thread": None,
        "stop_event": None,
        "result_queue": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_ss()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏃 Sports Telemetry")

    if DEMO_VIDEO_PATH.exists():
        st.markdown("### 🎬 Quick Start")
        if st.button("▶ Use Demo Video", use_container_width=True, type="primary"):
            st.session_state["use_demo"] = True
        st.markdown("---")

    st.markdown("### 📂 Upload Video")
    uploaded = st.file_uploader("MP4 / AVI / MOV", type=["mp4", "avi", "mov"])

    st.markdown("### ⚙️ Settings")
    meter_per_pixel = st.slider("Meter-per-Pixel", 0.001, 0.05, DEFAULT_METER_PER_PIXEL, 0.001,
                                help="Scale: metres per pixel")
    frame_skip = st.slider("Process every N-th frame", 1, 10, 3, 1,
                           help="Higher = faster but lower resolution tracking")
    resize_width = st.slider("Frame width (px)", 320, 1280, 480, 64,
                             help="Lower = faster inference")
    max_frames = st.slider("Max frames (0 = all)", 0, 2000, 200, 100)

    if st.session_state.get("running"):
        if st.button("⏹ Stop Processing", use_container_width=True, type="secondary"):
            if st.session_state.get("stop_event"):
                st.session_state["stop_event"].set()
            st.session_state["running"] = False

cfg = {
    "meter_per_pixel": meter_per_pixel,
    "frame_skip": frame_skip,
    "resize_width": resize_width,
    "max_frames": max_frames,
}


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏃 Sports Performance Telemetry</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8 multi-object tracking · real-time velocity · motion trails</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Determine video source & launch worker
# ─────────────────────────────────────────────────────────────────────────────
use_demo = st.session_state.pop("use_demo", False)
video_source = None

if use_demo and DEMO_VIDEO_PATH.exists():
    video_source = str(DEMO_VIDEO_PATH)
elif uploaded is not None:
    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush()
    video_source = tmp.name

if video_source and not st.session_state["running"]:
    # Stop any previous worker
    if st.session_state.get("stop_event"):
        st.session_state["stop_event"].set()

    model = load_yolo_model()
    stop_evt = threading.Event()
    result_q  = queue.Queue(maxsize=5)

    t = threading.Thread(
        target=_worker,
        args=(video_source, cfg, model, stop_evt, result_q),
        daemon=True,
    )
    t.start()

    st.session_state["running"] = True
    st.session_state["last_payload"] = None
    st.session_state["worker_thread"] = t
    st.session_state["stop_event"] = stop_evt
    st.session_state["result_queue"] = result_q


# ─────────────────────────────────────────────────────────────────────────────
# Main display area
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state["running"] and st.session_state["last_payload"] is None:
    # Landing state
    col1, col2 = st.columns(2)
    with col1:
        st.info("📂 Upload a video in the sidebar to start tracking.")
    with col2:
        if DEMO_VIDEO_PATH.exists():
            st.success("🎬 Or click **▶ Use Demo Video** for a quick demo!")
else:
    # Poll the queue for newest frame
    result_q = st.session_state.get("result_queue")
    payload = st.session_state.get("last_payload")

    if result_q is not None:
        newest = None
        while True:
            try:
                newest = result_q.get_nowait()
            except queue.Empty:
                break
        if newest is not None:
            if newest.get("error"):
                st.error(newest["error"])
                st.session_state["running"] = False
            else:
                payload = newest
                st.session_state["last_payload"] = payload
                if newest.get("done"):
                    st.session_state["running"] = False

    if payload:
        # ── Metrics row ──────────────────────────────────────────────────────
        frames_done = payload.get("frame_count", 0)
        total_f     = payload.get("total_frames", 0)
        vel_hist    = payload.get("velocity_history", {})
        n_tracks    = len(vel_hist)
        # Peak speed: use stored all-time peak if present, else compute from all samples
        if "peak_speed_ever" in payload:
            max_speed = payload["peak_speed_ever"]
        else:
            all_vals  = [v for series in vel_hist.values() for v in series]
            max_speed = max(all_vals, default=0.0)
        # Avg speed: mean of each track's most recent smoothed value
        current_speeds = [series[-1] for series in vel_hist.values() if series]
        avg_speed = sum(current_speeds) / len(current_speeds) if current_speeds else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{frames_done}</div>'
                        f'<div class="metric-lbl">Frames Processed</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{n_tracks}</div>'
                        f'<div class="metric-lbl">Active Tracks</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{max_speed:.2f} m/s</div>'
                        f'<div class="metric-lbl">Peak Speed</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_speed:.2f} m/s</div>'
                        f'<div class="metric-lbl">Avg Speed</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Video frame + chart ──────────────────────────────────────────────
        vid_col, chart_col = st.columns([3, 2])
        with vid_col:
            st.markdown("**📹 Live Tracking Feed**")
            jpeg = payload.get("jpeg")
            if jpeg:
                b64 = base64.b64encode(jpeg).decode("ascii")
                st.markdown(
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="width:100%;border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,0.5);" />',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Waiting for first frame…")

        with chart_col:
            st.markdown("**📈 Velocity Telemetry**")
            vel = payload.get("velocity_history", {})
            ts  = payload.get("timestamps", [])
            if vel and ts:
                st.plotly_chart(build_velocity_chart(vel, ts), use_container_width=True)
            else:
                st.info("Velocity data will appear once tracks are detected.")

        # ── Progress bar ─────────────────────────────────────────────────────
        prog = payload.get("progress", 0.0)
        if st.session_state["running"]:
            st.progress(prog, text=f"Processing… {prog*100:.1f}%")
        else:
            st.success(f"✅ Processing complete – {frames_done} frames analysed.")

    # Auto-refresh while the worker is still running
    if st.session_state["running"]:
        time.sleep(0.3)
        st.rerun()
