Three files delivered. Here's the complete engineering breakdown.

Technical Requirements Document — Summary
Architecture at a Glance
The application is a single-process, single-file Streamlit app structured in 12 discrete sections, each with a clear engineering responsibility. The hot path (frame decode → inference → telemetry → render) runs inside a while loop within a single Streamlit rerun, using st.empty() placeholders to stream UI updates without re-executing the full script.

Code Breakdown by Section
Section 1 — Global Constants
All compile-time values (DEFAULT_METER_PER_PIXEL, TRACK_COLOURS, MAX_TRAIL_LENGTH, VELOCITY_SMOOTH_WINDOW) are defined at module scope as typed constants. The TRACK_COLOURS list maps Track IDs to stable BGR tuples using modulo indexing — ensuring colour-to-ID mapping never varies across frames.
Section 2 — @st.cache_resource Model Loading
python@st.cache_resource(show_spinner="Loading YOLOv8n weights…")
def load_yolo_model(weights: str = MODEL_WEIGHTS) -> YOLO:
    model = YOLO(weights)
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # warm-up
    return model
The decorator pins the YOLO instance to server-process memory — it is constructed exactly once per server lifetime. The warm-up inference pass forces PyTorch's JIT compiler to build the compute graph before any user video arrives, eliminating the ~500 ms first-frame latency spike.
Section 3 — Session State Initialisation
pythondef init_session_state():
    defaults = {
        "track_history":    defaultdict(list),  # id → [(x,y), ...]
        "velocity_history": defaultdict(list),  # id → [v_mps, ...]
        "timestamps":       [],
        "frame_count":      0,
        "processing":       False,
        ...
    }
    for key, value in defaults.items():
        if key not in st.session_state:      # guard: don't clobber on rerun
            st.session_state[key] = value
defaultdict(list) is used for track_history and velocity_history so that the first access for any new Track ID automatically creates an empty list — no setdefault() guards needed in the hot loop.
Section 4 — Mathematical Core
Three pure functions, zero side effects:
python# 1. Centroid from YOLOv8 xyxy box
def compute_centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2.0,  (y1 + y2) / 2.0

# 2. Instantaneous velocity
def compute_velocity(prev, curr, delta_t, m_per_px):
    d_px = math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
    return (d_px * m_per_px) / delta_t          # m/s

# 3. Causal rolling mean (smoothing)
def smooth_velocity(series, window=5):
    return sum(series[-window:]) / len(series[-window:])
The velocity formula chain is: pixel displacement → metric displacement → divide by Δt = 1/fps. The calibration slider directly controls m_per_px and takes effect on the next computed frame because st.session_state["meter_per_pixel"] is read inside the loop on each iteration.
Section 5 — Frame Annotation
annotate_frame() draws four layered elements per track in a single function call: bounding box, filled label chip (ID + speed), centroid dot, and a motion trail polyline where stroke width scales with recency (thickness = max(1, int(3 * (i / len(trail))))). All drawing is in-place on the BGR frame ndarray.
Section 6 — build_velocity_chart() (Decoupled from Hot Loop)
The chart builder is a pure function that reads from session state and returns a go.Figure. Zero-padding aligns shorter velocity series (tracks that appeared mid-video) to the shared timestamp x-axis:
pythonpad_len = n_timestamps - len(velocities)
padded  = ([None] * pad_len) + list(velocities)
connectgaps=False preserves None gaps as visual breaks when a track is temporarily lost.
Section 8 — Sidebar (render_sidebar())
Returns a plain dict of control values rather than writing side effects. This keeps the sidebar composable and testable. The meter_per_pixel slider is mirrored into st.session_state immediately to allow live recalibration during an active run.
Section 10 — Core Processing Loop (run_tracking_loop())
This is the architectural centrepiece. The critical design decisions:

persist=True on model.track() — Without this, ByteTrack discards its Kalman filter state between calls and resets Track IDs every frame.
st.empty() placeholders — Created once before the while loop; .image() and .plotly_chart() called inside the loop replace (not append) content on each iteration, producing the streaming effect.
channels="BGR" — OpenCV decodes as BGR; Streamlit's st.image defaults to RGB. Omitting this flag produces colour-inverted frames.
cap.release() in finally — Ensures the video file handle is released even if an exception occurs mid-loop.

pythonwhile st.session_state["processing"]:
    ret, frame = cap.read()
    if not ret: break                          # end of video

    results = model.track(source=frame, persist=True, ...)
    for box, track_id in zip(boxes.xyxy, boxes.id):
        cx, cy = compute_centroid(*box.tolist())
        v_raw  = compute_velocity(history[-1], (cx,cy), delta_t, m_per_px)
        st.session_state["velocity_history"][track_id].append(v_raw)
        frame  = annotate_frame(frame, ...)

    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
    chart_placeholder.plotly_chart(build_velocity_chart(...), use_container_width=True)

Deployment in 2 Commands
bashpip install -r requirements.txt
streamlit run app.py
yolov8n.pt (~6 MB) downloads automatically from the Ultralytics hub on first run. No Docker, no GPU, no API keys required. Deployable to Streamlit Community Cloud with a single git push.