# 🏃 Real-Time Sports Performance Telemetry & Action Tracker

A single-page **Streamlit** web dashboard powered by **YOLOv8** that enables coaches, sports analysts, and engineers to upload video files and observe live multi-object tracking with real-time velocity telemetry.

> Upload a video of athletes sprinting, cars racing, or any moving objects — and instantly get bounding-box tracking, speed estimation, motion trails, and a live velocity graph.

---

## ✨ Features

| Feature | Description |
|---|---|
| **YOLOv8 Object Detection & Tracking** | Uses `yolov8n.pt` with ByteTrack for persistent multi-object tracking across frames |
| **Real-Time Velocity Estimation** | Computes instantaneous speed (m/s) for every tracked object using centroid displacement |
| **Live Dual-Panel UI** | Left panel shows the annotated video feed; right panel displays a live-updating Plotly velocity chart |
| **Motion Trails** | Colour-coded polyline trails that fade with recency, visualising each object's movement path |
| **Interactive Calibration** | Adjustable meter-per-pixel slider for accurate real-world speed mapping |
| **Demo Video Included** | One-click demo mode — no upload required to start exploring |
| **Background Processing** | YOLO inference runs in a background thread, keeping the Streamlit UI responsive |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | [Streamlit](https://streamlit.io/) ≥ 1.38.0 |
| Object Detection | [Ultralytics YOLOv8](https://docs.ultralytics.com/) ≥ 8.2.0 |
| Computer Vision | [OpenCV](https://opencv.org/) ≥ 4.10.0 |
| Numerical Computing | [NumPy](https://numpy.org/) ≥ 2.0.0 |
| Data Visualization | [Plotly](https://plotly.com/python/) ≥ 5.24.0 |
| Deep Learning Backend | [PyTorch](https://pytorch.org/) (installed as a dependency of Ultralytics) |

---

## 📋 Prerequisites

- **Python 3.10+** installed on your system
- **pip** package manager
- **Git** (optional, for cloning the repository)
- No GPU required — runs on CPU (GPU will be used automatically if available via PyTorch/CUDA)

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/RahullShettyy/Real-Time-Sports-Performance-Telemetry-Action-Tracker.git
cd Real-Time-Sports-Performance-Telemetry-Action-Tracker
```

Or download and extract the project ZIP manually.

### 2. (Recommended) Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages: Streamlit, Ultralytics, OpenCV, NumPy, and Plotly (along with their transitive dependencies like PyTorch).

### 4. Run the Application

```bash
python -m streamlit run app.py
```

The app will start and display:

```
Local URL:   http://localhost:8501
Network URL: http://<your-ip>:8501
```

Open **http://localhost:8501** in your browser.

> **Note:** On first run, the YOLOv8n weights (`yolov8n.pt`, ~6 MB) are already included in the project. If missing, they will be auto-downloaded from the Ultralytics hub.

---

## 📖 Usage Guide

### Quick Start with Demo Video

1. Launch the app with `python -m streamlit run app.py`
2. In the **sidebar**, click **▶ Use Demo Video**
3. Watch as the tracker annotates objects with bounding boxes, IDs, speed labels, and motion trails
4. View the live velocity graph updating in real time on the right panel

### Upload Your Own Video

1. In the **sidebar**, under **📂 Upload Video**, drag and drop or browse for a video file
2. Supported formats: `.mp4`, `.avi`, `.mov`
3. Processing begins automatically after upload
4. Use the **⏹ Stop Processing** button to halt at any time

### Configurable Settings

Adjust these parameters in the sidebar before or during processing:

| Setting | Default | Range | Description |
|---|---|---|---|
| **Meter-per-Pixel** | `0.01` | 0.001 – 0.05 | Scale factor to convert pixel displacement to real-world metres. Adjust based on your camera setup and scene. |
| **Process every N-th frame** | `3` | 1 – 10 | Higher values = faster processing but lower temporal resolution. Use `1` for maximum accuracy. |
| **Frame width (px)** | `480` | 320 – 1280 | Resize width for inference. Lower = faster inference speed. |
| **Max frames** | `200` | 0 – 2000 | Limit the number of frames to process. Set to `0` to process the entire video. |

---

## 📊 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🏃 Sports Performance Telemetry                                │
│  YOLOv8 multi-object tracking · real-time velocity · trails     │
├───────────┬───────────┬───────────┬───────────────────────────────┤
│  Frames   │  Active   │   Peak    │   Avg Speed                   │
│ Processed │  Tracks   │   Speed   │                               │
├───────────┴───────────┴───────────┴───────────────────────────────┤
│                          │                                        │
│   📹 Live Tracking Feed  │  📈 Velocity Telemetry (Plotly chart)  │
│   (Annotated video       │  (Multi-line graph: speed vs time      │
│    with bounding boxes,  │   per tracked object)                  │
│    trails & speed)       │                                        │
│                          │                                        │
├──────────────────────────┴────────────────────────────────────────┤
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  Processing… 65.0%                     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Real-Time-Sports-Performance-Telemetry-Action-Tracker/
│
├── app.py                  # Main Streamlit application (single-file architecture)
├── requirements.txt        # Python dependencies
├── yolov8n.pt              # Pre-trained YOLOv8 Nano weights (~6 MB)
├── README.md               # This file
├── PRD.md                  # Product Requirements Document
├── TRD.md                  # Technical Requirements Document
└── *.mp4                   # Demo video file
```

---

## 🏗️ Architecture Overview

The application follows a **single-process, background-thread** architecture:

```
┌──────────────────────────────────────────────┐
│              Streamlit Main Thread            │
│  (UI rendering, session state, user input)    │
│                                               │
│   st.rerun() loop polls result_queue          │
│   every 300ms for new annotated frames        │
└──────────────┬───────────────────────────────┘
               │ Launches
               ▼
┌──────────────────────────────────────────────┐
│           Background Worker Thread            │
│                                               │
│  Video Decode (OpenCV)                        │
│       ↓                                       │
│  Frame Resize & Skip                          │
│       ↓                                       │
│  YOLOv8 Inference + ByteTrack (persist=True)  │
│       ↓                                       │
│  Centroid → Velocity → Smoothing              │
│       ↓                                       │
│  Annotate Frame (boxes, trails, labels)       │
│       ↓                                       │
│  Encode JPEG → Push to result_queue           │
└──────────────────────────────────────────────┘
```

**Key design decisions:**
- **`persist=True`** on `model.track()` keeps ByteTrack's Kalman filter state across frames, maintaining consistent Track IDs
- **Background thread** prevents YOLO inference from blocking the Streamlit WebSocket and causing UI timeouts
- **`@st.cache_resource`** ensures the YOLO model is loaded once per server lifetime with a warm-up pass to eliminate first-frame latency
- **Rolling velocity smoothing** (window of 5) reduces noise from frame-to-frame jitter

---

## ⚠️ Troubleshooting

| Issue | Solution |
|---|---|
| `streamlit` command not found | Use `python -m streamlit run app.py` instead |
| `npm start` fails | This is a **Python** project — use `python -m streamlit run app.py` |
| Slow inference | Reduce **Frame width** and increase **Process every N-th frame** in the sidebar settings |
| Poor speed accuracy | Calibrate the **Meter-per-Pixel** slider based on known real-world distances in your video |
| Video won't open | Ensure the video codec is supported by OpenCV (H.264 recommended). Try converting with FFmpeg. |
| Port 8501 already in use | Run with a custom port: `python -m streamlit run app.py --server.port 8502` |
| Out of memory | Reduce **Frame width** to 320px and set a lower **Max frames** limit |

---

## 📄 License

This project is open source. See the repository for license details.

---

<p align="center">
  Built with ❤️ using Streamlit + YOLOv8
</p>
