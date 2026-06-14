Act as an expert Senior Product Manager specializing in AI-driven sports analytics. Write a comprehensive, concise Product Requirement Document (PRD) for a "Real-Time Sports Performance Telemetry & Action Tracker."

The product must be a single-page Streamlit web dashboard that allows coaches, sports analysts, and engineers to upload video files (e.g., athletes sprinting, Formula 1 cars racing) and observe live object tracking along with an instantaneous telemetry line graph mapping target speed over time.

Please format the PRD using clean markdown with the following sections:
1. **Product Overview & Core Value Proposition**: What problem does this solve for performance tracking?
2. **Target User Persona**: Sports analysts and coaches who need visual, data-backed proof of performance metrics without code complexity.
3. **Functional Requirements (FRs)**:
    - User-driven video file uploading (.mp4, .avi, .mov).
    - Synchronized dual-panel UI: Left panel displays the live tracking bounding boxes, right panel displays a real-time updating multi-line graph of velocity.
    - Interactive speed calibration slider (Meter-per-Pixel coefficient mapping).
4. **Non-Functional Requirements (NFRs)**: Latency, responsive layout, minimal dependency foot-print.
5. **UI/UX Wireframe Flow**: A clear markdown text schematic illustrating how components should look when the app initiates versus when the processing loop executes.