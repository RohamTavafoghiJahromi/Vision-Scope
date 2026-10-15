# VisionScope

> Advanced real-time visual AI scanner for live cameras and video files —
> detect, track, and describe objects using state-of-the-art computer vision models.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-orange)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-see%20below-lightgrey)](#license)

VisionScope is a desktop application that watches your webcam or a video file,
finds objects you describe in natural language or by reference image, draws
labels and bounding boxes on top of the video in real time, tracks objects
across frames, and — on demand — uses a vision-language model to describe what
it sees.

Built with **PySide6** for the interface and **PyTorch / Ultralytics /
Hugging Face Transformers** for the vision stack. Everything runs locally on
your GPU by default. No cloud calls, no telemetry, no account required.

---

## Table of contents

- [Features](#features)
- [Hardware requirements](#hardware-requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Model management](#model-management)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Running the tests](#running-the-tests)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Privacy](#privacy)
- [License and model attributions](#license-and-model-attributions)
- [Author](#author)

---

## Features

- **Live camera and video-file input** through a single unified pipeline.
- **Real-time object detection** using YOLO11 (nano / small / medium / large / xlarge).
- **Open-vocabulary text search** — type *"red backpack"*, *"cardboard box"*,
  *"silver wrench"* and Grounding DINO will find it, even though those aren't
  COCO classes.
- **Reference-image search** — upload a photo of an object and OWLv2 will find
  visually similar regions in the feed.
- **Object tracking** with stable IDs across frames (ByteTrack / BoT-SORT).
- **Optional segmentation masks** (MobileSAM) that outline an object's actual
  pixels instead of drawing a rectangle.
- **Vision-language descriptions** (BLIP) for any detected object or the whole
  frame, on demand.
- **Annotated screenshots and video export** — save the processed stream with
  boxes, labels, and tracking IDs baked in.
- **Playback controls** for video files: play, pause, seek, speed 0.25×–8× and
  unlimited.
- **Model management page** — everything downloads explicitly, nothing runs
  silently, every entry lists its license and size.
- **All inference runs off the UI thread.** The interface never freezes while
  a model is loading or a frame is being processed.
- **Automatic GPU detection** with graceful CPU fallback.
- **Dark, professional UI** built with PySide6.

---

## Hardware requirements

### Minimum
- Windows 10/11, Linux, or macOS
- Python 3.10, 3.11, or 3.12
- 8 GB system RAM
- CPU-only inference works but expect low frame rates

### Recommended (what VisionScope is designed for)
- Windows 10/11
- Python 3.11
- NVIDIA GPU with **≥ 6 GB VRAM** and driver **≥ 525.60**
- 16 GB system RAM
- ~10 GB free disk space for models

### Tested on
- Windows 11 · Python 3.11 · NVIDIA RTX 3070 (8 GB) · driver 5xx · CUDA runtime 12.1
- Windows 11 · Python 3.11 · CPU-only mode (works, slow)

---

## Installation

### 0. Check your hardware

Open PowerShell and run:

```powershell
nvidia-smi
```

You need **Driver Version ≥ 525.60** and the top-right `CUDA Version` field
should say **12.x**. If `nvidia-smi` doesn't exist, install the latest NVIDIA
driver from
<https://www.nvidia.com/Download/index.aspx>, reboot, and try again.

> **You do not need to install the CUDA Toolkit.** PyTorch ships its own CUDA
> runtime libraries inside the pip wheels. The full Toolkit is only needed if
> you compile CUDA extensions yourself — VisionScope does not.

### 1. Install Python 3.11

Download from <https://www.python.org/downloads/release/python-3119/>.
On the first installer screen, check **"Add python.exe to PATH"** before
clicking *Install Now*.

Verify:

```powershell
py -3.11 --version
# Expected: Python 3.11.x
```

### 2. Clone and create a virtual environment

```powershell
git clone https://github.com/<your-username>/visionscope.git
cd visionscope

py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Your prompt should now start with `(.venv)`.

### 3. Install PyTorch with CUDA

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --timeout 120 --retries 10
```

Verify the GPU is visible to PyTorch:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected:
```
2.5.1+cu121 True NVIDIA GeForce RTX 3070
```

If `cuda.is_available()` is `False`, stop here and read
[Troubleshooting → CUDA not available](#cuda-not-available).

### 4. Install the application dependencies

```powershell
pip install -r requirements.txt
```

> **Important:** VisionScope is pinned to `transformers==4.46.1`. Newer 5.x
> releases call `torch.accelerator.current_accelerator()`, which does not exist
> in torch 2.5.x. Upgrading transformers past 4.46.1 without also upgrading
> torch to 2.6+ will break open-vocabulary and reference-image detection.

### 5. Launch

```powershell
python run.py
```

On first launch, a `config.yaml` file is created from `config.example.yaml`,
and YOLO11-nano (~5 MB) downloads automatically.

The status bar at the bottom of the window will show your detected hardware:

```
Device: NVIDIA GeForce RTX 3070  |  CUDA: yes  |  Recommended: yolo11s.pt
```

### 6. Optional — download the extra models

Open the **Models** dialog from the toolbar and click **Download** next to
whichever ones you need. See [Model management](#model-management) for sizes
and what each one does.

---

## Quick start

### Find every cup on your webcam

1. **Open camera…** in the toolbar → index `0` → **OK**.
2. In the left panel, type `cup` in the text-target box and press **Enter**.
3. On the first frame, Grounding DINO (~700 MB) downloads and loads.
4. Cups now get boxes labeled `cup  0.xx`.

### Find a specific connector from a photo

1. **Open camera…** or **Open video…** as usual.
2. In the left panel, click **Load image…** and pick a clean photo of the
   object.
3. Lower the *match threshold* if you're missing it, raise it if you're
   getting false positives.

### Save an annotated clip

1. Start your source.
2. **Record annotated…** → choose a filename.
3. **Stop recording** in the toolbar when you're done.

Output files land in `outputs/` by default.

---

## Using the app

### Main window layout

```
┌─────────────┬─────────────────────────────────┬──────────────────┐
│  Search     │     Video viewer                │  Tracks          │
│  panel      │     (largest area)              │  Events          │
│             │                                 │  Selected object │
└─────────────┴─────────────────────────────────┴──────────────────┘
         Toolbar:  camera · file · screenshot · record · settings · models
         Status bar: device · FPS · latency · queue · dropped
         Playback bar:  Play | Pause | Stop | seek slider | time | speed
```

### Search panel (left)

- **Text targets** — type any phrase, press Enter. Multiple phrases combine
  into one Grounding DINO call per frame. Uncheck an entry to disable it
  without deleting it.
- **Reference image** — drag a `.png`/`.jpg` onto the dashed area, or click
  **Load image…**. First inference loads OWLv2 on demand.
- **Match threshold** — higher = stricter. This is *visual similarity*, not
  identity. Two similar-looking objects will match.

### Video viewer (center)

Rendered with boxes, labels (`label  #trackid  score`), and optional
segmentation masks. Labels flip below the box when there's no room above, and
font size auto-scales to the frame resolution.

### Right tabs

- **Tracks** — active track IDs with label and confidence.
- **Events** — `appear` / `disappear` lifecycle events, deduplicated.
- **Selected object** — crop preview + *Describe with vision-language model*.

### Toolbar

| Button | What it does |
|---|---|
| **Open camera…** | Prompts for a camera index (0 = built-in webcam). |
| **Open video…** | MP4, AVI, MOV, MKV, WebM. |
| **Screenshot (annotated)** | Saves current frame with overlays. |
| **Screenshot (clean)** | Saves without overlays. |
| **Record annotated…** | Begins writing an MP4 with boxes and labels. |
| **Stop recording** | Finalizes the recording. |
| **Settings** | Device, weights, thresholds, tracking, segmentation, save dir. |
| **Models** | Download and manage model weights. |

### Playback bar (bottom)

For video files: Play, Pause, Stop, seek slider, time indicator, and a
**Speed:** dropdown (0.25×, 0.5×, 1×, 2×, 4×, 8×, Max).

`Max` disables throttling entirely — the pipeline processes the file as fast
as your GPU allows. Useful for scanning a long recording.

### Keyboard shortcuts

| Key | Action |
|---|---|
| **Space** | Pause / resume the pipeline |

---

## Model management

VisionScope downloads **nothing** unless you click a Download button on the
Models page. Every entry shows its size, purpose, and license.

| Model | Size | Purpose | License |
|---|---|---|---|
| YOLO11-nano | ~5 MB | Fast closed-set detection (80 COCO classes) | AGPL-3.0 |
| YOLO11-small | ~19 MB | Same, more accurate, ~3× slower | AGPL-3.0 |
| Grounding DINO tiny | ~700 MB | Open-vocabulary text-prompted detection | Apache-2.0 |
| OWLv2 base | ~1.1 GB | Reference-image guided detection | Apache-2.0 |
| MobileSAM | ~40 MB | Optional segmentation masks | Apache-2.0 |
| BLIP captioning | ~1.8 GB | Detailed image descriptions | BSD-3-Clause |

### Where models are stored

- **Ultralytics weights** (`yolo11n.pt`, `mobile_sam.pt`): in the project root.
- **Hugging Face weights** (Grounding DINO, OWLv2, BLIP):
  `%USERPROFILE%\.cache\huggingface\hub\` on Windows, `~/.cache/huggingface/hub/`
  on Linux/macOS.

### Replacing or removing a model

Change the path or Hugging Face ID in `config.yaml` (or via **Settings**).
To remove a model, delete the corresponding `.pt` file or the
`models--<org>--<name>` folder in the Hugging Face cache.

### ⚠️ Ultralytics licensing note

YOLO11 code and weights are **AGPL-3.0**. If you distribute a closed-source
product built on top of VisionScope, you must either (a) purchase an
Ultralytics Enterprise license, (b) replace the YOLO backend with an
Apache-2.0-licensed detector, or (c) rely solely on Grounding DINO and disable
the YOLO path. See [License and model attributions](#license-and-model-attributions).

---

## Configuration

`config.yaml` is created automatically on first run from
`config.example.yaml`. Edit it directly or use the in-app **Settings** dialog.

```yaml
app:
  theme: dark
  save_dir: "./outputs"
  log_level: INFO
  event_history_max: 500

video:
  camera_index: 0
  target_width: 1280
  target_height: 720
  process_every_n_frames: 2     # higher = faster, less accurate on fast motion
  queue_size: 4
  prefer_smooth_playback: true

models:
  device: auto                  # auto | cuda | cpu
  detector_weights: "yolo11n.pt"  # try yolo11s.pt / yolo11m.pt for accuracy
  open_vocab_model: "IDEA-Research/grounding-dino-tiny"
  reference_model: "google/owlv2-base-patch16-ensemble"
  sam_model: "mobile_sam.pt"
  vlm_model: "Salesforce/blip-image-captioning-large"
  half_precision: true          # FP16 — faster, ~no accuracy loss

thresholds:
  detection_confidence: 0.35
  open_vocab_box_threshold: 0.30
  open_vocab_text_threshold: 0.25
  reference_match_threshold: 0.35

features:
  tracking_enabled: true
  segmentation_enabled: false   # heavy; enable only when you want masks
  motion_trails: false
  trail_length: 30

rendering:
  box_thickness: 2
  label_font_scale: 0.5         # auto-scaled by frame resolution
  show_confidence: true
  show_track_id: true
```

### Tuning tips

- **Higher accuracy** → `detector_weights: "yolo11m.pt"` and/or set
  `imgsz=960` in `visual_ai_scanner/detection.py`.
- **Higher FPS** → `process_every_n_frames: 3` or `4`; keep `imgsz=640`.
- **Fewer false positives** → raise `detection_confidence` to `0.5`–`0.6`.
- **Catch more objects** → lower `detection_confidence` to `0.20`–`0.25`.

---

## Project structure

```
visionscope/
├── run.py                          # entry point
├── config.example.yaml             # sample configuration (no secrets)
├── requirements.txt                # Python dependencies
├── README.md
├── tests/
│   └── test_core.py                # pytest smoke tests
└── visual_ai_scanner/
    ├── __init__.py
    ├── app.py                      # QApplication bootstrap
    ├── config.py                   # typed YAML config
    ├── hardware.py                 # GPU detection + advice
    ├── logging_setup.py
    ├── video.py                    # camera / file sources + bounded queue
    ├── detection.py                # YOLO / Grounding DINO / OWLv2
    ├── tracking.py                 # ByteTrack / BoT-SORT wrapper
    ├── segmentation.py             # MobileSAM wrapper
    ├── vlm.py                      # BLIP vision-language model
    ├── rendering.py                # boxes, labels, masks, trails
    ├── pipeline.py                 # QThread worker: video → detections
    ├── events.py                   # track lifecycle history
    ├── export.py                   # screenshots + annotated video
    ├── models.py                   # model availability + download
    └── ui/
        ├── __init__.py
        ├── main_window.py          # QMainWindow + signal wiring
        ├── video_widget.py         # frame → QPixmap rendering
        ├── panels.py               # search / tracks / events / object
        └── dialogs.py              # settings + model manager
```

The design is deliberately modular: **each backend is swappable**. To replace
YOLO with a different closed-set detector, write a new class with an
`infer(image, conf) -> list[Detection]` method and swap it in `pipeline.py`.
Grounding DINO and OWLv2 live in their own classes with the same interface.

---

## Running the tests

```powershell
pytest -q
```

Expected output:
```
12 passed, 1 skipped in ~3s
```

The skipped test is the video-writer roundtrip, which auto-skips if your
OpenCV build has no encoder backend.

Tests cover: config persistence, detection geometry, label rendering at frame
edges, bounded queue drop behavior, event deduplication, screenshot saving,
video file reading, hardware reporting, and confidence filtering.

The test suite is intentionally free of GPU requirements — it runs on a laptop
in CI.

---

## Troubleshooting

### CUDA not available

`torch.cuda.is_available()` returns `False`.

1. Run `nvidia-smi`. If it fails, install/update the NVIDIA driver and reboot.
2. If `nvidia-smi` works but torch still says `False`, your wheel is CPU-only:
   ```powershell
   pip uninstall -y torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --timeout 120 --retries 10
   ```
3. Confirm the pip prompt is inside `(.venv)`.

### `AttributeError: module 'torch' has no attribute 'accelerator'`

You have `transformers` 5.x with torch 2.5.x. Fix:

```powershell
pip install "transformers==4.46.1"
```

### `Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:178`

FFmpeg's decoder is being released mid-decode. This occurs when closing the
app while a video is playing, opening a new source while the previous one is
still running, or dragging the seek slider during playback. The current
version of `main_window._stop()` serializes teardown; if you see the assertion
after pulling the latest code, add this to the top of `run.py`:

```python
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"
```

### Camera fails to open

- Windows: **Settings → Privacy & security → Camera → Let desktop apps access
  your camera → On**.
- Try indices `1`, `2`, … in the camera dialog.
- Close other apps that use the camera (Zoom, Teams, OBS, browser tabs).

### Video file won't open at all

Some codecs (HEVC in MKV, AV1, 10-bit color) aren't supported by the default
OpenCV Windows build. Re-encode to H.264 MP4:

```powershell
winget install Gyan.FFmpeg
# then, in a new terminal:
ffmpeg -i "input.mkv" -c:v libx264 -preset medium -crf 20 -c:a aac "output.mp4"
```

### Out of memory (CUDA OOM)

- Settings → **Process every N frames** → 4 or 6.
- Switch detector to `yolo11n.pt`.
- Disable segmentation.
- Restart the app to clear the CUDA cache.
- Close other GPU apps (games, browser with hardware acceleration).

### Model download fails or times out

- Check that `huggingface.co` and `github.com` are reachable.
- If you're behind a proxy, set `HTTPS_PROXY` before launching.
- Retry from the Models page — interrupted Hugging Face downloads resume.

### App freezes for a few seconds on first detection

That's the first-time model load. Subsequent inferences reuse the loaded model.

---

## Known limitations

- **Reference-image matching is visual similarity, not identity.** Two
  visually similar objects will match. Always confirm visually.
- **Open-vocabulary detectors miss unusual phrasings.** Grounding DINO works
  best with lowercase noun phrases, one object per phrase.
- **Grounding DINO and OWLv2 are 5–15× slower than YOLO.** Use frame skipping
  (`process_every_n_frames ≥ 3`) whenever they're active.
- **BLIP produces short captions, not open-ended Q&A.** For questions like
  "what text is visible?" you'd need a larger VLM (see Roadmap).
- **No facial recognition or biometric features exist and none will be added.**
  VisionScope is scoped to objects, scenes, and non-identifying visual
  attributes.
- **Annotated video export uses the mp4v codec.** Most players accept it; if
  yours doesn't, use a different player (VLC is the most forgiving).
- **Speed control applies only to file playback.** Cameras run in real time.

---

## Roadmap

- [ ] Qwen2-VL integration for open-ended visual Q&A
- [ ] Click-to-segment (prompt SAM with a mouse click)
- [ ] Multi-object selection in the right panel
- [ ] Timeline scrubber over event history
- [ ] Offline batch processing mode (process a whole folder of videos)
- [ ] Optional cloud VLM backend (opt-in, env-var credentials, explicit
      consent dialog before frames leave the machine)
- [ ] Fine-tuning wizard for custom object classes

---

## Privacy

- **All processing is local by default.** No frame, image, or video ever
  leaves your machine.
- **No telemetry.** No usage data is collected or transmitted.
- **No accounts, no cloud.** VisionScope has no network dependency beyond
  one-time model downloads from Hugging Face and GitHub.
- Model weights are cached locally; you can run entirely offline after the
  first download.
- When a cloud VLM backend is added (see Roadmap), it will be strictly
  opt-in with an explicit consent dialog before any frame is transmitted.

---

## License and model attributions

VisionScope itself is released under the **MIT License** — see
`LICENSE`.

However, VisionScope uses third-party models with their own licenses:

| Component | License | Commercial use |
|---|---|---|
| **Ultralytics YOLO11** (code + weights) | AGPL-3.0 | ⚠️ Requires open-source or an Ultralytics Enterprise license |
| **Grounding DINO** (IDEA-Research) | Apache-2.0 | Typically fine |
| **OWLv2** (Google) | Apache-2.0 | Fine |
| **MobileSAM** | Apache-2.0 | Fine |
| **BLIP** (Salesforce) | BSD-3-Clause | Fine, attribution required |
| **PySide6** | LGPL-3.0 | Fine for dynamic linking |
| **PyTorch** | BSD-3-Clause | Fine |

**If you plan to ship a closed-source product using this application**, you
must address the Ultralytics AGPL requirement. Options:

1. Purchase an Ultralytics Enterprise license.
2. Replace the YOLO detector with an Apache-2.0 alternative (e.g., a
   torchvision detection model or a custom-trained detector).
3. Disable the YOLO path entirely and rely on Grounding DINO.

---

## Author

**Roham Tavafoghi**
Built as a real, working computer vision application — not a tutorial project.

Contributions, issues, and pull requests are welcome.

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO11 and the
  built-in ByteTrack / BoT-SORT / SAM wrappers.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for
  Grounding DINO, OWLv2, and BLIP.
- [IDEA-Research](https://github.com/IDEA-Research/GroundingDINO) for
  Grounding DINO.
- [Google Research](https://github.com/google-research/scenic) for OWLv2.
- [Salesforce Research](https://github.com/salesforce/BLIP) for BLIP.
- The PySide6 and OpenCV teams for the underlying UI and video frameworks.

---

## Star history

If VisionScope is useful to you, please consider starring the repo — it helps
others find it.
