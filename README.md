# Face Recognition (Python, OpenCV, dlib)

A small, resume-friendly face recognition demo built with Python, OpenCV, and `face_recognition` (dlib). It loads a set of known face images, then recognizes faces in a live webcam stream.

Optional: if `deepface` is installed, the script also displays an estimated age.

## What this project demonstrates

- Real-time face detection + face embedding extraction (`face_recognition`/dlib)
- Nearest-match identification against a known face gallery
- Simple computer-vision UI overlays (name + confidence)
- Defensive coding (missing images, no-face-in-image handling, optional dependencies)

## Repository contents

This repository is intentionally small and centered around one script:

- [face.py](face.py): main script
- [dlib-19.19.0-cp38-cp38-win_amd64.whl](dlib-19.19.0-cp38-cp38-win_amd64.whl): Windows wheel used by `face_recognition` (Python 3.8)
- Sample media files (images / pptx / docx / demo video)

Note: the sample images in the root folder are demo assets. For a professional portfolio, replace them with images you have permission to use (see “Ethics & privacy”).

## Requirements

- Windows 10/11
- Python 3.8 (recommended for the included dlib wheel)
- A working webcam

Python packages:

- `opencv-python`
- `face-recognition`

Optional (for age estimation):

- `deepface` (may pull heavier ML dependencies depending on your environment)

## Setup (Windows)

The repo includes a prebuilt `dlib` wheel for Python 3.8 on 64-bit Windows. If you use a different Python version, you may need to build/install `dlib` another way.

1) Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Upgrade pip

```powershell
python -m pip install --upgrade pip
```

3) Install dependencies

```powershell
pip install opencv-python face-recognition
```

If `face-recognition` fails due to `dlib`, install the provided wheel:

```powershell
pip install .\dlib-19.19.0-cp38-cp38-win_amd64.whl
pip install face-recognition
```

Optional (age estimation):

```powershell
pip install deepface
```

## Configure known faces

Edit the `people` list in [face.py](face.py). Each entry maps a display name to an image file path.

Guidelines for best results:

- Use a clear, front-facing photo
- One face per image
- Good lighting, minimal occlusion

If a “known” image contains no detectable face, the script will raise a clear error message.

## Run

```powershell
python face.py
```

Controls:

- Press `q` to quit.

## How matching works (high level)

1) Known images are encoded into 128-D face embeddings (dlib via `face_recognition`).
2) Each detected face in the webcam frame is encoded the same way.
3) The script computes distances between the detected embedding and all known embeddings.
4) If the smallest distance is below a threshold (default: `0.60`), the corresponding name is shown. Otherwise, the face is labeled “Unknown”.

The on-screen “Confidence” value is displayed as `1 - distance` (a simple heuristic for readability; it is not a calibrated probability).

## Troubleshooting

- Camera doesn’t open
  - Close other apps using the webcam (Zoom/Teams/Browser tabs).
  - If you have multiple cameras, adjust `camera_index` in the code.

- `ImportError` / build errors for `dlib`
  - Use Python 3.8 on Windows and install the included wheel.
  - On other Python versions, you may need a different wheel or a build toolchain.

- `deepface` install is slow / fails
  - Age estimation is optional. The script runs without it.
  - If installed, `deepface` may require additional ML dependencies depending on your system.

## Limitations

- This is a demo; it is not production-grade authentication.
- Recognition accuracy depends heavily on lighting, camera quality, pose, and dataset quality.
- Face recognition may perform differently across demographic groups; evaluate carefully.

## Ethics & privacy

- Obtain consent before collecting or using any facial images.
- Avoid using face recognition for high-stakes decisions.
- Store biometric data securely and follow applicable laws/policies.

## Project notes for a resume

- Highlight: real-time CV pipeline (capture → detect → embed → match → overlay)
- Highlight: robustness improvements (optional `deepface`, clear error handling)
- Highlight: Windows-specific dependency handling (`dlib` wheel constraints)
