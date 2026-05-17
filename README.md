# Refacer

![Version: v0.1.0](https://img.shields.io/badge/version-v0.1.0-green)
![License: PolyForm-NC-1.0.0](https://img.shields.io/badge/license-PolyForm--NC--1.0.0-blue)
![Platform: macOS 13+ | Linux | Windows 10/11 WSL2](https://img.shields.io/badge/platform-macOS%2013%2B%20%7C%20Linux%20%7C%20Windows%2010%2F11%20WSL2-lightgrey)

**Batch face anonymization and metadata scrubbing for photojournalists and activist photographers.**

Refacer is a locally-run, fully offline desktop tool that replaces detected faces in photos with AI-generated alternatives and strips all identifying metadata. No cloud. No API calls. Your images never leave your machine.

-----

## Why This Exists

Photographs are testimony. They capture moments, emotions, and stories that words can't. But sharing images of protests, demonstrations, or sensitive situations can put the people in them at risk — through facial recognition, embedded GPS coordinates, camera serial numbers, and more.

Refacer is built on a simple premise: you shouldn't have to choose between telling the story and protecting the people in it.

-----

## Features

- **Batch face anonymization** — process entire folders of images in one run
- **AI face replacement** — each detected face is replaced with a uniquely generated identity; results are non-deterministic and cannot be reversed
- **Face enhancement** — replaced faces are upscaled and refined for natural, high-quality results
- **Full metadata scrubbing** — strips all EXIF, XMP, and IPTC data including GPS coordinates, timestamps, and device identifiers
- **Fully offline** — no network calls at runtime, ever
- **CPU-first** — runs on modest hardware without a discrete GPU
- **Simple local UI** — browser-based Gradio interface, no technical setup required
- **CLI support** — scriptable via `python -m refacer` for advanced workflows

-----

## How It Works

Refacer runs each image through a modular pipeline:

```
Input Folder → Face Detection → Identity Generation → Face Swap → Enhancement → Composite → Metadata Scrub → Output Folder
```

Each face receives a randomly generated embedding with no seeding or shared state, ensuring unique replacements across every run.

-----

## Docker (recommended)

Docker is the easiest way to run Refacer — no Python environment or system dependency setup required.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — macOS 13+, Windows 10/11 64-bit with WSL2 enabled, or a supported Linux distro
- `inswapper_128.onnx` downloaded manually from [Google Drive](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view) and placed in a `models/` directory inside the repo

### Run

```bash
git clone https://github.com/strbck/refacer.git
cd refacer
mkdir -p models input output
# Place inswapper_128.onnx in models/ before continuing
docker compose up --build
```

The first build takes several minutes — it installs dependencies and pre-downloads the GFPGAN and InsightFace weights into the image. Subsequent starts are fast.

Open [http://localhost:7860](http://localhost:7860) in your browser. Drop images into the UI, click **Run**, and find anonymized results in your local `output/` folder.

**CLI via Docker:**

```bash
docker compose run --rm refacer python -m refacer --input /app/input --output /app/output
```

-----

## Installation (manual)

### Prerequisites

- Python 3.10+
- `exiftool` (system dependency)

**macOS:**

```bash
brew install exiftool
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt install libimage-exiftool-perl
```

**Windows:**
Download and install from [exiftool.org](https://exiftool.org). Add to your system PATH.

### Install Refacer

```bash
git clone https://github.com/strbck/refacer.git
cd refacer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py  # downloads GFPGANv1.4 weights automatically
```

`inswapper_128.onnx` must be downloaded manually from [Google Drive](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view) and placed in `models/` inside the repo.

-----

## Usage

### UI (recommended)

```bash
python ui/app.py
```

Opens a local Gradio interface in your browser. Select your input folder, hit **Run**, and find anonymized images in your output folder.

### CLI

```bash
python -m refacer --input /path/to/photos --output /path/to/output
```

-----

## Platform Support

Refacer currently requires Docker. Supported environments are those that can run Docker Engine or Docker Desktop:

|Platform              |Supported                              |
|----------------------|---------------------------------------|
|macOS 13+             |✅                                      |
|Linux (supported distros)|✅                                  |
|Windows 10/11 64-bit  |✅ Requires WSL2                        |
|Older / unsupported OS|❌ Not supported at this time           |

-----

## Technical Stack

|Component         |Library                              |
|------------------|-------------------------------------|
|Face Detection    |InsightFace (buffalo_l backend)      |
|Face Replacement  |inswapper_128 (random identity swap) |
|Face Enhancement  |GFPGAN v1.4                          |
|Metadata Scrubbing|exiftool                             |
|UI                |Gradio (local only)                  |
|Inference         |onnxruntime (CPU)                    |

-----

## Privacy & Security

- Refacer is **fully air-gapped at runtime** — no telemetry, no model API calls, no external connections of any kind
- All processing happens on your local machine
- Output images are verified to contain zero metadata fields before being written
- Face replacement uses randomly generated identity embeddings — reversal attacks are infeasible by design

-----

## Limitations

- Very small faces in dense crowd shots may be missed or flagged as low-confidence
- Extreme profile angles and heavily occluded faces (e.g. surgical masks + sunglasses) may not be detected reliably
- Processing time varies by hardware; no benchmarks published yet

-----

## Roadmap

- [x] v0.1.0 — core pipeline (detection, face replacement, enhancement, metadata scrub, Gradio UI)
- [ ] v0.2.0 — confidence threshold UI controls, manual review queue for low-confidence detections
- [ ] v0.3.0 — higher-realism enhancement options
- [ ] v1.0.0 — packaged installers for macOS, Linux, and Windows

-----

## Contributing

The core pipeline is stable at v0.1.0 and contributions are welcome. Please open an issue before starting significant work so we can discuss approach and avoid duplication.

-----

## Third-Party Model Notices

Refacer depends on pre-trained model weights that are not bundled in this repository but are required to run the software. These weights carry their own license restrictions:

| Model | Source | License |
|---|---|---|
| `inswapper_128.onnx` | InsightFace | Non-commercial / research use only — InsightFace pretrained models are "available for non-commercial research purposes only" |
| `GFPGANv1.4.pth` | Tencent ARC / NTU S-Lab | [NTU S-Lab License 1.0](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) — non-commercial academic research only |
| `buffalo_l` (downloaded automatically by InsightFace at first run) | InsightFace | Non-commercial / research use only |

These restrictions apply regardless of the terms of the Refacer source code license. Refacer's own PolyForm Noncommercial license is intentionally aligned with these upstream constraints.

-----

## Ethics & Intended Use

Refacer is built specifically for photojournalists, documentary photographers, activists, and researchers who need to protect the identities of vulnerable or at-risk individuals in images.

It is not intended for any use that obscures identity for deceptive, harmful, or illegal purposes. This software is not intended for use on images where anonymization could obstruct a legitimate legal or journalistic investigation.

-----

## License

[PolyForm Noncommercial License 1.0.0](LICENSE)

Refacer is free for any non-commercial purpose. You may use, copy, modify, and redistribute it provided you do not use it for commercial purposes and any redistribution carries the same permissions. Commercial use requires a separate agreement with the project maintainers.

This license was chosen to align with the non-commercial restrictions of the upstream model weights Refacer depends on (see [Third-Party Model Notices](#third-party-model-notices) above).

-----