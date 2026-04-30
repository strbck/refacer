# Refacer

![Status: Early Development](https://img.shields.io/badge/status-early%20development-orange)
![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Platform: macOS | Linux | Windows](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

**Batch face anonymization and metadata scrubbing for photojournalists and activist photographers.**

Refacer is a locally-run, fully offline desktop tool that replaces detected faces in photos with AI-generated alternatives and strips all identifying metadata. No cloud. No API calls. Your images never leave your machine.

> ⚠️ **This project is in early development.** The pipeline is not yet complete and the tool is not ready for production use.

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

## Installation

> ⚠️ Installation instructions will be finalized at v0.1.0. The steps below reflect the intended setup flow.

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
python scripts/download_models.py  # downloads inswapper and GFPGAN model weights (~500MB)
```

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

|Platform|Supported                   |
|--------|----------------------------|
|macOS   |✅                           |
|Linux   |✅                           |
|Windows |⚠️ Beta — testing in progress|

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
- Processing time varies by hardware — benchmarks will be published at v0.1.0

-----

## Roadmap

- [ ] v0.1.0 — core pipeline (detection, face replacement, enhancement, metadata scrub, Gradio UI)
- [ ] v0.2.0 — confidence threshold UI controls, manual review queue for low-confidence detections
- [ ] v0.3.0 — higher-realism enhancement options
- [ ] v1.0.0 — packaged installers for macOS, Linux, and Windows

-----

## Contributing

This project is in early development and not yet ready for contributions. Once the core pipeline is stable at v0.1.0, contributions will be welcome. Watch this repo for updates.

-----

## Ethics & Intended Use

Refacer is built specifically for photojournalists, documentary photographers, activists, and researchers who need to protect the identities of vulnerable or at-risk individuals in images.

It is not intended for any use that obscures identity for deceptive, harmful, or illegal purposes. This software is not intended for use on images where anonymization could obstruct a legitimate legal or journalistic investigation.

-----

## License

[AGPL-3.0](LICENSE)

Refacer is free and open source software. The AGPL license ensures that any forks or deployments — including hosted versions — must also remain open source. 

-----