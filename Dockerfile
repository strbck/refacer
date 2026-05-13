FROM python:3.11-slim

# System dependencies:
#   build-essential + cmake  — compile insightface C++ extensions (no pre-built wheel on slim)
#   libgl1 + libglib2.0-0   — required by opencv-python-headless at runtime
#   libimage-exiftool-perl  — metadata scrubbing (exiftool)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /refacer

# Install Python dependencies.
# torch is CPU-only — the default PyPI wheel includes CUDA and is ~800 MB larger.
#
# basicsr (transitive dep of gfpgan) uses a legacy setup.py whose setup_requires
# fetches CUDA wheels that conflict with each other.  Pre-installing torch+numpy
# and then installing basicsr with --no-build-isolation makes it reuse the
# already-present deps instead of triggering that CUDA resolution.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.1.2+cpu" "torchvision==0.16.2+cpu" "numpy<2" Cython
RUN pip install --no-cache-dir --no-build-isolation basicsr
RUN pip install --no-cache-dir --no-build-isolation \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy source.
COPY . .

# Pre-download GFPGAN weights into /refacer/weights/ — a path that is NOT
# volume-mounted at runtime.  The models/ volume only needs to contain the
# user-supplied inswapper_128.onnx; mounting it must not shadow these weights.
RUN python scripts/download_models.py --dest /refacer/weights
ENV GFPGAN_MODEL_PATH=/refacer/weights/GFPGANv1.4.pth

# Pre-download InsightFace buffalo_l detection weights into the image.
# This avoids a ~300 MB download on first container start.
RUN python - <<'EOF'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
EOF

# models/ is mounted at runtime (contains inswapper_128.onnx).
# input/ and output/ are mounted for image I/O.
VOLUME ["/refacer/models", "/refacer/input", "/refacer/output"]

EXPOSE 7860

CMD ["python", "ui/app.py"]
