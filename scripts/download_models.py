import os
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = [
    {
        'name': 'GFPGANv1.4.pth',
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
    },
]

for model in MODELS:
    dest = os.path.join(MODELS_DIR, model['name'])
    if os.path.exists(dest):
        print(f"Already exists, skipping: {model['name']}")
        continue
    print(f"Downloading {model['name']}...")
    urllib.request.urlretrieve(model['url'], dest)
    print(f"Saved to {dest}")

print("\nNote: inswapper_128.onnx must be downloaded manually from:")
print("https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view")
print(f"Place it in: {MODELS_DIR}")