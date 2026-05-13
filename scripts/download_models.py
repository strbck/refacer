import argparse
import os
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(REPO_ROOT, 'models')

MODELS = [
    {
        'name': 'GFPGANv1.4.pth',
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
    },
]


def download(dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    for model in MODELS:
        path = os.path.join(dest, model['name'])
        if os.path.exists(path):
            print(f"Already exists, skipping: {model['name']}")
            continue
        print(f"Downloading {model['name']}...")
        urllib.request.urlretrieve(model['url'], path)
        print(f"Saved to {path}")

    print("\nNote: inswapper_128.onnx must be downloaded manually from:")
    print("https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view")
    print(f"Place it in: {DEFAULT_DEST}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Refacer model weights.")
    parser.add_argument(
        '--dest',
        default=DEFAULT_DEST,
        metavar='DIR',
        help='Directory to save weights into (default: models/ in the repo root).',
    )
    args = parser.parse_args()
    download(args.dest)
