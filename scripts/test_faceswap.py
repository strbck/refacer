import cv2
import insightface
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align
from gfpgan import GFPGANer

# Base paths — relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
INPUT_DIR = os.path.join(REPO_ROOT, 'input')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')
INSWAPPER_PATH = os.path.join(MODELS_DIR, 'inswapper_128.onnx')
GFPGAN_PATH = os.path.join(MODELS_DIR, 'GFPGANv1.4.pth')

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate models exist before doing anything
for path, name in [(INSWAPPER_PATH, 'inswapper_128.onnx'), (GFPGAN_PATH, 'GFPGANv1.4.pth')]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {name} — run scripts/download_models.py first")

# Collect input images
supported_exts = ('.jpg', '.jpeg', '.png', '.tiff', '.webp')
input_images = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(supported_exts)
]

if not input_images:
    raise FileNotFoundError(f"No images found in {INPUT_DIR} — add at least one image to process")

print(f"Found {len(input_images)} image(s) to process")

# Initialize models once
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1024, 1024))

swapper = get_model(INSWAPPER_PATH, providers=['CPUExecutionProvider'])

enhancer = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# Process each image
for filename in input_images:
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    print(f"\nProcessing: {filename}")

    img = cv2.imread(input_path)
    if img is None:
        print(f"  Could not read {filename}, skipping")
        continue

    result = img.copy()

    # Detect faces
    faces = app.get(img)
    img = cv2.imread(input_path)
    result = img.copy()
    print(f"  Found {len(faces)} face(s)")

    if len(faces) == 0:
        print(f"  No faces detected, skipping")
        continue

    for face in faces:
        # 1. Generate random identity
        random_embedding = np.random.randn(512).astype(np.float32)
        random_embedding /= np.linalg.norm(random_embedding)

        # 2. Compute latent
        latent = random_embedding.reshape((1, -1))
        latent = np.dot(latent, swapper.emap)
        latent /= np.linalg.norm(latent)

        # 3. Crop and align face region
        aimg, M = face_align.norm_crop2(result, face.kps, swapper.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / swapper.input_std, swapper.input_size,
                                      (swapper.input_mean, swapper.input_mean, swapper.input_mean),
                                      swapRB=True)

        # 4. Run inference
        pred = swapper.session.run(swapper.output_names,
                                    {swapper.input_names[0]: blob,
                                     swapper.input_names[1]: latent})[0]

        # 5. Post-process output
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        # 6. Color correct to match original skin tone
        bbox = face.bbox.astype(int)
        orig_face_region = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        for c in range(3):
            orig_mean = orig_face_region[:,:,c].mean()
            orig_std = orig_face_region[:,:,c].std()
            fake_mean = bgr_fake[:,:,c].mean()
            fake_std = bgr_fake[:,:,c].std()
            bgr_fake[:,:,c] = np.clip(
                (bgr_fake[:,:,c] - fake_mean) * (orig_std / (fake_std + 1e-6)) + orig_mean,
                0, 255
            ).astype(np.uint8)

        # 7. Warp back to full image size
        IM = cv2.invertAffineTransform(M)
        bgr_fake_full = cv2.warpAffine(bgr_fake, IM, (result.shape[1], result.shape[0]), borderValue=0.0)

        # 8. Build landmark-based mask
        kps = face.kps.astype(int)
        eye_center = ((kps[0][0] + kps[1][0]) // 2, (kps[0][1] + kps[1][1]) // 2)
        mouth_center = ((kps[3][0] + kps[4][0]) // 2, (kps[3][1] + kps[4][1]) // 2)
        face_width = int(np.linalg.norm(kps[1] - kps[0]) * 2.2)
        face_height = int(np.linalg.norm(mouth_center - np.array(eye_center)) * 2.8)
        center_x = (eye_center[0] + mouth_center[0]) // 2
        center_y = (eye_center[1] + mouth_center[1]) // 2

        hull_points = np.array([
            [eye_center[0] - face_width // 2, eye_center[1] - face_height // 4],
            [eye_center[0] + face_width // 2, eye_center[1] - face_height // 4],
            [kps[1][0] + face_width // 4, kps[1][1] + face_height // 6],
            [kps[4][0] + face_width // 6, kps[4][1] + face_height // 6],
            [mouth_center[0], mouth_center[1] + face_height // 4],
            [kps[3][0] - face_width // 6, kps[3][1] + face_height // 6],
            [kps[0][0] - face_width // 4, kps[0][1] + face_height // 6],
        ], dtype=np.int32)

        mask = np.zeros((result.shape[0], result.shape[1]), dtype=np.float32)
        cv2.fillConvexPoly(mask, cv2.convexHull(hull_points), 255)
        k = 31
        mask = cv2.GaussianBlur(mask, (2*k+1, 2*k+1), 0)
        mask = mask / 255
        mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])

        # 9. Seamless clone composite
        center_point = (center_x, center_y)
        result = cv2.seamlessClone(
            bgr_fake_full.astype(np.uint8),
            result,
            (mask * 255).astype(np.uint8),
            center_point,
            cv2.NORMAL_CLONE
        )

    # 10. GFPGAN enhancement — runs once after all faces processed
    _, _, result = enhancer.enhance(
        result,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    cv2.imwrite(output_path, result)
    print(f"  Saved to output/{filename}")

print("\nAll done!")