"""
refacer.swap
~~~~~~~~~~~~
Per-face swap logic: identity generation, inference, colour correction,
and compositing back into the full image.

Each public function is intentionally narrow so that pipeline.py can
wrap them individually in try/except and continue on partial failure.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_random_latent(swapper) -> np.ndarray:
    """Return a random normalised latent vector compatible with *swapper*."""
    embedding = np.random.randn(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    latent = embedding.reshape((1, -1))
    latent = np.dot(latent, swapper.emap)
    latent /= np.linalg.norm(latent)
    return latent


def swap_face(swapper, image: np.ndarray, face) -> np.ndarray:
    """
    Swap a single detected *face* in *image* and composite the result back.

    Parameters
    ----------
    swapper:
        Loaded inswapper model from ModelBundle.
    image:
        Full BGR image (uint8) — used for colour reference and compositing.
    face:
        A face object returned by insightface FaceAnalysis.get().

    Returns
    -------
    result : np.ndarray
        Full image (same shape as *image*) with this face replaced.
    """
    from insightface.utils import face_align

    # 1. Random identity
    latent = generate_random_latent(swapper)

    # 2. Crop and align
    aimg, M = face_align.norm_crop2(image, face.kps, swapper.input_size[0])
    blob = cv2.dnn.blobFromImage(
        aimg,
        1.0 / swapper.input_std,
        swapper.input_size,
        (swapper.input_mean, swapper.input_mean, swapper.input_mean),
        swapRB=True,
    )

    # 3. Inference
    pred = swapper.session.run(
        swapper.output_names,
        {swapper.input_names[0]: blob, swapper.input_names[1]: latent},
    )[0]

    # 4. Post-process
    img_fake = pred.transpose((0, 2, 3, 1))[0]
    bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

    # 5. Colour-correct to match original skin tone
    bbox = face.bbox.astype(int)
    orig_face_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if orig_face_region.size > 0:
        for c in range(3):
            orig_mean = orig_face_region[:, :, c].mean()
            orig_std = orig_face_region[:, :, c].std()
            fake_mean = bgr_fake[:, :, c].mean()
            fake_std = bgr_fake[:, :, c].std()
            bgr_fake[:, :, c] = np.clip(
                (bgr_fake[:, :, c] - fake_mean)
                * (orig_std / (fake_std + 1e-6))
                + orig_mean,
                0,
                255,
            ).astype(np.uint8)

    # 6. Warp back to full image size
    IM = cv2.invertAffineTransform(M)
    bgr_fake_full = cv2.warpAffine(
        bgr_fake,
        IM,
        (image.shape[1], image.shape[0]),
        borderValue=0.0,
    )

    # 7. Landmark-based mask
    kps = face.kps.astype(int)
    eye_center = (
        (kps[0][0] + kps[1][0]) // 2,
        (kps[0][1] + kps[1][1]) // 2,
    )
    mouth_center = (
        (kps[3][0] + kps[4][0]) // 2,
        (kps[3][1] + kps[4][1]) // 2,
    )
    face_width = int(np.linalg.norm(kps[1] - kps[0]) * 2.2)
    face_height = int(
        np.linalg.norm(np.array(mouth_center) - np.array(eye_center)) * 2.8
    )
    center_x = (eye_center[0] + mouth_center[0]) // 2
    center_y = (eye_center[1] + mouth_center[1]) // 2

    hull_points = np.array(
        [
            [eye_center[0] - face_width // 2, eye_center[1] - face_height // 4],
            [eye_center[0] + face_width // 2, eye_center[1] - face_height // 4],
            [kps[1][0] + face_width // 4, kps[1][1] + face_height // 6],
            [kps[4][0] + face_width // 6, kps[4][1] + face_height // 6],
            [mouth_center[0], mouth_center[1] + face_height // 4],
            [kps[3][0] - face_width // 6, kps[3][1] + face_height // 6],
            [kps[0][0] - face_width // 4, kps[0][1] + face_height // 6],
        ],
        dtype=np.int32,
    )

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(hull_points), 255)
    k = 31
    mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)
    mask = mask / 255
    mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])

    # 8. Seamless clone composite
    result = cv2.seamlessClone(
        bgr_fake_full.astype(np.uint8),
        image,
        (mask * 255).astype(np.uint8),
        (center_x, center_y),
        cv2.NORMAL_CLONE,
    )

    return result
