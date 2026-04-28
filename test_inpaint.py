from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from PIL import Image
import numpy as np

# Load image and mask
image = Image.open(r"C:\Users\carte\Downloads\test.png").convert("RGB")
mask = Image.open(r"C:\Users\carte\Downloads\mask.png").convert("L")

image_np = np.array(image)
mask_np = np.array(mask)

# Load model
model = ModelManager(name="lama", device="cpu")

config = Config(
    ldm_steps=1,
    hd_strategy=HDStrategy.ORIGINAL,
    hd_strategy_crop_margin=32,
    hd_strategy_crop_trigger_size=512,
    hd_strategy_resize_limit=512,
)

result = model(image_np, mask_np, config)

result_squeezed = result.squeeze()
result_final = np.clip(result_squeezed, 0, 255).astype(np.uint8)

# Swap RGB channels back — lama-cleaner is outputting BGR internally
result_fixed = result_final[:, :, ::-1].copy()

Image.fromarray(result_fixed).save(r"C:\Users\carte\Downloads\output.png")
print("Done!")