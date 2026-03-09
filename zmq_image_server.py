import os
import sys

import numpy as np
import torch
import zmq
from PIL import Image

# Ensure sam3 package is importable
sys.path.insert(0, os.path.expanduser("~/sam3"))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS

PORT = 5555
ALPHA = 0.5
CONFIDENCE = 0.5

# --- Torch setup for Ampere+ GPUs ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# --- Load SAM3 model once ---
print("Loading SAM 3 model...")
bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=CONFIDENCE)
print("SAM 3 model loaded.")

# --- ZMQ setup ---
ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind(f"tcp://0.0.0.0:{PORT}")
print(f"Listening on port {PORT}...")

while True:
    # Receive RGB image + metadata
    # Supports "prompt" (single string) or "prompts" (list of strings)
    meta = sock.recv_json()
    data = sock.recv()
    img = np.frombuffer(data, dtype=meta["dtype"]).reshape(meta["shape"])

    # Normalize to a list of prompts
    if "prompts" in meta:
        prompts = meta["prompts"]
    else:
        prompts = [meta.get("prompt", "object")]

    # Convert to PIL and compute image backbone once
    pil_img = Image.fromarray(img)
    inference_state = processor.set_image(pil_img)

    # Run inference for each prompt, collect all masks
    overlay = img.copy()
    color_idx = 0
    total_objects = 0
    per_prompt = {}

    for prompt in prompts:
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

        num_objects = len(inference_state["scores"])
        per_prompt[prompt] = num_objects
        total_objects += num_objects

        for i in range(num_objects):
            mask = inference_state["masks"][i].squeeze(0).cpu().numpy()  # (H, W)
            color = COLORS[color_idx % len(COLORS)]
            color255 = (color * 255).astype(np.uint8)
            mask_bool = mask > 0.5
            for c in range(3):
                overlay[..., c][mask_bool] = (
                    ALPHA * color255[c] + (1 - ALPHA) * overlay[..., c][mask_bool]
                ).astype(np.uint8)
            color_idx += 1

    # Send back the overlay image
    sock.send_json(
        {
            "shape": list(overlay.shape),
            "dtype": str(overlay.dtype),
            "num_objects": total_objects,
            "per_prompt": per_prompt,
        },
        zmq.SNDMORE,
    )
    sock.send(overlay.tobytes())
    print(f"Processed frame {img.shape}, prompts={prompts}, found {total_objects} object(s) {per_prompt}")
