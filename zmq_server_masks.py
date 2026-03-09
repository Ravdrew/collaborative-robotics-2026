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

PORT = 5555
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
    meta = sock.recv_json()
    data = sock.recv()
    img = np.frombuffer(data, dtype=meta["dtype"]).reshape(meta["shape"])

    prompt = meta["prompt"]

    # Convert to PIL and compute image backbone once
    pil_img = Image.fromarray(img)
    inference_state = processor.set_image(pil_img)

    # Run prompt
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    num_objects = len(inference_state["scores"])

    # Combine masks
    H, W = img.shape[:2]
    combined_mask = np.zeros((H, W), dtype=bool)

    for i in range(num_objects):
        mask = inference_state["masks"][i].squeeze(0).cpu().numpy() > 0.5
        combined_mask |= mask

    combined_mask = combined_mask.astype(np.uint8)

    # Send back mask
    sock.send_json(
        {
            "shape": list(combined_mask.shape),
            "dtype": str(combined_mask.dtype),
            "num_objects": num_objects,
        },
        zmq.SNDMORE,
    )
    sock.send(combined_mask.tobytes())

    print(f"Processed frame {img.shape}, prompt={prompt}, found {num_objects} object(s)")
