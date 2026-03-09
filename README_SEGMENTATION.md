# SAM3 ZMQ Segmentation Pipeline

Real-time text-prompted segmentation using SAM3 over ZMQ. A RealSense camera client captures RGB frames, sends them to a GPU server running SAM3, and displays the segmentation overlay.

## Architecture

```
[Robot]                          [GPU Server]
realsense_zmq_client.py  --->   zmq_image_server.py
  - Captures RGB + depth          - Loads SAM3 model once
  - Sends RGB frame + prompt      - Runs text-prompted segmentation
  - Displays overlay result  <--- - Returns RGB overlay (masks blended at 50%)
```

**Protocol (ZMQ REQ/REP):**
- Client → Server: JSON metadata `{"shape", "dtype", "prompt"}` + raw RGB bytes
- Server → Client: JSON metadata `{"shape", "dtype", "num_objects"}` + RGB overlay bytes

## Setup
~
The server needs the SAM3 repo with a uv venv (PyTorch 2.10 + CUDA 12.8):

```bash
cd ~/sam3
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

The client only needs `pyrealsense2`, `numpy`, `zmq`, and `opencv-python`.

## Running

**1. Start the server (GPU machine):**

```bash

# ssh command:
ssh giuse@100.77.113.90
cd ~/sam3 && source .venv/bin/activate && python ~/zmq_image_server.py
```

Wait for "SAM 3 model loaded." before starting the client.

**2. Start the client (robot):**

```bash
python ~/realsense_zmq_client.py
```

Two OpenCV windows appear: "RealSense RGB" (raw camera) and "SAM3 Result" (segmentation overlay). Press `q` to quit.

## Changing the text prompt

Edit the `"prompt"` value in `realsense_zmq_client.py` line 38:

```python
"prompt": "banana"  # change to whatever object you want to segment
```

## Files

- `zmq_image_server.py` — SAM3 segmentation server (runs on GPU machine)
- `realsense_zmq_client.py` — RealSense camera client (runs on robot)
- `sam3/` — SAM3 model repo
