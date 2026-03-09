import pyrealsense2 as rs
import numpy as np
import zmq
import cv2

REMOTE_IP = "100.77.113.90"
PORT = 5555

# ZMQ setup
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect(f"tcp://{REMOTE_IP}:{PORT}")
print(f"Connecting to tcp://{REMOTE_IP}:{PORT}...")

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # BGR, 640x480
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Send RGB image + text prompts
        sock.send_json({"shape": rgb_image.shape, "dtype": str(rgb_image.dtype), "prompts": ["banana", "phone"]}, zmq.SNDMORE)
        sock.send(rgb_image.tobytes())

        # Receive result image
        meta = sock.recv_json()
        data = sock.recv()
        result_image = np.frombuffer(data, dtype=meta["dtype"]).reshape(meta["shape"])

        # Display original and SAM3 overlay side by side
        # Convert overlay from RGB back to BGR for OpenCV display
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("RealSense RGB", color_image)
        cv2.imshow("SAM3 Result", result_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    ctx.destroy()
