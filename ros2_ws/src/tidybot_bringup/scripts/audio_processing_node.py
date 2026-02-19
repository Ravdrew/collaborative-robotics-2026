#!/usr/bin/env python3
import threading
import time
import json
from typing import Any, Dict, Literal
import wave

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from tidybot_msgs.srv import AudioRecord

# Google Cloud Speech-to-Text
from google.cloud import speech

# Gemini (Developer API)
from google import genai
from pydantic import BaseModel, Field


# ---------- Gemini structured output schema ----------
class LLMParse(BaseModel):
    pick_target: str = Field(description="The object/fruit to pick up (e.g., 'apple', 'banana', 'water bottle').")
    place_target: Literal["none", "basket", "hand"] = Field(
        description="Where to place it: none, basket, or hand."
    )
    rationale: str = Field(description="Short reasoning. Keep it brief.")


SYSTEM_INSTRUCTION = """You are a robot instruction parser.

Given a speech transcription, extract:
- pick_target: the item/fruit/object the user wants the robot to pick up
- place_target: where to put it, must be exactly one of: "none", "basket", "hand"

Rules:
- If user says to return/put/place in a basket/bin/container, choose "basket".
- If user says to hand/give/pass to a person / into their hand / palm, choose "hand".
- If no destination is mentioned, choose "none".
- pick_target should be a short noun phrase, lowercase, no punctuation.
Return JSON that matches the schema exactly.
"""


class AudioProcessingNode(Node):
    def __init__(self):
        super().__init__("audio_processing_node")

        # ---- Params ----
        self.declare_parameter("state_topic", "/state_machine")
        self.declare_parameter("record_service", "/microphone/record")
        self.declare_parameter("record_duration_sec", 10.0)

        self.declare_parameter("is_recording_topic", "/is_recording")
        self.declare_parameter("pick_topic", "/pick_target")
        self.declare_parameter("place_topic", "/place_target")

        self.declare_parameter("speech_language_code", "en-US")
        # For short clips, sync recognize() is fine. If you later do longer audio, use long_running_recognize.
        self.declare_parameter("speech_model", "latest_short")

        self.declare_parameter("gemini_model", "gemini-2.5-flash")  # adjust as needed

        self.declare_parameter("test_audio_path", "")

        self.test_audio_path = self.get_parameter("test_audio_path").value

        self.state_topic = self.get_parameter("state_topic").value
        self.record_service = self.get_parameter("record_service").value
        self.record_duration = float(self.get_parameter("record_duration_sec").value)

        self.is_recording_topic = self.get_parameter("is_recording_topic").value
        self.pick_topic = self.get_parameter("pick_topic").value
        self.place_topic = self.get_parameter("place_topic").value

        self.speech_language_code = self.get_parameter("speech_language_code").value
        self.speech_model = self.get_parameter("speech_model").value
        self.gemini_model = self.get_parameter("gemini_model").value

        # ---- ROS I/O ----
        self.state_sub = self.create_subscription(String, self.state_topic, self.on_state, 10)
        self.is_recording_pub = self.create_publisher(Bool, self.is_recording_topic, 10)
        self.pick_pub = self.create_publisher(String, self.pick_topic, 10)
        self.place_pub = self.create_publisher(String, self.place_topic, 10)

        # ---- Service client ----
        if not self.test_audio_path:
            self.mic_client = self.create_client(AudioRecord, self.record_service)
            self.get_logger().info(f"Waiting for {self.record_service} ...")
            if not self.mic_client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Service {self.record_service} not available")
            self.get_logger().info("Microphone service connected.")

        # ---- Google clients ----
        self.speech_client = speech.SpeechClient()
        self.gemini_client = genai.Client()  # GEMINI_API_KEY picked up automatically :contentReference[oaicite:3]{index=3}

        # ---- Pipeline guard ----
        self._lock = threading.Lock()
        self._running = False

        self.get_logger().info(
            f"Listening on {self.state_topic}. Trigger state='audio_processing'. "
            f"Will record {self.record_duration:.1f}s."
        )

    # -------- ROS callbacks --------
    def on_state(self, msg: String):
        state = (msg.data or "").strip()
        if state != "audio_processing":
            return

        with self._lock:
            if self._running:
                self.get_logger().warn("Audio pipeline already running; ignoring trigger.")
                return
            self._running = True

        self.get_logger().info("Trigger received: audio_processing. Starting pipeline thread...")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

        # -------- WAV loading (for test_audio_path) --------
    def _load_wav(self, path: str):
        """
        Load a WAV file and return (float_samples_list, sample_rate).

        Supports:
          - 16-bit PCM (int16) mono/stereo
          - 32-bit float WAV mono/stereo

        Stereo is downmixed to mono by averaging channels.
        """
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()   # bytes per sample
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            # int16 PCM
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            # Could be float32 WAV or int32 PCM. Most "float WAV" encoders store float32.
            # We'll assume float32; if it's int32 PCM, values will be huge and we can detect.
            audio = np.frombuffer(raw, dtype=np.float32)
            # If it's actually int32 PCM misread as float32, numbers will be absurd; clamp later.
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes (need 2 or 4).")

        # Reshape for channels and downmix if needed
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        # Ensure in [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)

        return audio.tolist(), int(sample_rate)

    # -------- Pipeline --------
    def _run_pipeline(self):
        try:
            if self.test_audio_path:
                self.get_logger().info(f"Using test audio: {self.test_audio_path}")
                audio_data, sample_rate = self._load_wav(self.test_audio_path)
            else:
                audio_resp = self._record_for_duration(self.record_duration)
                audio_data = audio_resp.audio_data
                sample_rate = audio_resp.sample_rate

            transcript = self._transcribe_google(audio_data, sample_rate)
            parsed = self._parse_with_gemini(transcript)
            self._publish_targets(parsed.pick_target, parsed.place_target)

        except Exception as e:
            self.get_logger().error(f"Pipeline failed: {e}")
        finally:
            with self._lock:
                self._running = False

    # -------- Recording --------
    def _call_mic_service(self, start: bool) -> AudioRecord.Response:
        req = AudioRecord.Request()
        req.start = start
        future = self.mic_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None:
            raise RuntimeError("Microphone service call failed (timeout or error).")
        return future.result()

    def _set_is_recording(self, val: bool):
        msg = Bool()
        msg.data = val
        self.is_recording_pub.publish(msg)

    def _record_for_duration(self, seconds: float) -> AudioRecord.Response:
        # Start recording
        start_resp = self._call_mic_service(True)
        if not start_resp.success:
            raise RuntimeError(f"Start recording failed: {start_resp.message}")

        self._set_is_recording(True)
        self.get_logger().info(f"Recording for {seconds:.1f}s...")
        time.sleep(seconds)

        # Stop recording
        stop_resp = self._call_mic_service(False)
        self._set_is_recording(False)

        if not stop_resp.success:
            raise RuntimeError(f"Stop recording failed: {stop_resp.message}")

        self.get_logger().info(
            f"Recorded: {stop_resp.duration:.2f}s @ {stop_resp.sample_rate} Hz, "
            f"samples={len(stop_resp.audio_data)}"
        )
        return stop_resp

    # -------- Transcription (Google Cloud STT) --------
    def _transcribe_google(self, float_samples: list, sample_rate: int) -> str:
        if not float_samples:
            raise RuntimeError("No audio samples received from microphone service.")

        # Convert float32 [-1,1] -> int16 PCM (LINEAR16)
        audio = np.array(float_samples, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767).astype(np.int16).tobytes()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=int(sample_rate),
            language_code=self.speech_language_code,
            model=self.speech_model,
        )
        audio_req = speech.RecognitionAudio(content=pcm16)

        resp = self.speech_client.recognize(config=config, audio=audio_req)
        if not resp.results:
            return ""

        # Take the top alternative of the first result (simple default)
        alt = resp.results[0].alternatives[0]
        return (alt.transcript or "").strip()

    # -------- LLM parsing (Gemini structured output) --------
    def _parse_with_gemini(self, transcript: str) -> LLMParse:
        prompt = f'Transcription:\n"""{transcript}"""\n\nExtract pick_target and place_target.'

        resp = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "response_mime_type": "application/json",
                "response_json_schema": LLMParse.model_json_schema(),
            },
        )

        # resp.text should be valid JSON matching the schema :contentReference[oaicite:4]{index=4}
        parsed = LLMParse.model_validate_json(resp.text)

        # Basic cleanup
        parsed.pick_target = parsed.pick_target.strip().lower()
        return parsed

    # -------- Publish outputs --------
    def _publish_targets(self, pick_target: str, place_target: str):
        pick_msg = String()
        pick_msg.data = pick_target
        self.get_logger().info(f"Publishing pick_target='{pick_target}', place_target='{place_target}'")
        self.pick_pub.publish(pick_msg)

        place_msg = String()
        place_msg.data = place_target
        self.place_pub.publish(place_msg)


def main():
    rclpy.init()
    node = AudioProcessingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
