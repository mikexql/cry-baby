import pathlib
import queue
import threading
import wave
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
from hexalog.ports import Logger
from huggingface_hub.file_download import uuid

from cry_baby.app.core.ports import Recorder


@dataclass
class PyaudioRecordingSettings:
    number_of_audio_signals: int  # 1 for mono, 2 for stereo
    frames_per_buffer: int  # e.g. 1024
    recording_rate_hz: int  # e.g. 44100
    duration_seconds: float  # e.g. 4

    def __post_init__(self):
        if self.number_of_audio_signals not in [1, 2]:
            raise ValueError("number_of_audio_signals must be 1 or 2")


class PyaudioRecorder(Recorder):
    def __init__(
        self,
        temp_path: pathlib.Path,
        logger: Logger,
        settings: PyaudioRecordingSettings,
    ):
        self.temp_path = temp_path
        self.logger = logger
        self.settings = settings
        self.temp_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    def record(self) -> pathlib.Path:
        file_path = self.temp_path / f"{uuid.uuid4()}.wav"
        self._record_and_save(file_path)
        return file_path

    def continuously_record(self) -> Optional[queue.Queue]:
        audio_recorded_queue = queue.Queue()
        recording_thread = threading.Thread(
            target=self._record_continuous,
            args=(audio_recorded_queue,),
        )
        self.logger.debug("Starting recording thread")
        recording_thread.daemon = True
        recording_thread.start()
        self.logger.debug("Recording thread started")
        return audio_recorded_queue

    def _record_and_save(self, file_path: pathlib.Path):
        self.logger.debug(
            "Begin recording audio",
            duration=self.settings.duration_seconds,
            recording_rate_hz=self.settings.recording_rate_hz,
            frames_per_buffer=self.settings.frames_per_buffer,
        )
        frames = []
        for _ in range(
            0,
            int(
                self.settings.recording_rate_hz
                / self.settings.frames_per_buffer
                * self.settings.duration_seconds
            ),
        ):
            frame = sd.rec(
                self.settings.frames_per_buffer,
                samplerate=self.settings.recording_rate_hz,
                channels=self.settings.number_of_audio_signals,
                dtype='int16'
            )
            sd.wait()
            frames.append(frame)
        self._write_to_file(file_path, frames)

    def _record_continuous(self, audio_recorded_queue: queue.Queue):
        while True:
            self.logger.debug("Starting to record continuously")
            frames = []
            for _ in range(
                0,
                int(
                    self.settings.recording_rate_hz
                    / self.settings.frames_per_buffer
                    * self.settings.duration_seconds
                ),
            ):
                frame = sd.rec(
                    self.settings.frames_per_buffer,
                    samplerate=self.settings.recording_rate_hz,
                    channels=self.settings.number_of_audio_signals,
                    dtype='int16'
                )
                sd.wait()
                frames.append(frame)
            file_path = self.temp_path / f"{uuid.uuid4()}.wav"
            self._write_to_file(file_path, frames)
            audio_recorded_queue.put(file_path)

    def _write_to_file(self, file_path: pathlib.Path, frames: List[np.ndarray]):
        self.temp_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        wave_file = wave.open(str(file_path), "wb")
        wave_file.setnchannels(self.settings.number_of_audio_signals)
        wave_file.setsampwidth(2)  # 2 bytes for int16
        wave_file.setframerate(self.settings.recording_rate_hz)
        wave_file.writeframes(b"".join([frame.tobytes() for frame in frames]))
        wave_file.close()
        self.logger.debug("Written to file", file_path=file_path)
