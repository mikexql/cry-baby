import pathlib
import queue
import threading
from typing import Optional

import hexalog.ports

from cry_baby.app.core import ports
from cry_baby.pkg.audio_file_client.core.ports import AudioFileClient


class CryBabyService(ports.Service):
    def __init__(
        self,
        logger: hexalog.ports.Logger,
        classifier: ports.Classifier,
        recorder: ports.Recorder,
        repository: ports.Repository,
        audio_file_client: AudioFileClient,
    ):
        self.logger = logger
        self.classifier = classifier
        self.recorder = recorder
        self.repository = repository
        self.audio_file_client = audio_file_client
        self.thread = None
        self._running = threading.Event()

    def evaluate_from_microphone(self) -> float:
        """
        Record audio and classify it
        return the probability that the audio contains a baby crying
        """
        self.logger.info("Service beginning to evaluate audio from microphone")
        audio_file = self.recorder.record()
        self.logger.info(f"Recorded audio file: {audio_file}")
        prediction = self.classifier.classify(audio_file)
        self.logger.info(f"Prediction: {prediction}")
        return prediction

    def continuously_evaluate_from_microphone(self) -> Optional[queue.Queue]:
        file_written_notification_queue = self.recorder.continuously_record()
        self._running.set()
        signal_thread = threading.Thread(
            target=self._handle_files_written,
            args=(file_written_notification_queue, self.classifier),
        )
        signal_thread.start()
        self.logger.info(
            "Service beginning to continuously evaluate audio from system audio"
        )
        self.thread = signal_thread
        return file_written_notification_queue

    def _handle_files_written(
        self, file_written_queue: queue.Queue, classifier: ports.Classifier
    ):
        while self._running.is_set():
            try:
                file_path: pathlib.Path = file_written_queue.get(timeout=1)
            except queue.Empty:
                continue
            if not self._running.is_set():
                break
            self.logger.info(f"Processing file: {file_path}")
            # TODO: Sampling rate, and threshold DB, should not be hardcoded
            if not self.audio_file_client.check_audio_loudness(
                file_path, threshold_db=48, sampling_rate_hz=16000
            ):
                self.logger.debug(
                    "File not loud enough, removing", audio_file_path=file_path
                )
                file_path.unlink()
                continue
            self.logger.debug(f"File written: {file_path}")
            prediction = classifier.classify(file_path)
            self.logger.debug(f"Prediction: {prediction}")
            self.repository.save(file_path, prediction)

    def stop_continuous_evaluation(self):
        self.logger.info("Service stopping continuous evaluation")
        self._running.clear()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.logger.info("Continuous evaluation stopped")
