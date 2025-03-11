import json
import multiprocessing
import os
import subprocess
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import xxhash
from PIL import Image
from imagehash import phash
from tqdm import tqdm

from src.logger import ColoredLogger

SCAN_STATE_FILE = "scan_state.json"


class VideoDuplicateFinder:
    def __init__(self, scan_dir, scan_state, duplicates_file, workers=None, verbose=False):
        self.scan_dir = Path(scan_dir)
        self.scan_state_file = Path(scan_state)
        self.duplicates_file = Path(duplicates_file)
        self.tmp_dir = Path("/tmp/video_dedupe")
        self.tmp_dir.mkdir(exist_ok=True)
        self.processed_files = self.load_scan_state()
        self.hash_index = {}
        self.lock = threading.Lock()
        self.verbose = True
        self.logger = ColoredLogger(logging.DEBUG) if verbose else ColoredLogger(logging.INFO)
        self.workers = workers or max(2, multiprocessing.cpu_count() // 2)

    def log(self, level, message):
        if self.verbose:
            self.logger.log(level, message)

    def load_scan_state(self):
        if self.scan_state_file.exists():
            with open(self.scan_state_file, "r") as f:
                return json.load(f)
        return {}

    def save_scan_state(self):
        with open(self.scan_state_file, "w") as f:
            json.dump(self.processed_files, f, indent=4)

    @staticmethod
    def get_video_duration(file_path):
        command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return int(float(result.stdout.strip()))
        except ValueError:
            return 60  # Default to 60 seconds if duration cannot be determined

    def normalize_video(self, file_path: str) -> Optional[str]:
        """Normalizes a video file by resizing it to 320x180 and encoding it in H.264.

        Args:
            file_path (str): The path to the video file to normalize.

        Returns:
            Optional[str]: The path to the normalized video file, or None if an error occurred.
        """
        if not os.path.exists(file_path):
            self.log("WARNING", f"File does not exist: {file_path}")
            return None

        temp_video = os.path.join(self.tmp_dir, os.path.basename(file_path))

        # Check if filename is too long (Unix systems usually have a 255-char limit)
        if len(temp_video) > 255:
            self.log("WARNING", f"Filename too long for {temp_video}. Generating temporary name.")
            temp_video = os.path.join(self.tmp_dir, "temp_normalized.mp4")

        command = [
            "ffmpeg", "-i", str(file_path), "-vf", "scale=320:180",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            str(temp_video)
        ]

        try:
            result = subprocess.run(
                command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                self.log("ERROR", f"FFmpeg error while normalizing {file_path}: {result.stderr}")
                return None

            self.log("INFO", f"Normalized video created at {temp_video}")
            return temp_video

        except Exception as e:
            self.log("ERROR", f"Error normalizing video {file_path}: {e}")
            return None

    def extract_audio(self, file_path: str) -> Optional[str]:
        """
        Extract audio from a video file and save it as a low-quality WAV file.
        Handles long output file names by generating a temporary name if necessary.
        """
        if not file_path or not os.path.isfile(file_path):
            self.log("ERROR", f"File does not exist: {file_path}")
            return None

        temp_audio = os.path.join(self.tmp_dir, os.path.basename(file_path))

        # Check if filename is too long (Unix systems usually have a 255-char limit)
        if len(temp_audio) > 255:
            self.log("WARNING", f"Filename too long for {temp_audio}. Generating temporary name.")
            temp_audio = os.path.join(self.tmp_dir, "temp_normalized.wav")

        command = [
            "ffmpeg", "-i", str(file_path),
            "-c:a", "pcm_s16le", "-ar", "8000", "-ac", "1", str(temp_audio)
        ]

        try:
            result = subprocess.run(
                command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                self.log("ERROR", f"FFmpeg error while extracting audio from {file_path}: {result.stderr}")
                return None

            self.log("INFO", f"Audio extracted to {temp_audio}")
            return temp_audio

        except Exception as e:
            self.log("ERROR", f"Error extracting audio from {file_path}: {e}")
            return None

    def extract_image(self, video_path):
        image_path = self.tmp_dir / (video_path.stem + ".jpg")
        command = ["ffmpeg", "-i", str(video_path), "-frames:v", "1", "-y", str(image_path)]
        return self.run_ffmpeg(command, image_path)

    def run_ffmpeg(self, command, output_file):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        last_size = 0
        last_update_time = time.time()

        while process.poll() is None:
            time.sleep(10)
            if output_file.exists():
                current_size = output_file.stat().st_size
                if current_size > last_size:
                    last_size = current_size
                    last_update_time = time.time()
                elif time.time() - last_update_time > 300:
                    self.log("ERROR", f"Process stalled, killing ffmpeg and removing {output_file}.")
                    process.kill()
                    if output_file.exists():
                        output_file.unlink()  # **Delete the stalled file**
                    return None

        return output_file if output_file.exists() else None

    @staticmethod
    def get_hash(file_path):
        hasher = xxhash.xxh64()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def get_image_hash(image_path):
        return str(phash(Image.open(image_path)))

    def process_video(self, file_path):
        if str(file_path).lower() in self.processed_files:
            return

        try:
            normalized_video = self.normalize_video(file_path)
            if not normalized_video:
                return

            audio_file = self.extract_audio(normalized_video)
            audio_hash = self.get_hash(audio_file) if audio_file else None
            image_file = self.extract_image(normalized_video)
            image_hash = self.get_image_hash(image_file) if image_file else None
            video_hash = self.get_hash(normalized_video)

            self.processed_files[str(file_path).lower()] = {
                "audio_hash": audio_hash,
                "image_hash": image_hash,
                "video_hash": video_hash
            }
            self.save_scan_state()

            self.cleanup_tmp_files([normalized_video, audio_file, image_file])
        except Exception as e:
            self.log("ERROR", f"Error processing {file_path}: {e}")

    @staticmethod
    def cleanup_tmp_files(files):
        for file in files:
            if file and file.exists():
                file.unlink()

    def scan_videos(self):
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"}
        video_files = [p for p in self.scan_dir.rglob("*")
                       if p.suffix.lower() in video_extensions and str(p).lower() not in self.processed_files]

        batch_size = 500
        for i in range(0, len(video_files), batch_size):
            batch = video_files[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.process_video, file): file for file in batch}
                with tqdm(total=len(batch)) as pbar:
                    for future in as_completed(futures):
                        future.result()  # Ensure exceptions are raised if any occur
                        pbar.update(1)

        self.log("INFO", "Scan completed.")
        self.save_scan_state()


if __name__ == "__main__":
    scan_directory = input("Where to start scanning: ")
    scan_state_file = input("Where to save the scan save file: ")
    duplicates_save_file = input("Where to save the duplicates file: ")
    finder = VideoDuplicateFinder(scan_directory, scan_state_file, duplicates_save_file)
    finder.scan_videos()
