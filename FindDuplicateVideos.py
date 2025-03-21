#!/usr/bin/env python3

import base64
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import audioread
    import imagehash
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing module {e.name}.")
    print("Please install required dependencies using:")
    print("    pip install -r requirements.txt")
    sys.exit(1)

from src.logger import ColoredLogger

SCAN_STATE_FILE = "scan_state.json"


class VideoDuplicateFinder:
    def __init__(
            self, scan_dir: str, scan_state: str, duplicates_file: str,
            workers: Optional[int] = None, verbose: bool = True
    ) -> None:
        """Initializes the VideoDuplicateFinder class.

        Args:
            scan_dir (str): Directory to scan for videos.
            scan_state (str): Path to the scan state file.
            duplicates_file (str): Path to save duplicates.
            workers (Optional[int]): Number of workers (default: CPU count / 2).
            verbose (bool): Enable verbose logging.
        """
        self.scan_dir: Path = Path(scan_dir)
        self.scan_state_file: Path = Path(scan_state)
        self.duplicates_file: Path = Path(duplicates_file)
        self.tmp_dir: Path = Path("/tmp/video_dedupe")
        self.tmp_dir.mkdir(exist_ok=True)
        self.processed_files: Dict[str, Dict[str, Optional[str]]] = self.load_scan_state()
        self.hash_index: Dict[str, List[str]] = {}
        self.lock: threading.Lock = threading.Lock()
        self.verbose: bool = verbose
        self.logger: ColoredLogger = ColoredLogger(logging.DEBUG) if verbose else ColoredLogger(logging.INFO)
        self.workers: int = workers or max(2, multiprocessing.cpu_count() // 2)

    def log(self, level: str, message: str) -> None:
        """Logs a message with the specified log level."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level.upper()

        if level not in level_map:
            self.logger.error(f"Invalid log level: {level}. Defaulting to INFO.")
            log_level = logging.INFO
        else:
            log_level = level_map[level]

        if self.verbose:
            self.logger.log(log_level, message)

    def load_scan_state(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Loads the scan state from a file.

        Returns:
            Dict[str, Dict[str, Optional[str]]]: The scan state data.
        """
        if self.scan_state_file.exists():
            with open(self.scan_state_file, "r") as f:
                return json.load(f)
        return {}

    def save_scan_state(self) -> None:
        """Safely saves the scan state to a file with a thread lock."""
        with self.lock:
            with open(self.scan_state_file, "w") as f:
                json.dump(self.processed_files, f, indent=4)

    @staticmethod
    def cleanup_tmp_files(files: List[Optional[str]]) -> None:
        """Removes temporary files from the system.

        Args:
            files (List[Optional[str]]): List of file paths to delete.
        """
        for file in files:
            if file and os.path.exists(file):
                os.remove(file)

    @staticmethod
    def get_video_duration(file_path: Union[str, Path]) -> int:
        """Gets the duration of a video file.

        Args:
            file_path (Union[str, Path]): Path to the video file.

        Returns:
            int: Duration in seconds (default: 60 if unknown).
        """
        command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return int(float(result.stdout.strip()))
        except ValueError:
            return 60

    @staticmethod
    def generate_temp_filename(extension: str = "") -> str:
        """
        Generates a random Base64-encoded string as a temporary filename.
        Optionally appends a file extension if provided.

        Args:
            extension (str): The file extension (e.g., '.mp4', '.wav').

        Returns:
            str: A unique filename with the given extension.
        """
        random_bytes = random.getrandbits(48).to_bytes(6, 'big')
        base64_name = base64.urlsafe_b64encode(random_bytes).decode("utf-8").rstrip("=")
        return f"{base64_name}{extension}"

    def copy_file_to_local(self, source_path: Union[str, Path]) -> Optional[Path]:
        """Copies a file to the local /tmp directory."""
        source_path = Path(source_path)
        destination_path = self.tmp_dir / source_path.name

        if len(str(destination_path)) > 255:
            self.log("WARNING", f"Filename too long for {destination_path}. Generating temporary name.")
            destination_path = self.tmp_dir / self.generate_temp_filename(source_path.suffix)

        try:
            shutil.copy2(source_path, destination_path)
            self.log("INFO", f"Copied {source_path} to {destination_path}")

            # ✅ Ensure the file is fully written before continuing
            while not destination_path.exists():
                time.sleep(1)  # Give it a second to complete

            return destination_path
        except Exception as e:
            self.log("ERROR", f"Failed to copy {source_path} to {destination_path}: {e}")
            return None

    def normalize_video(self, file_path: Union[Path, str]) -> Optional[str]:
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

        if len(temp_video) > 255:
            self.log("WARNING", f"Filename too long for {temp_video}. Generating temporary name.")
            temp_video = os.path.join(self.tmp_dir, self.generate_temp_filename(".mp4"))

        command: List[str] = [
            "ffmpeg", "-i", str(file_path), "-vf", "scale=320:180",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            str(temp_video)
        ]

        try:
            result: Optional[str] = self.run_ffmpeg(command, temp_video)

            if result is None:
                self.log("ERROR", f"FFmpeg error while normalizing {file_path}")
                return None

            self.log("INFO", f"Normalized video created at {temp_video}")
            return temp_video

        except Exception as e:
            self.log("ERROR", f"Error normalizing video {file_path}: {e}")
            return None

    def extract_audio(self, file_path: str) -> Optional[str]:
        """Extracts audio from a video file and saves it as a low-quality WAV file.

        Args:
            file_path (str): The path to the video file.

        Returns:
            Optional[str]: The path to the extracted audio file, or None if an error occurred.
        """
        if not os.path.isfile(file_path):
            self.log("ERROR", f"File does not exist: {file_path}")
            return None

        temp_audio = os.path.join(self.tmp_dir, os.path.basename(file_path))

        if len(temp_audio) > 255:
            self.log("WARNING", f"Filename too long for {temp_audio}. Generating temporary name.")
            temp_audio = os.path.join(self.tmp_dir, self.generate_temp_filename(".wav"))

        command: List[str] = [
            "ffmpeg", "-i", str(file_path),
            "-c:a", "pcm_s16le", "-ar", "8000", "-ac", "1", str(temp_audio)
        ]

        try:
            result: Optional[str] = self.run_ffmpeg(command, temp_audio)

            if result is None:
                self.log("ERROR", f"FFmpeg error while extracting audio from {file_path}")
                return None

            self.log("INFO", f"Audio extracted to {temp_audio}")
            return temp_audio

        except Exception as e:
            self.log("ERROR", f"Error extracting audio from {file_path}: {e}")
            return None

    def get_audio_hash(self, audio_path: str) -> str:
        """Computes SHA256 hash from audio file using audioread.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: The computed SHA256 hash of the audio fingerprint.
        """
        hasher = hashlib.sha256()
        try:
            with audioread.audio_open(audio_path) as f:
                for buf in f:
                    hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            self.log("ERROR", f"Failed to compute audio hash: {e}")
            return ""

    def run_ffmpeg(self, command: List[str], output_file: str) -> Optional[str]:
        """Executes an FFmpeg command and returns the output file path if successful."""
        self.log("DEBUG", f"Running FFmpeg command: {' '.join(command)}")

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        last_size: int = 0
        last_update_time: float = time.time()

        while process.poll() is None:
            time.sleep(10)

            if os.path.exists(output_file):
                current_size: int = os.path.getsize(output_file)
                self.log("DEBUG", f"FFmpeg output file size: {current_size} bytes")

                if current_size > last_size:
                    last_size = current_size
                    last_update_time = time.time()
                elif time.time() - last_update_time > 60:  # Reduce timeout for debugging
                    self.log("ERROR", f"FFmpeg process stalled. Killing it.")
                    process.kill()
                    return None

        if process.returncode == 0 and os.path.exists(output_file):
            self.log("INFO", f"FFmpeg successfully created {output_file}")
            return output_file
        else:
            self.log("ERROR", f"FFmpeg failed. Return code: {process.returncode}")
            return None

    @staticmethod
    def get_sha256_hash(file_path: str) -> str:
        """Computes a SHA256 hash for a given file.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: The computed SHA256 hash.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_perceptual_hash(self, video_path: str) -> Optional[str]:
        """Generates perceptual hashes for a video file using FFmpeg.

        Args:
            video_path (str): Path to the video file.

        Returns:
            Optional[str]]: A list of perceptual hashes and the generated hash file path.
        """
        if not video_path or not os.path.isfile(video_path):
            self.log("ERROR", f"File does not exist: {video_path}")
            return None

        keyframes_dir = f"{video_path}_frames"

        if len(keyframes_dir) > 255:
            self.log(
                "WARNING",
                f"Keyframes directory name too long for {keyframes_dir}. Generating temporary name."
            )
            keyframes_dir = os.path.join(os.path.dirname(video_path), self.generate_temp_filename())

        command: List[str] = [
            "ffmpeg", "-i", str(video_path), "-vf", "select=eq(pict_type\\,I)",
            "-vsync", "vfr", "-q:v", "1", f"{keyframes_dir}/frame-%03d.jpg"
        ]

        try:
            self.log("DEBUG", f"Generating perceptual hash for {video_path}")
            os.makedirs(keyframes_dir, exist_ok=True)

            result: Optional[str] = self.run_ffmpeg(command, f"{keyframes_dir}/frame-001.jpg")

            if result is None or not os.listdir(keyframes_dir):
                self.log("ERROR", f"Failed to generate perceptual hash for {video_path}")
                return None  # Return empty list and None for file path

            frame_files = os.listdir(keyframes_dir)
            frame_hashes = []
            for frame in frame_files:
                try:
                    frame_path = os.path.join(keyframes_dir, frame)
                    frame_hash = str(imagehash.average_hash(Image.open(frame_path)))
                    frame_hashes.append(frame_hash)
                except Exception as e:
                    self.log("WARNING", f"Error processing frame {frame}: {e}")

            if not frame_hashes:
                self.log("ERROR", f"No valid hashes generated for {video_path}.")
                return None

            combined_hash = hashlib.sha256("".join(frame_hashes).encode()).hexdigest()
            self.log("INFO", f"Perceptual hash generated for {video_path}")
            return combined_hash

        except Exception as e:
            self.log("ERROR", f"Unexpected error generating perceptual hash for {video_path}: {e}")
            return None

        finally:
            # Ensure cleanup of keyframes directory
            if os.path.exists(keyframes_dir):
                for frame in os.listdir(keyframes_dir):
                    try:
                        os.remove(os.path.join(keyframes_dir, frame))
                    except Exception as e:
                        self.log("WARNING", f"Failed to remove frame {frame}: {e}")
                try:
                    os.rmdir(keyframes_dir)
                    self.log("DEBUG", f"Temporary directory {keyframes_dir} removed.")
                except Exception as e:
                    self.log("WARNING", f"Failed to remove temporary directory {keyframes_dir}: {e}")

    def process_video(self, file_path: Union[Path, str]) -> None:
        """Processes a video file by copying, normalizing, hashing, and extracting keyframes.

        Args:
            file_path (Union[Path, str]): Path to the video file.
        """
        if str(file_path).lower() in self.processed_files:
            return

        try:
            # ✅ Step 1: Copy video to /tmp
            local_video_path = self.copy_file_to_local(file_path)
            if not local_video_path or not local_video_path.exists():
                self.log("ERROR", f"Failed to copy video to /tmp: {file_path}")
                return

            # ✅ Ensure the file is large enough (non-empty)
            if local_video_path.stat().st_size == 0:
                self.log("ERROR", f"Copied file is empty: {local_video_path}")
                return

            # ✅ Step 2: Normalize the copied video
            normalized_video = self.normalize_video(local_video_path)
            if not normalized_video:
                return

            # ✅ Step 3: Compute SHA256 hash for normalized video
            video_hash = self.get_sha256_hash(normalized_video)

            # ✅ Step 4: Extract normalized audio from normalized video
            normalized_audio = self.extract_audio(normalized_video)
            if not normalized_audio:
                return

            # ✅ Step 5: Compute SHA256 audio fingerprint hash
            audio_hash = self.get_audio_hash(normalized_audio)

            # ✅ Step 6: Generate perceptual hashes using FFmpeg
            perceptual_hash = self.generate_perceptual_hash(normalized_video)

            # ✅ Store hashes in scan state
            with self.lock:
                self.processed_files[str(file_path).lower()] = {
                    "video_hash": video_hash if video_hash else "",
                    "audio_hash": audio_hash if audio_hash else "",
                    "perceptual_hashes": perceptual_hash if perceptual_hash else ""
                }
                self.save_scan_state()

            # ✅ Cleanup temporary files (now includes `hash_file`)
            self.cleanup_tmp_files([local_video_path, normalized_video, normalized_audio])

        except Exception as e:
            self.log("ERROR", f"Error processing {file_path}: {e}")

    def scan_videos(self) -> None:
        """Scans and processes video files in the specified directory."""
        video_extensions: set = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"}
        video_files: List[Path] = [
            p for p in self.scan_dir.rglob("*")
            if p.suffix.lower() in video_extensions and str(p).lower() not in self.processed_files
        ]

        batch_size: int = 500
        for i in range(0, len(video_files), batch_size):
            batch = video_files[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures: set = set()
                with tqdm(total=len(batch)) as pbar:
                    for file in batch:
                        # Limit concurrent futures to the max workers
                        if len(futures) >= self.workers:
                            done, _ = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            futures.difference_update(done)  # Safe removal

                        futures.add(executor.submit(self.process_video, file))

                    # Ensure all remaining futures complete before exiting
                    if futures:
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                future.result()
                                pbar.update(1)  # Update only on success
                            except Exception as e:
                                self.log("ERROR", f"Error processing video: {e}")

            executor.shutdown(wait=True)  # Ensure all tasks finish before exit

        self.log("INFO", "Scan completed.")
        self.save_scan_state()


if __name__ == "__main__":
    scan_directory = input("Where to start scanning: ")
    scan_state_file = input("Where to save the scan save file: ")
    duplicates_save_file = input("Where to save the duplicates file: ")
    finder = VideoDuplicateFinder(scan_directory, scan_state_file, duplicates_save_file)
    finder.scan_videos()
