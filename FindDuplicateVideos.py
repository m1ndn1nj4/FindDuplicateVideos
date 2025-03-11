import json
import multiprocessing
import os
import subprocess
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import xxhash
from PIL import Image
from imagehash import phash
from tqdm import tqdm

from src.logger import ColoredLogger

SCAN_STATE_FILE = "scan_state.json"


class VideoDuplicateFinder:
    def __init__(
        self, scan_dir: str, scan_state: str, duplicates_file: str,
        workers: Optional[int] = None, verbose: bool = False
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
        if self.verbose:
            self.logger.log(level, message)

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
        """Saves the scan state to a file."""
        with open(self.scan_state_file, "w") as f:
            json.dump(self.processed_files, f, indent=4)

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

        temp_video: str = os.path.join(self.tmp_dir, os.path.basename(file_path))

        if len(temp_video) > 255:
            self.log("WARNING", f"Filename too long for {temp_video}. Generating temporary name.")
            temp_video = os.path.join(self.tmp_dir, "temp_normalized.mp4")

        command: List[str] = [
            "ffmpeg", "-i", str(file_path), "-vf", "scale=320:180",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            str(temp_video)
        ]

        try:
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

            if result.returncode != 0:
                self.log("ERROR", f"FFmpeg error while normalizing {file_path}: {result.stderr}")
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

        temp_audio: str = os.path.join(self.tmp_dir, os.path.basename(file_path))

        if len(temp_audio) > 255:
            self.log("WARNING", f"Filename too long for {temp_audio}. Generating temporary name.")
            temp_audio = os.path.join(self.tmp_dir, "temp_normalized.wav")

        command: List[str] = [
            "ffmpeg", "-i", str(file_path),
            "-c:a", "pcm_s16le", "-ar", "8000", "-ac", "1", str(temp_audio)
        ]

        try:
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

            if result.returncode != 0:
                self.log("ERROR", f"FFmpeg error while extracting audio from {file_path}: {result.stderr}")
                return None

            self.log("INFO", f"Audio extracted to {temp_audio}")
            return temp_audio

        except Exception as e:
            self.log("ERROR", f"Error extracting audio from {file_path}: {e}")
            return None

    def extract_image(self, video_path: str) -> Optional[str]:
        """Extracts a single frame from a video and saves it as an image."""
        image_path: str = os.path.join(self.tmp_dir, f"{Path(video_path).stem}.jpg")
        command: List[str] = ["ffmpeg", "-i", video_path, "-frames:v", "1", "-y", image_path]
        return self.run_ffmpeg(command, image_path)

    def run_ffmpeg(self, command: List[str], output_file: str) -> Optional[str]:
        """Executes an FFmpeg command and monitors for stalls.

        Args:
            command (List[str]): The FFmpeg command to execute.
            output_file (str): The expected output file.

        Returns:
            Optional[str]: The output file path if successful, otherwise None.
        """
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        last_size: int = 0
        last_update_time: float = time.time()

        while process.poll() is None:
            time.sleep(10)
            if os.path.exists(output_file):
                current_size: int = os.path.getsize(output_file)
                if current_size > last_size:
                    last_size = current_size
                    last_update_time = time.time()
                elif time.time() - last_update_time > 300:
                    self.log("ERROR", f"Process stalled. Killing FFmpeg and removing {output_file}.")
                    process.kill()
                    os.remove(output_file) if os.path.exists(output_file) else None
                    return None

        return output_file if os.path.exists(output_file) else None

    @staticmethod
    def get_hash(file_path: Union[str, Path]) -> str:
        """Computes a hash for a given file.

        Args:
            file_path (Union[str, Path]): Path to the file.

        Returns:
            str: The computed hash.
        """
        hasher = xxhash.xxh64()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def get_image_hash(image_path: Union[str, Path]) -> str:
        """Computes the perceptual hash of an image.

        Args:
            image_path (Union[str, Path]): Path to the image.

        Returns:
            str: The computed image hash.
        """
        return str(phash(Image.open(image_path)))

    def process_video(self, file_path: str) -> None:
        """Processes a video file by normalizing it, extracting audio, and computing hashes.

        Args:
            file_path (str): Path to the video file.
        """
        if str(file_path).lower() in self.processed_files:
            return

        try:
            normalized_video: Optional[str] = self.normalize_video(file_path)
            if not normalized_video:
                return

            audio_file: Optional[str] = self.extract_audio(normalized_video)
            audio_hash: Optional[str] = self.get_hash(audio_file) if audio_file else None
            image_file: Optional[str] = self.extract_image(normalized_video)
            image_hash: Optional[str] = self.get_image_hash(image_file) if image_file else None
            video_hash: Optional[str] = self.get_hash(normalized_video)

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
    def cleanup_tmp_files(files: List[Optional[str]]) -> None:
        """Removes temporary files from the system.

        Args:
            files (List[Optional[str]]): List of file paths to delete.
        """
        for file in files:
            if file and os.path.exists(file):
                os.remove(file)

    def scan_videos(self) -> None:
        """Scans and processes video files in the specified directory."""
        video_extensions: set = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"}
        video_files: List[Path] = [p for p in self.scan_dir.rglob("*")
                                   if p.suffix.lower() in video_extensions and str(p).lower() not in self.processed_files]

        batch_size: int = 500
        for i in range(0, len(video_files), batch_size):
            batch = video_files[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.process_video, file): file for file in batch}
                with tqdm(total=len(batch)) as pbar:
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)

        self.log("INFO", "Scan completed.")
        self.save_scan_state()


if __name__ == "__main__":
    scan_directory = input("Where to start scanning: ")
    scan_state_file = input("Where to save the scan save file: ")
    duplicates_save_file = input("Where to save the duplicates file: ")
    finder = VideoDuplicateFinder(scan_directory, scan_state_file, duplicates_save_file)
    finder.scan_videos()
