import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union

import audioread
import imagehash
from PIL import Image


class VideoProcessing:
    def __init__(self, tmp_dir: str = "/tmp/video_dedupe"):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)

    def get_video_duration(self, file_path: Union[str, Path]) -> int:
        """Gets the duration of a video file."""
        command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return int(float(result.stdout.strip()))
        except ValueError:
            return 60

    def normalize_video(self, file_path: Union[Path, str]) -> Optional[str]:
        """Normalizes a video file."""
        temp_video = self.tmp_dir / Path(file_path).name

        if len(str(temp_video)) > 255:
            temp_video = self.tmp_dir / self.generate_temp_filename(".mp4")

        command = ["ffmpeg", "-i", str(file_path), "-vf", "scale=320:180",
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35", str(temp_video)]

        return self.run_ffmpeg(command, str(temp_video))

    def extract_audio(self, file_path: str) -> Optional[str]:
        """Extracts audio from a video file."""
        temp_audio = self.tmp_dir / Path(file_path).name

        if len(str(temp_audio)) > 255:
            temp_audio = self.tmp_dir / self.generate_temp_filename(".wav")

        command = ["ffmpeg", "-i", str(file_path),
                   "-c:a", "pcm_s16le", "-ar", "8000", "-ac", "1", str(temp_audio)]

        return self.run_ffmpeg(command, str(temp_audio))

    def get_audio_hash(self, audio_path: str) -> str:
        """Computes SHA256 hash from audio."""
        hasher = hashlib.sha256()
        try:
            with audioread.audio_open(audio_path) as f:
                for buf in f:
                    hasher.update(buf)
            return hasher.hexdigest()
        except Exception:
            return ""

    def generate_perceptual_hash(self, video_path: str) -> Optional[str]:
        """Generates perceptual hashes for a video file."""
        keyframes_dir = f"{video_path}_frames"
        os.makedirs(keyframes_dir, exist_ok=True)

        command = ["ffmpeg", "-i", str(video_path), "-vf", "select=eq(pict_type\\,I)",
                   "-vsync", "vfr", "-q:v", "1", f"{keyframes_dir}/frame-%03d.jpg"]

        result = self.run_ffmpeg(command, f"{keyframes_dir}/frame-001.jpg")

        if result is None or not os.listdir(keyframes_dir):
            return None

        frame_files = os.listdir(keyframes_dir)
        frame_hashes = [str(imagehash.average_hash(Image.open(os.path.join(keyframes_dir, frame))))
                        for frame in frame_files]

        return hashlib.sha256("".join(frame_hashes).encode()).hexdigest() if frame_hashes else None

    def run_ffmpeg(self, command: List[str], output_file: str) -> Optional[str]:
        """Executes an FFmpeg command."""
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        last_size = 0
        last_update_time = time.time()

        while process.poll() is None:
            time.sleep(10)
            if os.path.exists(output_file):
                current_size = os.path.getsize(output_file)
                if current_size > last_size:
                    last_size = current_size
                    last_update_time = time.time()
                elif time.time() - last_update_time > 300:
                    process.kill()
                    return None

        return output_file if process.returncode == 0 and os.path.exists(output_file) else None