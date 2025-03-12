import base64
import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union


class FileManager:
    def __init__(self, tmp_dir: str = "/tmp/video_dedupe"):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()

    def copy_file_to_local(self, source_path: Union[str, Path]) -> Optional[Path]:
        """Copies a file to the local /tmp directory."""
        source_path = Path(source_path)
        destination_path = self.tmp_dir / source_path.name

        if len(str(destination_path)) > 255:
            destination_path = self.tmp_dir / self.generate_temp_filename(source_path.suffix)

        try:
            shutil.copy2(source_path, destination_path)
            return destination_path
        except Exception as e:
            print(f"Error copying file: {e}")
            return None

    def cleanup_tmp_files(self, files: List[Optional[str]]) -> None:
        """Removes temporary files from the system."""
        for file in files:
            if file and os.path.exists(file):
                os.remove(file)

    def generate_temp_filename(self, extension: str = "") -> str:
        """Generates a random Base64-encoded filename."""
        random_bytes = os.urandom(6)
        base64_name = base64.urlsafe_b64encode(random_bytes).decode("utf-8").rstrip("=")
        return f"{base64_name}{extension}"

    def get_sha256_hash(self, file_path: str) -> str:
        """Computes SHA256 hash for a given file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_scan_state(self, scan_state_file: str) -> Dict[str, Dict[str, Optional[str]]]:
        """Loads scan state from a file."""
        if os.path.exists(scan_state_file):
            with open(scan_state_file, "r") as f:
                return json.load(f)
        return {}

    def save_scan_state(self, scan_state_file: str, data: Dict[str, Dict[str, Optional[str]]]) -> None:
        """Saves scan state to a file with a thread lock."""
        with self.lock:
            with open(scan_state_file, "w") as f:
                json.dump(data, f, indent=4)