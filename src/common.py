import base64
import hashlib
import json
import logging
import math
import os
import platform
import random
import signal
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Optional, Callable, Any, Tuple, Dict, List

import audioread
import imagehash
import psutil
from PIL import Image
from tqdm import tqdm

# ---------------------- Setup and Utilities ----------------------

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class LogColors:
    DEBUG: str = "\033[34m"
    INFO: str = "\033[32m"
    WARNING: str = "\033[33m"
    ERROR: str = "\033[31m"
    RESET: str = "\033[0m"


def log_debug(message: str) -> None:
    tqdm.write(f"{LogColors.DEBUG}[DEBUG]{LogColors.RESET} {message}")


def log_info(message: str) -> None:
    tqdm.write(f"{LogColors.INFO}[INFO]{LogColors.RESET} {message}")


def log_warning(message: str) -> None:
    tqdm.write(f"{LogColors.WARNING}[WARNING]{LogColors.RESET} {message}")


def log_error(message: str) -> None:
    tqdm.write(f"{LogColors.ERROR}[ERROR]{LogColors.RESET} {message}")


class TemporaryFileManager:
    def __init__(self):
        self.temp_files: List[str] = []

    def register(self, file_path: str) -> None:
        if file_path and os.path.exists(file_path):
            self.temp_files.append(file_path)
            log_debug(f"Temporary file registered: {file_path}")
        else:
            log_warning(f"Attempted to register non-existent or missing file: {file_path}")

    def cleanup(self) -> None:
        log_info("Starting cleanup of temporary files.")
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    log_debug(f"Temporary file removed: {file_path}")
                else:
                    log_warning(f"Temporary file already removed or missing: {file_path}")
            except Exception as e:
                log_error(f"Failed to remove temporary file {file_path}: {e}")
        self.temp_files.clear()
        log_info("Cleanup of temporary files completed.")


# ---------------------- Error Tracking for Decoding Failures ----------------------

DECODE_ERRORS_FILE = "decode_errors.json"


def log_decode_error(file_path: str, error_message: str) -> None:
    """Log decoding errors to a JSON file."""
    try:
        decode_errors_path = os.path.join(os.path.dirname(CHECKPOINT_FILE), DECODE_ERRORS_FILE)
        if not os.path.exists(decode_errors_path):
            with open(decode_errors_path, "w") as f:
                json.dump([], f)  # Initialize the file with an empty list

        # Load existing errors
        with open(decode_errors_path, "r") as f:
            errors = json.load(f)

        # Append the new error
        errors.append({"file": file_path, "error": error_message})

        # Save the updated errors back to the file
        with open(decode_errors_path, "w") as f:
            json.dump(errors, f, indent=4)

        log_warning(f"Logged decode error for {file_path}: {error_message}")
    except Exception as e:
        log_error(f"Failed to log decode error for {file_path}: {e}")


# ---------------------- File and Video Validation ----------------------

def get_video_properties(file_path: str) -> Optional[Tuple[float, int, int]]:
    # Validate that the file path is non-empty and the file exists
    if not file_path or not os.path.isfile(file_path):
        log_error(f"Invalid file path or file does not exist: {file_path}")
        return None

    try:
        # Run FFprobe to fetch metadata for the first video stream
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration,width,height",
                "-of", "json", file_path
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

        # Check if FFprobe execution was successful
        if result.returncode != 0:
            log_error(f"FFprobe failed for {file_path}: {result.stderr.strip()}")
            return None

        # Parse the JSON output from FFprobe
        metadata = json.loads(result.stdout)

        # Check if any video streams are present in the metadata
        if "streams" not in metadata or not metadata["streams"]:
            log_warning(f"No video stream found in metadata for {file_path}.")
            return None

        # Extract duration, width, and height from the metadata
        video_stream = metadata["streams"][0]
        duration = float(video_stream.get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Validate the extracted properties
        if duration <= 0 or width <= 0 or height <= 0:
            log_warning(f"Invalid video properties for {file_path}")
            return None

        # Return the extracted properties
        return duration, width, height

    except Exception as e:
        # Log an error message if an exception occurs
        log_error(f"Error retrieving video properties for {file_path}: {e}")
        return None


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


def monitor_file_growth(file_path: str, interval: int = 60, timeout: int = 600) -> bool:
    """
    Monitors the growth of a file's size at a specified interval over a timeout period.
    Returns True if the file grows, False if no growth is detected within the timeout.
    """
    if not os.path.exists(file_path):
        log_error(f"File does not exist for monitoring: {file_path}")
        return False

    last_size = os.path.getsize(file_path)
    start_time = time.time()

    while time.time() - start_time < timeout:
        time.sleep(interval)
        current_size = os.path.getsize(file_path)

        if current_size > last_size:
            last_size = current_size
            start_time = time.time()  # Reset the timer on growth
        elif time.time() - start_time >= timeout:
            log_warning(f"File processing stalled for {file_path} after timeout.")
            return False

    return True  # Growth completed successfully or no timeout reached


def process_with_monitoring(
    process_function: Callable[..., Optional[str]],
    monitor_file_path: str,
    monitor_interval: int = 60,
    monitor_timeout: int = 600,
    **kwargs
) -> Optional[str]:
    """
    Runs a process with file size monitoring to detect stalls and handle timeouts.
    """
    monitor_event = threading.Event()

    def monitor():
        if not monitor_file_growth(monitor_file_path, interval=monitor_interval, timeout=monitor_timeout):
            log_warning(f"Timeout reached for monitoring {monitor_file_path}.")
            monitor_event.set()

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    try:
        result = process_function(**kwargs)
        if result is None:
            log_warning(f"Processing function returned None for {monitor_file_path}.")
        monitor_event.set()  # Stop monitoring once processing completes
        monitor_thread.join()
        return result
    except Exception as e:
        log_error(f"Error during processing {monitor_file_path}: {e}")
        monitor_event.set()
        monitor_thread.join()
        return None


def is_network_path(path: str) -> bool:
    # Validate that the path exists; otherwise, return False immediately
    if not os.path.exists(path):
        return False  # Path does not exist, so cannot determine network status

    # macOS/Linux-specific logic
    if platform.system() in ["Darwin", "Linux"]:
        try:
            # Retrieve device information using os.stat
            path_device = os.stat(path).st_dev
            parent_device = os.stat(os.path.dirname(path)).st_dev

            # If devices differ, it's likely a separate mount point
            if path_device != parent_device:
                # Determine the mount point type by parsing the mount configuration file
                with open(
                    "/proc/mounts" if platform.system() == "Linux" else "/etc/mtab"
                ) as f:
                    mounts = f.readlines()
                    for mount in mounts:
                        # Check if the path matches a known network-based file system
                        if path in mount and any(
                            fs_type in mount for fs_type in ["nfs", "smbfs", "cifs"]
                        ):
                            return True  # Path is network-based
            return False  # Path is local or removable media
        except Exception as e:
            # Fallback: Check for network path heuristics (e.g., smb://)
            return path.startswith("//") or path.startswith("smb://")

    # Windows-specific logic
    elif platform.system() == "Windows":
        # Check for UNC paths
        if path.startswith("\\\\"):  # UNC paths start with "\\"
            return True
        try:
            # Use ctypes to query the drive type for the path
            import ctypes
            drive = os.path.splitdrive(path)[0]  # Extract the drive letter
            if drive:
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\")
                # DRIVE_REMOTE (4) indicates a network drive
                return drive_type == 4
        except Exception:
            pass

    # Default to False for paths that do not match any criteria
    return False


def copy_file_to_local(
    file_path: str, local_dir: str = "/tmp", chunk_size: int = 1024 * 1024
) -> Optional[str]:
    # Construct the full path for the destination file in the local directory
    local_path = os.path.join(local_dir, os.path.basename(file_path))

    try:
        # Log the start of the copy operation
        log_info(f"Copying {file_path} to local directory {local_dir}")

        # Open the source file in binary mode for reading and the destination
        # file in binary mode for writing
        with open(file_path, "rb") as src, open(local_path, "wb") as dst:
            while chunk := src.read(chunk_size):  # Read file in chunks
                dst.write(chunk)  # Write each chunk to the destination file

        # Register the copied file with TemporaryFileManager
        temp_file_manager = TemporaryFileManager()
        temp_file_manager.register(local_path)

        # Return the destination path upon successful completion
        return local_path
    except Exception as e:
        # Log an error message if any exception occurs during the copy operation
        log_error(f"Failed to copy {file_path} to local directory: {e}")
        return None


def calculate_average_file_size(file_paths: List[str]) -> float:
    # Check if the file list is empty; log a warning and return 0.0 if true
    if not file_paths:
        log_warning("No files provided to calculate the average size.")
        return 0.0

    try:
        # Calculate the total size of all valid files in bytes
        total_size = sum(
            os.path.getsize(file) for file in file_paths if os.path.isfile(file)
        )

        # If no valid files are found, log a warning and return 0.0
        valid_files = sum(1 for file in file_paths if os.path.isfile(file))
        if valid_files == 0:
            log_warning("No valid files provided to calculate the average size.")
            return 0.0

        # Calculate the average file size in bytes and convert to gigabytes
        average_size = total_size / valid_files
        return average_size / (1024 ** 3)  # Convert bytes to GB

    except Exception as e:
        # Log an error if an exception occurs and return 0.0
        log_error(f"Error calculating average file size: {e}")
        return 0.0


def get_video_quality(video_path: str) -> Optional[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,bit_rate,codec_name", "-of", "json", video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            log_error(f"ffprobe failed for {video_path}: {result.stderr}")
            return None

        metadata = json.loads(result.stdout)
        stream = metadata.get("streams", [{}])[0]

        return {
            "path": video_path,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "bitrate": int(stream.get("bit_rate", 0)),
            "codec": stream.get("codec_name"),
        }
    except Exception as e:
        log_error(f"Error retrieving metadata for {video_path}: {e}")
        return None


def select_best_quality(duplicate_group: List[str]) -> Dict[str, Any]:
    quality_data = []
    for video in duplicate_group:
        metadata = get_video_quality(video)
        if metadata:
            quality_data.append(metadata)

    # Sort by resolution (width x height) and then bitrate
    quality_data.sort(
        key=lambda x: (x.get("width", 0) * x.get("height", 0), x.get("bitrate", 0)),
        reverse=True
    )

    if not quality_data:
        log_warning(f"No valid metadata for duplicates: {duplicate_group}")
        return {"original_file": None, "duplicate_files": duplicate_group}

    # The first item is the best-quality file
    original_file = quality_data[0]["path"]
    duplicate_files = [x["path"] for x in quality_data[1:]]

    return {"original_file": original_file, "duplicate_files": duplicate_files}


# ---------------------- Signal and Pause Management ----------------------

pause_event = threading.Event()
pause_event.set()  # Allow execution initially
executor_lock = threading.Lock()
executor: Optional[ThreadPoolExecutor] = None
current_max_workers: Optional[int] = None


def signal_pause(signal_number: int, frame: Optional[Any]) -> None:
    if pause_event.is_set():
        # Pause execution by clearing the event
        pause_event.clear()
        log_warning("Script paused. Send the signal again to resume.")
    else:
        # Resume execution by setting the event
        pause_event.set()
        log_info("Script resumed.")


# Register the signal handler for SIGUSR1
signal.signal(signal.SIGUSR1, signal_pause)


def paused_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    # Continuously check if the script is paused
    while not pause_event.is_set():
        log_debug("Task is paused. Waiting to resume...")
        time.sleep(1)  # Wait for 1 second before checking again

    # Execute the task function once the script is resumed
    return func(*args, **kwargs)


# ---------------------- Resource Management ----------------------

def calculate_dynamic_max_workers() -> int:
    try:
        cpu_count = os.cpu_count()  # Get the number of available CPU cores
        memory_info = psutil.virtual_memory()  # Get system memory information
        total_memory_gb = memory_info.total / (1024 ** 3)  # Convert to GB
        memory_per_worker_gb = 0.5  # Assume 0.5 GB per worker

        # Calculate max workers based on CPU and memory limits
        max_workers = min(cpu_count * 2, int(total_memory_gb / memory_per_worker_gb))

        # Ensure at least one worker is returned
        return max(1, max_workers)
    except Exception as e:
        log_error(f"Error calculating dynamic max workers: {e}")
        return 1  # Fallback to 1 worker in case of an error


def calculate_max_network_workers(
    total_bandwidth_gbps: float = 2.5,
    file_size_gb: float = 0.5,
    safety_factor: float = 0.8
) -> int:
    try:
        # Validate inputs
        if total_bandwidth_gbps <= 0:
            log_error("Invalid bandwidth provided. Must be greater than 0.")
            return 1
        if file_size_gb <= 0:
            log_error("Invalid file size provided. Must be greater than 0.")
            return 1

        # Convert bandwidth to GBps and apply safety factor
        bandwidth_gb_per_second = (total_bandwidth_gbps / 8) * safety_factor
        log_debug(f"Effective Bandwidth (GBps after safety factor): {bandwidth_gb_per_second}")

        # Calculate max workers
        max_workers = math.ceil(bandwidth_gb_per_second / file_size_gb)
        log_debug(f"Calculated max workers before adjustment: {max_workers}")

        # Ensure at least one worker is returned
        return max(1, max_workers)
    except Exception as e:
        log_error(f"Error calculating max network workers: {e}")
        return 1  # Fallback to 1 worker in case of an error


def dynamic_worker_adjustment(is_network: bool, interval: int = 10) -> None:
    global executor, current_max_workers
    while True:
        # Sleep for the specified interval before recalculating workers
        time.sleep(interval)
        new_max_workers = (
            calculate_max_network_workers() if is_network else calculate_dynamic_max_workers()
        )

        # Update executor if the number of workers has changed
        if new_max_workers != current_max_workers:
            log_info(f"Adjusting workers from {current_max_workers} to {new_max_workers}")
            with executor_lock:
                executor.shutdown(wait=False)
                executor = ThreadPoolExecutor(max_workers=new_max_workers)
                current_max_workers = new_max_workers


# ---------------------- Graceful Shutdown Checkpoint ----------------------

CHECKPOINT_FILE = "processing_checkpoint.json"


def save_checkpoint(
    processed_data: Dict[str, Dict[str, str]], checkpoint_file: str
) -> bool:
    try:
        # Open the checkpoint file in write mode and save the processed data
        with open(checkpoint_file, "w") as f:
            json.dump(processed_data, f, indent=4)  # Serialize with indentation
        return True
    except Exception as e:
        # Log an error message if the save operation fails
        log_error(f"Failed to save checkpoint: {e}")
        return False


def load_checkpoint(checkpoint_file: str) -> Dict[str, Dict[str, str]]:
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_file):
        return {}  # Return an empty dictionary if the file does not exist

    try:
        # Open the checkpoint file in read mode and load the JSON data
        with open(checkpoint_file, "r") as f:
            return json.load(f)  # Deserialize the JSON data into a dictionary
    except Exception as e:
        # Log an error message if the load operation fails
        log_error(f"Failed to load checkpoint: {e}")
        return {}

# ---------------------- Video Processing and Duplicates ----------------------


def normalize_video(
        file_path: str,
        output_dir: str = "/tmp",
        resolution: Tuple[int, int] = (160, 120)
) -> Optional[str]:
    if not file_path or not os.path.isfile(file_path):
        log_error(f"File does not exist: {file_path}")
        return None

    normalized_name = f"normalized_{os.path.basename(file_path)}"
    normalized_path = os.path.join(output_dir, normalized_name)

    # Check if the output file path is too long
    if len(normalized_path) > 255:  # Typical max path length in most OSes
        log_warning(f"Output file name too long for {normalized_path}. Generating temporary name.")
        temp_name = generate_temp_filename(extension=".mp4")
        normalized_path = os.path.join(output_dir, temp_name)

    try:
        log_debug(f"Normalizing video: {file_path} to {resolution[0]}x{resolution[1]} with output {normalized_path}")

        result = subprocess.run(
            [
                "ffmpeg", "-i", file_path,
                "-vf", f"scale={resolution[0]}:{resolution[1]}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
                normalized_path
            ],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            log_error(f"FFmpeg error while normalizing {file_path}: {result.stderr}")
            return None

        log_info(f"Normalized video created at {normalized_path}")
        return normalized_path

    except Exception as e:
        log_error(f"Error normalizing video {file_path}: {e}")
        return None


def extract_audio_to_wav(video_path: str, output_path: str, original_file: str) -> Optional[str]:
    """
    Extract audio from a video file and save it as a low-quality WAV file.
    Handles long output file names by generating a temporary name if necessary.
    """
    if not video_path or not os.path.isfile(video_path):
        log_error(f"File does not exist: {video_path}")
        return None

    # Check if the output file path is too long
    if len(output_path) > 255:  # Typical max path length in most OSes
        log_warning(f"Output file name too long for {output_path}. Generating temporary name.")
        extension = os.path.splitext(output_path)[1] or ".wav"
        temp_name = generate_temp_filename(extension)
        output_path = os.path.join(os.path.dirname(output_path), temp_name)

    try:
        log_debug(f"Extracting audio from {video_path} to {output_path} as WAV")

        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-c:a", "pcm_s16le",  # Use 16-bit PCM for WAV files
                "-ar", "8000",        # Set sample rate to 8 kHz (low quality)
                "-ac", "1",           # Mono audio
                output_path
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            log_error(f"FFmpeg error while extracting audio from {video_path}: {result.stderr}")
            return None

        log_info(f"Audio extracted to {output_path}")
        return output_path

    except Exception as e:
        log_error(f"Error extracting audio from {video_path}: {e}")
        return None


def get_audio_properties(audio_path: str) -> dict:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=bit_rate,sample_rate,duration",
                "-of", "json",
                audio_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running ffprobe: {result.stderr}")
            return {}

        return json.loads(result.stdout).get("streams", [{}])[0]
    except Exception as e:
        print(f"Error analyzing audio file: {e}")
        return {}


def generate_audio_fingerprint(audio_path: str, buffer_limit: int = None) -> Optional[str]:
    if not audio_path or not os.path.isfile(audio_path):
        log_error("Invalid audio path provided. Cannot generate fingerprint.")
        return None

    try:
        log_debug(f"Generating audio fingerprint for {audio_path}")
        hash_obj = hashlib.sha256()
        buffer_count = 0

        with audioread.audio_open(audio_path) as audio:
            for buffer in audio:
                if not buffer:
                    continue

                hash_obj.update(buffer)  # Update hash incrementally
                buffer_count += 1

                if buffer_limit and buffer_count > buffer_limit:
                    log_error(f"Exceeded maximum buffer count for {audio_path}. Aborting.")
                    return None

        fingerprint = hash_obj.hexdigest()
        log_info(f"Audio fingerprint generated for {audio_path}")
        return fingerprint

    except audioread.DecodeError:
        log_error(f"Error decoding audio file: {audio_path}. Ensure the file is a valid format.")
        return None
    except Exception as e:
        log_error(f"Unexpected error generating audio fingerprint for {audio_path}: {e}")
        return None


def generate_perceptual_hash(file_path: str) -> Optional[str]:
    if not file_path or not os.path.isfile(file_path):
        log_error(f"File does not exist: {file_path}")
        return None

    keyframes_dir = f"{file_path}_frames"

    # Check if the keyframes directory path is too long
    if len(keyframes_dir) > 255:
        log_warning(f"Keyframes directory name too long for {keyframes_dir}. Generating temporary name.")
        temp_name = generate_temp_filename()
        keyframes_dir = os.path.join(os.path.dirname(file_path), temp_name)

    try:
        log_debug(f"Generating perceptual hash for {file_path}")
        os.makedirs(keyframes_dir, exist_ok=True)

        result = subprocess.run(
            [
                "ffmpeg", "-i", file_path, "-vf", "select=eq(pict_type\\,I)",
                "-vsync", "vfr", "-q:v", "1", f"{keyframes_dir}/frame-%03d.jpg"
            ],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            log_error(f"FFmpeg error while extracting keyframes from {file_path}: {result.stderr}")
            return None

        frame_files = os.listdir(keyframes_dir)
        if not frame_files:
            log_error(f"No keyframes extracted from {file_path}.")
            return None

        frame_hashes = []
        for frame in frame_files:
            try:
                frame_path = os.path.join(keyframes_dir, frame)
                frame_hash = str(imagehash.average_hash(Image.open(frame_path)))
                frame_hashes.append(frame_hash)
            except Exception as e:
                log_warning(f"Error processing frame {frame}: {e}")

        if not frame_hashes:
            log_error(f"No valid hashes generated for {file_path}.")
            return None

        combined_hash = hashlib.sha256("".join(frame_hashes).encode()).hexdigest()
        log_info(f"Perceptual hash generated for {file_path}")
        return combined_hash

    except Exception as e:
        log_error(f"Unexpected error generating perceptual hash for {file_path}: {e}")
        return None

    finally:
        if os.path.exists(keyframes_dir):
            for frame in os.listdir(keyframes_dir):
                try:
                    os.remove(os.path.join(keyframes_dir, frame))
                except Exception as e:
                    log_warning(f"Failed to remove frame {frame}: {e}")
            try:
                os.rmdir(keyframes_dir)
                log_debug(f"Temporary directory {keyframes_dir} removed.")
            except Exception as e:
                log_warning(f"Failed to remove temporary directory {keyframes_dir}: {e}")


def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> Optional[str]:
    # Validate input parameters
    if not file_path or not isinstance(file_path, str):
        log_error("Invalid file path provided. Cannot calculate file hash.")
        return None  # Fail fast if the file path is invalid.

    if not os.path.exists(file_path):
        log_error(f"File does not exist: {file_path}")
        return None  # Fail fast if the file does not exist.

    if not os.path.isfile(file_path):
        log_error(f"Path is not a file: {file_path}")
        return None  # Fail fast if the path is not a regular file.

    # Initialize SHA-256 hash object
    sha256 = hashlib.sha256()

    try:
        log_debug(f"Calculating file hash for {file_path}")
        # Read the file in chunks to avoid memory issues with large files
        with open(file_path, "rb") as file:
            while chunk := file.read(chunk_size):
                sha256.update(chunk)

        # Return the calculated hash as a hexadecimal string
        file_hash = sha256.hexdigest()
        log_info(f"File hash calculated for {file_path}: {file_hash}")
        return file_hash

    except PermissionError:
        log_error(f"Permission denied: Unable to read file {file_path}")
        return None  # Fail fast if there is a permission error

    except (OSError, IOError) as e:
        log_error(f"Error reading file {file_path}: {e}")
        return None  # Fail fast on file I/O errors

    except Exception as e:
        log_error(f"Unexpected error calculating hash for {file_path}: {e}")
        return None  # Fail fast on unexpected exceptions


def process_video_file(file_path: str) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """
    Processes a video file by normalizing it, extracting audio, generating hashes,
    and handling cases where file or directory names are too long.
    """
    temp_manager = TemporaryFileManager()
    file_mapping = {}  # To track original filename and temporary name mappings

    try:
        # Step 1: Normalize video
        log_info(f"Processing {file_path}")
        normalized_name = f"normalized_{os.path.basename(file_path)}"
        normalized_path = os.path.join("/tmp", normalized_name)

        # Handle long filenames for normalization output
        if len(normalized_path) > 255:
            log_warning(f"Output file name too long for {normalized_path}. Generating temporary name.")
            original_extension = os.path.splitext(file_path)[1]  # Extract original extension
            temp_name = generate_temp_filename(extension=original_extension)
            normalized_path = os.path.join("/tmp", temp_name)
            file_mapping[temp_name] = os.path.basename(file_path)

        # Use `process_with_monitoring` to normalize the video
        normalized_file = process_with_monitoring(
            process_function=normalize_video,
            monitor_file_path=normalized_path,
            file_path=file_path,
            output_dir=os.path.dirname(normalized_path)
        )
        if not normalized_file:
            error_message = "Normalization failed"
            log_decode_error(file_path, error_message)
            log_error(f"{error_message} for {file_path}. Skipping processing.")
            return file_path, {}
        temp_manager.register(normalized_file)

        # Step 2: Generate file hash
        hash_results = {"file_hash": calculate_file_hash(normalized_file)}
        if not hash_results["file_hash"]:
            error_message = "File hash calculation failed"
            log_decode_error(file_path, error_message)
            log_error(f"{error_message} for {file_path}. Skipping processing.")
            return file_path, {}

        # Step 3: Extract audio and generate audio fingerprint
        audio_output_path = f"{normalized_file}.wav"

        # Handle long filenames for audio extraction
        if len(audio_output_path) > 255:
            log_warning(f"Output file name too long for {audio_output_path}. Generating temporary name.")
            temp_name = generate_temp_filename(extension=".wav")
            audio_output_path = os.path.join("/tmp", temp_name)
            file_mapping[temp_name] = os.path.basename(file_path)

        extracted_audio = process_with_monitoring(
            process_function=extract_audio_to_wav,
            monitor_file_path=audio_output_path,
            video_path=normalized_file,
            output_path=audio_output_path,
            original_file=file_path
        )
        if extracted_audio:
            temp_manager.register(extracted_audio)
            audio_fingerprint = generate_audio_fingerprint(extracted_audio)
            hash_results["audio_fingerprint"] = audio_fingerprint or None
        else:
            error_message = "Audio extraction failed"
            log_decode_error(file_path, error_message)
            log_error(f"{error_message} for {file_path}.")

        # Step 4: Generate perceptual hash
        perceptual_hash = process_with_monitoring(
            process_function=generate_perceptual_hash,
            monitor_file_path=normalized_file,
            file_path=normalized_file
        )
        if not perceptual_hash:
            error_message = "Perceptual hash generation failed"
            log_decode_error(file_path, error_message)
            log_error(f"{error_message} for {file_path}.")
        hash_results["perceptual_hash"] = perceptual_hash or None

        # Log mappings of temporary names to original file names
        if file_mapping:
            log_info(f"File mappings for {file_path}: {file_mapping}")

        return file_path, hash_results

    finally:
        temp_manager.cleanup()


def find_duplicate_videos(directory: str, save_interval: int, checkpoint_file: str) -> List[List[str]]:
    """
    Identify duplicate video files within a specified directory.

    This function scans a directory for video files, computes hashes using
    multiple methods for each file, and identifies duplicates by comparing
    their hash values. It supports resuming partially completed operations
    using a checkpoint file and periodically saves progress to avoid data loss
    during long-running operations.

    Args:
        directory (str): Path to the directory containing video files.
        save_interval (int): Number of files processed before saving progress.
        checkpoint_file (str): Path to the JSON file for saving progress.

    Returns:
        List[List[str]]: Groups of duplicate video file paths.
    """
    if not directory or not isinstance(directory, str):
        log_error("Invalid directory provided. Cannot scan for duplicate videos.")
        return []

    if not os.path.isdir(directory):
        log_error(f"Directory does not exist: {directory}")
        return []

    video_extensions = {".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}
    hashes = defaultdict(list)

    # Load checkpoint to resume processing
    processed_data = load_checkpoint(checkpoint_file)
    processed_files = set(processed_data.keys())
    counter = 0

    # Collect all video files
    try:
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if os.path.splitext(file)[1].lower() in video_extensions
        ]
    except Exception as e:
        log_error(f"Error traversing directory {directory}: {e}")
        return []

    total_files = len(all_files)
    unprocessed_files = [file for file in all_files if file not in processed_files]
    total_unprocessed = len(unprocessed_files)

    if total_files == 0:
        log_warning("No video files found in the directory.")
        return []

    log_info(f"Total files: {total_files}")
    log_info(f"Already processed files: {len(processed_files)}")
    log_info(f"Remaining files to process: {total_unprocessed}")

    # Initialize tqdm for progress tracking
    with executor_lock:
        try:
            with tqdm(
                    total=total_unprocessed,
                    desc="Processing Videos",
                    unit="file",
                    dynamic_ncols=True
            ) as progress_bar:
                futures = {
                    executor.submit(
                        process_with_monitoring,
                        process_function=process_video_file,
                        monitor_file_path=file_path,
                        file_path=file_path
                    ): file_path
                    for file_path in unprocessed_files
                }

                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        file_path, hash_results = future.result()
                        if not hash_results:
                            log_warning(f"No hash results for {file_path}. Skipping...")
                            continue

                        # Store hashes
                        processed_data[file_path] = hash_results
                        for hash_type, hash_value in hash_results.items():
                            if not hash_value:
                                log_warning(f"Empty hash value for {file_path}, skipping this hash type.")
                                continue
                            hashes[(hash_type, hash_value)].append(file_path)

                        processed_files.add(file_path)
                        counter += 1

                        if counter % save_interval == 0:
                            save_checkpoint(processed_data, checkpoint_file)
                            log_info(f"Checkpoint saved after processing {counter} files.")

                    except TimeoutError:
                        log_error(f"Task for {file_path} exceeded the time limit.")
                    except Exception as e:
                        log_error(f"Task failed for {file_path}: {e}")
                    finally:
                        progress_bar.update(1)
        except Exception as e:
            log_error(f"Error during video processing: {e}")
            return []

    duplicates = [files for files in hashes.values() if len(files) > 1]
    save_checkpoint(processed_data, checkpoint_file)
    log_info(f"Final checkpoint saved after processing all files.")

    if not duplicates:
        log_info("No duplicate video files found.")
        return []

    log_info(f"Found {len(duplicates)} groups of duplicate videos.")
    return duplicates


def write_duplicates_to_file(duplicates: List[List[str]], output_file: str) -> bool:
    if not duplicates:
        log_warning("No duplicates to write to the file.")
        return False

    if not output_file or not isinstance(output_file, str):
        log_error("Invalid output file path provided. Cannot write duplicates.")
        return False

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_info(f"Created missing directory for output file: {output_dir}")
    except Exception as e:
        log_error(f"Failed to create directory for output file: {e}")
        return False

    try:
        log_info(f"Writing duplicates to file: {output_file}")
        output_data = []
        for group in duplicates:
            best_quality = select_best_quality(group)
            output_data.append(best_quality)

        with open(output_file, "w") as file:
            json.dump(output_data, file, indent=4)

        log_info(f"Successfully wrote duplicates to {output_file}")
        return True
    except Exception as e:
        log_error(f"Error writing duplicates to file: {e}")
        return False


# ---------------------- Main Entry Point ----------------------

def main() -> None:
    try:
        directory = input(
            "Enter the directory to scan for duplicate videos (default: './videos'): "
        ).strip() or "./videos"
        checkpoint_dir = input(
            "Enter the directory to save the checkpoint file (default: './checkpoints'): "
        ).strip() or "./checkpoints"
        checkpoint_file = input(
            "Enter the checkpoint file name (without extension, default: 'processing_checkpoint'): "
        ).strip() or "processing_checkpoint"
        output_file = input(
            "Enter the path to save duplicates.txt (default: './duplicates.txt'): "
        ).strip() or "./duplicates.txt"

        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_file}.json")
        decode_errors_path = os.path.join(checkpoint_dir, DECODE_ERRORS_FILE)
        video_extensions = {".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}

        # Collect all files in the directory
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if os.path.splitext(file)[1].lower() in video_extensions
        ]
        log_debug(f"Collected {len(all_files)} files: {all_files[:10]}")  # Log first 10 files as a preview

        if not all_files:
            log_error("No video files found in the specified directory.")
            return

        # Calculate average file size
        average_file_size = calculate_average_file_size(all_files)
        log_info(f"Average file size: {average_file_size:.2f} GB")

        is_network = is_network_path(directory)
        safety_factor = 2.5
        global executor, current_max_workers
        current_max_workers = (
            calculate_max_network_workers(total_bandwidth_gbps=2.5, file_size_gb=average_file_size,
                                          safety_factor=safety_factor)
            if is_network
            else calculate_dynamic_max_workers()
        )
        log_debug(f"Current max workers: {current_max_workers}")
        executor = ThreadPoolExecutor(max_workers=current_max_workers)

        # Process videos and find duplicates
        duplicates = find_duplicate_videos(directory, save_interval=2, checkpoint_file=checkpoint_path)
        write_duplicates_to_file(duplicates, output_file)

        log_info(f"Decode errors (if any) saved to {decode_errors_path}")

    except Exception as e:
        log_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()