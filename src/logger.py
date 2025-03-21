import logging
import sys
from typing import Dict, Union

from tqdm import tqdm

class ColoredFormatter(logging.Formatter):
    """Custom Formatter to add colors to log levels for CLI output.

    Attributes:
        COLORS (Dict[int, str]): A dictionary mapping log levels to ANSI color codes.
        RESET (str): ANSI reset code to reset color formatting after each message.
    """

    COLORS: Dict[int, str] = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with appropriate color coding.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with ANSI color codes.
        """
        color = self.COLORS.get(record.levelno, self.RESET)
        log_msg = super().format(record)
        return f"{color}{log_msg}{self.RESET}"


class ColoredLogger:
    """Singleton Logger with color-coded output for CLI applications.

    This class ensures that only one logger instance is created, and it formats
    messages with ANSI colors based on log levels.

    Attributes:
        _instance (ColoredLogger): Singleton instance of the logger.
        logger (logging.Logger): The underlying logger instance.
    """

    _instance = None
    _log_level = logging.INFO

    def __new__(cls, level: int = logging.INFO) -> "ColoredLogger":
        """Creates a single instance of the logger with a specified log level.

        Args:
            level (int): The logging level (default: logging.INFO).

        Returns:
            ColoredLogger: The singleton instance of the logger.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(level)
        else:
            cls._instance.set_level(level)
        return cls._instance

    def _initialize(self, level: int) -> None:
        """Initializes the logger with a console handler and colored formatter."""
        self.logger = logging.getLogger("ColoredLogger")
        self.set_level(level)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_level(self, level: int) -> None:
        """Sets the logging level dynamically."""
        self._log_level = level
        self.logger.setLevel(level)

    def log(self, level: Union[str, int], message: str) -> str:
        """Logs a message at the specified level using tqdm.write().

        Args:
            level (Union[str, int]): The log level as a string (e.g., "DEBUG", "INFO") or an integer.
            message (str): The message to log.

        Returns:
            str: The formatted log message.
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        # Convert integer levels back to their string representation
        if isinstance(level, int):
            level = next((k for k, v in level_map.items() if v == level), "INFO")
        elif isinstance(level, str):
            level = level.upper()
        else:
            level = "INFO"

        if level not in level_map:
            tqdm.write(f"\033[93m[WARNING] Invalid log level: {level}. Defaulting to INFO.\033[0m")
            log_level = logging.INFO
        else:
            log_level = level_map[level]

        formatted_message = f"[{level}] {message}"
        tqdm.write(formatted_message)  # ✅ Ensures log messages don't interfere with progress bars
        # self.logger.log(log_level, message)
        return formatted_message

    def debug(self, message: str) -> None:
        """Logs a debug message."""
        tqdm.write(f"\033[94m[DEBUG]\033[0m {message}")  # Blue
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Logs an info message."""
        tqdm.write(f"\033[92m[INFO]\033[0m {message}")  # Green
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Logs a warning message."""
        tqdm.write(f"\033[93m[WARNING]\033[0m {message}")  # Yellow
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Logs an error message."""
        tqdm.write(f"\033[91m[ERROR]\033[0m {message}")  # Red
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Logs a critical message."""
        tqdm.write(f"\033[95m[CRITICAL]\033[0m {message}")  # Magenta
        self.logger.critical(message)
