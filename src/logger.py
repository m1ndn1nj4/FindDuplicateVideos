import logging
import sys
from typing import Dict


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

    def __new__(cls) -> "ColoredLogger":
        """Creates a single instance of the logger.

        Returns:
            ColoredLogger: The singleton instance of the logger.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initializes the logger with a console handler and colored formatter."""
        self.logger = logging.getLogger("ColoredLogger")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = ColoredFormatter("[%(levelname)s] %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message: str) -> None:
        """Logs a debug message.

        Args:
            message (str): The message to log.
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Logs an info message.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Logs a warning message.

        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Logs an error message.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Logs a critical message.

        Args:
            message (str): The message to log.
        """
        self.logger.critical(message)
