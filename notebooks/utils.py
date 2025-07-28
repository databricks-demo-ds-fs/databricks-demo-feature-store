# Databricks notebook source
import builtins
import gc
import inspect
import json
import logging
import os
import signal
import sys
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from enum import Enum
from pprint import pprint
import typing as tp

import matplotlib.pyplot as plt
import pyspark
import requests
from dateutil.relativedelta import relativedelta
from delta.tables import DeltaTable
from IPython.display import HTML, display
from pyspark.errors import ParseException
from pyspark.sql import DataFrame, SparkSession, functions as f
from pyspark.sql.types import *
from pyspark.sql.utils import AnalysisException
from tqdm.auto import tqdm

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

# MAGIC %run /Shared/databricks-demo-feature-store/notebooks/globals

# COMMAND ----------

all_inputs = dict(dbutils.widgets.getAll())
env = all_inputs.get("env", "dev") # dev, prod, staging

if env not in ["dev", "prod", "staging"]:
    raise Exception("Invalid environment. Please select 'dev', 'prod' or 'staging'")

print(f"environment: {env}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Globals

# COMMAND ----------

no_prod_ind = int(env!="prod")

base_path = base_path_dev if no_prod_ind else base_path_prod
monitor_path = monitor_path_dev if no_prod_ind else monitor_path_prod

# COMMAND ----------

fs_database = fs_db_dev if no_prod_ind else fs_db_prod
fs_base_path = fs_base_path_dev if no_prod_ind else fs_base_path_prod
print(fs_base_path)
print(fs_database)

# COMMAND ----------

del no_prod_ind, base_path_dev, monitor_path_dev, base_path_prod, monitor_path_prod, fs_db_dev, fs_db_prod, fs_base_path_dev, fs_base_path_prod

# COMMAND ----------

source_datasets = [
    # description_name, table_name, database
    ("dim_customers", "clientes", "demo_db"),
    ("fact_products", "productos", "demo_db"),
    ("fact_credits_payment", "pagos", "demo_db"),
    ("fact_credit_risk", "buro_credito", "demo_db"),
    ("fact_liabilities_transactions", "transacciones", "demo_db"),
]

# COMMAND ----------

summary_path = f"{monitor_path}/fs_data_load/validation_summary"
detail_path = f"{monitor_path}/fs_data_load/validation_detail"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logger Custom Class

# COMMAND ----------

class LogLevel(Enum):
    """
    Enumeration for representing different log levels with associated colors.

    This enum defines standard log levels (INFO, WARNING, ERROR) and stores
    their string representation and ANSI color codes for formatted console output.

    Attributes:
        INFO (tuple): Represents the informational level. Contains level name "INFO"
                      and blue color code.
        WARNING (tuple): Represents the warning level. Contains level name "WARNING"
                         and yellow color code.
        ERROR (tuple): Represents the error level. Contains level name "ERROR"
                       and red color code.
    """
    INFO = ("INFO", '\033[34m')      # Blue
    WARNING = ("WARNING", '\033[33m')  # Yellow
    ERROR = ("ERROR", '\033[31m')      # Red

    def __init__(self, level_name: str, color_code: str):
        """
        Initializes a LogLevel enum member.

        Args:
            level_name (str): The string name of the log level (e.g., "INFO").
            color_code (str): The ANSI escape code for the color associated
                              with this log level.
        """
        self.level_name = level_name
        self.color_code = color_code
        self.reset_code = '\033[0m' # ANSI code to reset color


class ColoredFormatter(logging.Formatter):
    """
    A custom logging formatter that adds ANSI color codes to log messages.

    This formatter enhances the standard Python logging by colorizing the entire
    log line based on the log level. It uses the `LogLevel` enum to determine
    the appropriate color for each level.

    Inherits:
        logging.Formatter: Base class for log formatters.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record and applies color coding.

        Overrides the base class `format` method to retrieve the color
        associated with the record's log level from `LogLevel`. If a color
        is found, it's applied to the formatted log message.

        Args:
            record (logging.LogRecord): The log record to format.
                It contains all the information pertinent to the event being logged.

        Returns:
            str: The formatted log message string, with ANSI color codes applied
                 if a corresponding color for the log level is defined.
        """
        # Get the color for this log level from the LogLevel enum
        color = None
        reset_code = LogLevel.INFO.reset_code # Default reset code

        for level in LogLevel:
            if level.level_name == record.levelname:
                color = level.color_code
                reset_code = level.reset_code
                break

        # If color not found, use default (no color)
        if color is None:
            color = ''
            # Ensure reset code is empty if no color is applied to avoid printing reset code alone
            effective_reset_code = ''
        else:
            effective_reset_code = reset_code

        # Format the entire message using the parent class's formatter
        formatted_message = super().format(record)

        # Color the entire line
        colored_message = f"{color}{formatted_message}{effective_reset_code}"

        return colored_message


class CustomLogger:
    """
    Provides a custom logging interface with colored console output.

    This logger is designed to integrate with the standard Python `logging`
    module but enhances it by using `ColoredFormatter` to apply ANSI colors
    to log messages based on their severity level (defined in `LogLevel`).
    It ensures that logger setup is handled internally, including clearing
    any pre-existing handlers for the given logger name to prevent duplicate
    log entries.

    Attributes:
        logger_name (str): The name of the logger instance.
        _logger (logging.Logger): The underlying Python logger instance.
    """

    def __init__(self, logger_name: str):
        """
        Initializes the CustomLogger with the specified name and configures
        it for colored output using `_setup_logger`.

        Args:
            logger_name (str): The name for the logger. This name is used to
                               get or create a logger instance from the Python
                               logging module.
        """
        self.logger_name = logger_name
        self._setup_logger()

    def _setup_logger(self):
        """
        Configures the underlying logger instance.

        This method performs the initial setup for the logger:
        - Retrieves the logger instance using `self.logger_name`.
        - Clears any existing handlers to prevent duplicate log messages.
        - Sets the logger's level to `logging.INFO`.
        - Creates a `StreamHandler` to output logs to `sys.stdout`.
        - Instantiates `ColoredFormatter` for colorized output and applies it
          to the handler.
        - Adds the configured handler to the logger.
        - Disables propagation to the root logger to avoid duplicate messages
          if the root logger is also configured.
        - Stores the configured logger instance in `self._logger`.
        """
        # Clear any existing handlers for this logger
        logger_instance = logging.getLogger(self.logger_name)
        logger_instance.handlers.clear()

        # Set logger level
        logger_instance.setLevel(logging.INFO)

        # Create handler with colored formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create and set colored formatter
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger_instance.addHandler(handler)

        # Don't propagate to root logger to avoid duplicates
        logger_instance.propagate = False

        # Store the logger instance
        self._logger = logger_instance

    def log(self, level: LogLevel, message: str, **kwargs):
        """
        Logs a message with the specified log level and color.

        This method uses the underlying Python logger to output the message.
        The color of the message is determined by the `level` argument,
        as defined in the `LogLevel` enum and applied by `ColoredFormatter`.

        Args:
            level (LogLevel): The log level for the message (e.g., `LogLevel.INFO`,
                              `LogLevel.WARNING`, `LogLevel.ERROR`).
            message (str): The message string to log.

        Notes:
            If an unknown `LogLevel` is provided, it defaults to logging
            as an INFO level message, prefixed with the unknown level's name.
        """
        if level == LogLevel.INFO:
            self._logger.info(message, **kwargs)
        elif level == LogLevel.WARNING:
            self._logger.warning(message, **kwargs)
        elif level == LogLevel.ERROR:
            self._logger.error(message, **kwargs)
        else:
            # Fallback for unknown levels
            self._logger.info(
                f"{(level.level_name if isinstance(level, LogLevel) else level)}: " + \
                f"{message}"
            )

    def info(self, message: str, **kwargs):
        """
        Logs a message with the INFO level.

        Args:
            message (str): The message string to log.
        """
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """
        Logs a message with the WARNING level.

        Args:
            message (str): The message string to log.
        """
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """
        Logs a message with the ERROR level.

        Args:
            message (str): The message string to log.
        """
        self.log(LogLevel.ERROR, message, **kwargs)

# COMMAND ----------

logger = CustomLogger("Feature Store")

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing functions

# COMMAND ----------

def _detect_string_columns(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Detects string columns in a PySpark DataFrame.

    Args:
        df (DataFrame): The PySpark DataFrame to analyze.

    Returns:
        list: A list of column names that contain string data.
    """
    string_columns = []
    for field in df.schema.fields:
        # TODO: omit some types of columns like mails
        if isinstance(field.dataType, StringType):
            string_columns.append(field.name)
    return string_columns


def _clean_accents(df, columns):
    """
    Cleans accents, dieresis and virgulillas of all languages in columns of a PySpark DataFrame.
    """
    accents_mapping = {
        # Español y Portugués
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n',
        'ã': 'a', 'õ': 'o', 'ç': 'c',

        # Francés
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ë': 'e', 'ï': 'i', 'ü': 'u',

        # Alemán
        'ä': 'a', 'ö': 'o', 'ü': 'u',

        # Otros europeos
        'ø': 'o', 'å': 'a',

        # Diéresis y acentos especiales
        'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',

        # Caracteres con virgulilla y otros signos diacríticos
        'ã': 'a', 'ẽ': 'e', 'ĩ': 'i', 'õ': 'o', 'ũ': 'u',

        # Caracteres adicionales
        'æ': 'ae', 'œ': 'oe'
    }


    for accent, replacement in accents_mapping.items():
        # Construir diccionario de operaciones de reemplazo
        replace_ops = {
            column: f.regexp_replace(f.col(column), accent, replacement)
            for column in columns
        }
        df = df.withColumns(replace_ops)

    return df


def replace_no_alphanum_dataframe(
    df: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    Replaces all non-alphanumeric values ​​in a column by '_', clean accents,
    and lowercase all string values.

    It does in automatically, detecting string columns.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with alphanumeric strings separated by '_'.
    """
    string_columns = _detect_string_columns(df)
    # Characters to replace
    chars_to_replace = r"[\[\]\(\)\*\ \:\.\-\;\<\?\/\,\'\_\&]"

    # Lowercase the column values
    df = df.withColumns({col: f.lower(f.col(col)) for col in string_columns})

    # Clean accents
    df = _clean_accents(df, string_columns)

    # Trimming columns
    df = df.withColumns({col: f.trim(f.col(col)) for col in string_columns})

    # Replace all specified characters with an underscore
    df = df.withColumns({col: f.regexp_replace(f.col(col), chars_to_replace, "_") for col in string_columns})

    # Handle multiple underscores by replacing them with a single underscore
    df = df.withColumns({col: f.regexp_replace(f.col(col), "_+", "_") for col in string_columns})

    df = df.withColumns({col: f.regexp_replace(f.col(col), " ", "_") for col in string_columns})

    # Handle initial and final underscores (drop)
    df = df.withColumns({col: f.regexp_replace(f.col(col), r"^_+|_+$", "") for col in string_columns})

    return df


def _trim_column_in_dataframe(
    df: pyspark.sql.DataFrame, column_name: tp.List[str]
) -> pyspark.sql.DataFrame:
    """
    Trim whitespace from string columns in a PySpark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.
        columns_name (List[str]): A list of column names to trim.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with trimmed string columns.
    """
    df = df.withColumn(column_name, f.trim(f.col(column_name)))
    return df


def trim_columns_dataframe(
    df: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    Trim whitespace from string columns in a PySpark DataFrame.

    It does in automatically, detecting string columns.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with trimmed string columns.
    """
    string_columns = _detect_string_columns(df)

    for col in string_columns:
        logger.info(f"Trimming column: {col}")
        df = _trim_column_in_dataframe(df, column_name=col)

    return df


def preprocessing_ingesting_tables(
    df: pyspark.sql.dataframe,
    params: tp.Dict = None
) -> pyspark.sql.dataframe:
    print("Performing preprocessing, running: trim_columns_dataframe")
    df = trim_columns_dataframe(df)

    print("Performing preprocessing, running: replace_no_alphanum_dataframe")
    df = replace_no_alphanum_dataframe(df)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature functions

# COMMAND ----------

def shift_date(df: pyspark.sql.DataFrame, params):
    """
    Shifts a date column in a Spark DataFrame by a specified number of days, months, and years,
    and creates a new column with the shifted date.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame containing the date column.
        params (dict): Dictionary with the following keys:
            - new_column_name (str): Name of the new column to be created.
            - date_column (dict): Dictionary with:
                - name (str): Name of the date column to shift.
                - format (str): Format of the date column (e.g., 'yyyy-MM-dd').
            - days_to_add (int, optional): Number of days to add to the date. Defaults to 0.
            - months_to_add (int, optional): Number of months to add to the date. Defaults to 0.
            - years_to_add (int, optional): Number of years to add to the date. Defaults to 0.

    Returns:
        pyspark.sql.DataFrame: DataFrame with the new shifted date column.
    """

    if not params:
        return df

    date_column_name = params["date_column"]["name"]
    date_column_format = params["date_column"]["format"]

    new_column_name = params["new_column_name"]

    if date_column_name not in df.columns:
        raise (
            f"The date column name `{date_column_name}` especified was not found in the DataFrame columns."
        )

    days_to_add = params.get("days_to_add", 0)
    months_to_add = params.get("months_to_add", 0)
    years_to_add = params.get("years_to_add", 0)
    df = df.withColumn(
        new_column_name,
        f.date_add(
            f.to_date(f.col(date_column_name), date_column_format).cast("date"),
            days_to_add,
        ),
    )

    if days_to_add==0:
        df = df.withColumn(new_column_name, f.date_format(f.col(new_column_name), "yyyy-MM-01").cast("date"))

    df = df.withColumn(
        new_column_name,
        f.add_months(new_column_name, months_to_add + years_to_add * 12),
    )

    return df

# COMMAND ----------

def decimals_to_floats(sparkdf):
    """
    Takes a spark dataframe and casts all decimal columns to floats to avoid
    inconsistencies when dealing with aggregations and computations with floats
    """
    decimals_to_cast_ops = {
        c: sparkdf[c].cast(FloatType())
        for c, t in sparkdf.dtypes if t.startswith("decimal")
    }
    sparkdf = sparkdf.withColumns(decimals_to_cast_ops)
    return sparkdf

# COMMAND ----------

def longs_to_integers(sparkdf):
    """
    Takes a spark dataframe and casts all long columns to integers to avoid
    inconsistencies when dealing with aggregations and computations with integers
    """
    decimals_to_cast_ops = {
        field.name: f.col(field.name).cast(IntegerType())
        for field in sparkdf.schema.fields
        if isinstance(field.dataType, LongType)
    }
    for col_name, cast_op in decimals_to_cast_ops.items():
        sparkdf = sparkdf.withColumn(col_name, cast_op)
    return sparkdf


# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Store Saving

# COMMAND ----------

class InputTimeoutError(Exception):
    """
    Exception raised when a user input operation exceeds its allocated time.

    This exception is typically raised by functions or methods that require
    user interaction within a specific timeframe.

    Inherits:
        Exception: Base class for all built-in exceptions.
    """


class TimedInput:
    """
    Class that provides functionality for requesting input with a time limit.
    """

    def __init__(self):
        """
        Initializes the TimedInput class.

        This constructor attempts to determine the terminal size and the operating
        environment (Windows, Linux, or Jupyter notebook) to select the appropriate
        method for handling timed input. It sets default terminal dimensions if
        they cannot be determined.

        Attributes:
            columns (int): The width of the terminal in characters.
            rows (int): The height of the terminal in lines.
            is_windows (bool): True if the operating system is Windows.
            is_linux (bool): True if the operating system is Linux/POSIX.
            is_notebook (bool): True if running within an IPython kernel (e.g., Jupyter).
            get_input (tp.Callable[..., bool]): Method selected for handling timed input.
            timeout_occurred (bool): Flag set to True if a timeout event happens.
            original_sigalrm_handler (tp.Optional[tp.Callable]): Stores the original
                SIGALRM handler if modified, otherwise None.
        """
        # Try to get terminal size if available
        try:
            self.columns, self.rows = os.get_terminal_size()
        except OSError:
            self.columns, self.rows = 80, 24  # Default if not available

        # Determine the operating system
        self.is_windows = os.name == 'nt'
        self.is_linux = os.name == 'posix'
        self.is_notebook = 'ipykernel' in sys.modules

        # Set up input method based on environment
        if self.is_notebook:
            self.get_input = self._get_input_notebook
        elif self.is_linux or self.is_windows:  # Covers most common non-notebook terminals
            self.get_input = self._get_input_cross_platform
        else:
            # Fallback for other environments, defaulting to cross_platform which uses threads
            self.get_input = self._get_input_cross_platform

        self.timeout_occurred: bool = False
        self.original_sigalrm_handler: tp.Optional[tp.Callable] = None

    def format_time_remaining(self, seconds: float) -> str:
        """
        Formats a duration in seconds into a "mm:ss" string.

        Args:
            seconds (float): The total number of seconds to format.
                             Can be a float, but will be floored to the nearest
                             integer for minutes and seconds calculation.

        Returns:
            str: A string representing the time in "mm:ss" format, with minutes
                 and seconds zero-padded to two digits (e.g., "05:30").
        """
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def display_countdown(
        self,
        end_time: float,
        stop_event: threading.Event,
        timeout_event: threading.Event,
        input_received_event: threading.Event
    ):
        """
        Displays a countdown timer (mm:ss) in the console.

        This method is typically run in a separate thread. It updates a line in the
        console with the remaining time. It uses ANSI escape codes to position the
        cursor and clear the line, aiming to minimize interference with user input
        on other lines. The countdown stops if the `stop_event` is set, if the
        `input_received_event` is set, or if the `end_time` is reached (at which
        point `timeout_event` is set).

        Args:
            end_time (float): The timestamp (seconds since epoch) when the countdown
                              should end.
            stop_event (threading.Event): An event that, when set, signals this
                                          countdown display to terminate.
            timeout_event (threading.Event): An event that this method will set if
                                             the countdown reaches zero.
            input_received_event (threading.Event): An event that signals input
                                                   has been received successfully.

        Notes:
            This method assumes a terminal environment that supports ANSI escape codes.
            The timer display is fixed to the first line of the console (`timer_line = 1`).
        """
        timer_line = 1  # Display timer on the first line

        # Perform an initial update immediately
        current_remaining = max(0, end_time - time.time())
        time_str = f"Time remaining: {self.format_time_remaining(current_remaining)}"

        # ANSI escape codes: save cursor, move to timer_line:1, clear line, print, restore cursor
        sys.stdout.write(f"\033[s\033[{timer_line};1H\033[K{time_str}\033[u")
        sys.stdout.flush()

        # Loop to update the timer every second until stop_event is set or time runs out
        while not stop_event.is_set() and not input_received_event.is_set():
            time.sleep(1)  # Wait for one second

            # Check again if we should stop
            if stop_event.is_set() or input_received_event.is_set():
                break

            current_remaining = max(0, end_time - time.time())
            time_str = f"Time remaining: {self.format_time_remaining(current_remaining)}"

            # Update the timer display line
            sys.stdout.write(f"\033[s\033[{timer_line};1H\033[K{time_str}\033[u")
            sys.stdout.flush()

            if current_remaining <= 0:
                # Only set timeout if input hasn't been received
                if not input_received_event.is_set():
                    timeout_event.set()  # Signal that time has run out
                break

    def alarm_handler(self, signum: int, frame: tp.Optional[types.FrameType]):
        """
        Signal handler for `SIGALRM` to indicate a timeout.

        This method is intended to be used as a signal handler for `signal.SIGALRM`.
        When triggered, it sets the `self.timeout_occurred` flag to True and raises
        an `InputTimeoutError` to interrupt blocking operations like `input()`.

        Args:
            signum (int): The signal number that was received.
            frame (Optional[types.FrameType]): The current stack frame at the time
                                              the signal was received.

        Raises:
            InputTimeoutError: Always raised to signal that the input timed out.
        """
        self.timeout_occurred = True
        raise InputTimeoutError("Timeout occurred while waiting for input.")

    def get_yes_no_input(
        self,
        timeout_seconds: int = 900,
        input_message: str = "Please enter 'y' or 'n': "
    ) -> bool:
        """
        Requests a 'yes' or 'no' (y/n) input from the user with a specified timeout.

        This method dispatches to an environment-specific input handler
        (`_get_input_notebook` or `_get_input_cross_platform`) determined during
        `__init__`. It prompts the user with `input_message` and waits for a 'y'
        or 'n' response (case-insensitive) within `timeout_seconds`.
        The countdown timer restarts if invalid input is provided.

        Args:
            timeout_seconds (int, optional): The maximum time in seconds to wait for
                                             input. Defaults to 900 (15 minutes).
            input_message (str, optional): The message prompt displayed to the user.
                                           Defaults to "Please enter 'y' or 'n': ".

        Returns:
            bool: True if the user enters 'y' (case-insensitive), False if 'n'.

        Raises:
            InputTimeoutError: If no valid input is received before the timeout in the
                               called private method.
            KeyboardInterrupt: If the user manually interrupts (e.g., Ctrl+C)
                               in the notebook environment when it's not due to timeout.
        """
        return self.get_input(timeout_seconds=timeout_seconds, input_message=input_message)

    def _get_input_notebook(self, timeout_seconds: int, input_message: str) -> bool:
        """
        Handles timed 'y/n' input specifically for Jupyter notebook environments.

        This method attempts to get a 'y' or 'n' (case-insensitive) input from the
        user within the given `timeout_seconds`. It displays a countdown timer.
        If the timer expires, it tries to interrupt the `input()` call using
        `os.kill(os.getpid(), signal.SIGINT)` which typically raises a `KeyboardInterrupt`.
        The timer resets if invalid input is provided.

        Args:
            timeout_seconds (int): The maximum time in seconds to wait for input.
            input_message (str): The prompt message displayed to the user.

        Returns:
            bool: True if the user enters 'y', False if 'n'.

        Raises:
            InputTimeoutError: If no valid input is received before the timeout,
                               or if a `KeyboardInterrupt` occurs due to timeout.
            KeyboardInterrupt: If the user manually interrupts (e.g., Ctrl+C)
                               not related to the timeout mechanism.
        """
        print()  # Line for spacing

        while True:  # Iterate in case the input given was not part of the expected one
            # Setup timeout tracking
            self.timeout_occurred = False
            start_time = time.time()
            end_time = start_time + timeout_seconds

            # Setup timer thread
            timer_thread_stopped = threading.Event()
            input_received_successfully = threading.Event()

            def update_timer_display():
                """
                Continuously updates the countdown timer display in the notebook.

                This function runs in a separate thread. It prints the remaining time,
                refreshing the line. If the timeout is reached AND input hasn't been
                received successfully, it sets the `self.timeout_occurred` flag and
                attempts to interrupt the main thread's `input()` call by sending a
                `SIGINT` signal. The loop terminates if `timer_thread_stopped` event
                is set, timeout occurs, or input is received successfully.
                """
                while not timer_thread_stopped.is_set() and not input_received_successfully.is_set():
                    remaining = max(0, end_time - time.time())
                    # `\r` moves cursor to beginning of line, `end=""` prevents newline
                    print(f"\rTime remaining: {self.format_time_remaining(remaining)}", end="")
                    sys.stdout.flush()  # Ensure it's written immediately

                    # Check for timeout only if input hasn't been received
                    if time.time() >= end_time and not input_received_successfully.is_set():
                        self.timeout_occurred = True
                        # In a notebook, os.kill(os.getpid(), signal.SIGINT)
                        # is a common way to interrupt input().
                        if hasattr(signal, 'SIGINT'):
                            try:
                                # This will send SIGINT to the process
                                # which should interrupt input() with KeyboardInterrupt
                                os.kill(os.getpid(), signal.SIGINT)
                            except Exception:  # pylint: disable=broad-except
                                # Ignore errors if kill fails (e.g. permissions, process state)
                                pass
                        timer_thread_stopped.set()  # Signal thread to stop
                        break

                    time.sleep(0.5)  # Update display roughly every half second

            # Start timer display thread
            timer_thread = threading.Thread(target=update_timer_display)
            timer_thread.daemon = True
            timer_thread.start()

            try:
                # Use a custom signal handler if possible
                if hasattr(signal, 'SIGALRM'):
                    # Set up alarm signal to interrupt after timeout_seconds
                    self.original_sigalrm_handler = signal.signal(
                        signal.SIGALRM,
                        self.alarm_handler
                    )
                    signal.alarm(int(timeout_seconds))

                # Get user input
                user_input = input(input_message)

                # Check if timeout occurred during input (before processing the input)
                current_time = time.time()
                if self.timeout_occurred or current_time >= end_time:
                    raise InputTimeoutError("input not provided.")

                # Process input
                cleaned_input = user_input.strip().lower()
                if cleaned_input in ['y', 'n']:
                    # Signal that input was received successfully
                    input_received_successfully.set()
                    timer_thread_stopped.set()

                    # Wait for timer thread to finish
                    timer_thread.join(timeout=1.0)

                    return cleaned_input == 'y'
                else:
                    print("\n\nInvalid input. Please enter only 'y' or 'n'.")
                    # Continue the loop to get new input
                    timer_thread_stopped.set()
                    # Wait for timer thread to finish before next iteration
                    timer_thread.join(timeout=1.0)
                    # Reset timer by continuing the loop

            except KeyboardInterrupt as e:
                # Check if it's due to timeout or user interruption
                if self.timeout_occurred or time.time() >= end_time:
                    raise InputTimeoutError("input not provided.") from e
                raise

            except InputTimeoutError:
                # Propagate timeout error
                raise

            except Exception as e:
                # For other exceptions, check if timeout occurred
                if self.timeout_occurred or time.time() >= end_time:
                    raise InputTimeoutError("input not provided.") from e
                # Otherwise re-raise the exception
                print(f"Error: {e}")
                raise

            finally:
                # Clean up signal handler if set
                if hasattr(signal, 'SIGALRM') and self.original_sigalrm_handler:
                    signal.alarm(0)  # Cancel the alarm
                    signal.signal(signal.SIGALRM, self.original_sigalrm_handler)

                # Stop timer thread
                timer_thread_stopped.set()

                # Wait for timer thread to finish
                if timer_thread.is_alive():
                    timer_thread.join(timeout=1.0)

                # Final check for timeout only if input wasn't received successfully
                if not input_received_successfully.is_set() and (self.timeout_occurred or time.time() >= end_time):
                    raise InputTimeoutError("input not provided.")

    def _get_input_cross_platform(self, timeout_seconds: int, input_message: str) -> bool:
        """
        Handles timed 'y/n' input for cross-platform terminal environments.

        This method uses threading to manage a countdown timer and non-blocking input.
        It clears the screen and uses ANSI escape codes to position the cursor for
        the timer, input prompt, and error messages. The goal is to provide a clean
        interface that refreshes on invalid input or times out.

        Args:
            timeout_seconds (int): The maximum time in seconds to wait for input.
            input_message (str): The prompt message displayed to the user.

        Returns:
            bool: True if the user enters 'y', False if 'n'.

        Raises:
            InputTimeoutError: If no valid input is received before the timeout.

        Notes:
            - Clears the console screen (`os.system('cls'/'clear')`) at the start,
              on timeout, and on valid input.
            - Assumes a terminal that supports ANSI escape codes for cursor
              positioning and line clearing.
            - Line positions for timer, input, and errors are hardcoded.
        """
        timer_line = 1
        input_line = 3
        error_line = 4

        # Clear the screen using a cross-platform approach
        os.system('cls' if os.name == 'nt' else 'clear')

        # Setup initial screen
        print()  # Spacing line

        # Main input loop
        while True:
            input_received = threading.Event()
            input_received_successfully = threading.Event()
            user_input: tp.List[tp.Optional[str]] = [None]
            is_valid: tp.List[bool] = [False]

            # Initialize timer events
            start_time = time.time()
            end_time = start_time + timeout_seconds
            stop_countdown_event = threading.Event()
            timeout_event = threading.Event()
            self.timeout_occurred = False  # Reset for each attempt

            # ANSI: move to input_line, clear line, print prompt
            sys.stdout.write(f"\033[{input_line};1H\033[K{input_message}")
            sys.stdout.flush()

            def get_input_thread_target() -> None:
                """
                Worker function to capture user input in a separate thread.

                Reads from `builtins.input()`. Stores the cleaned (lowercase, stripped)
                input in `user_input[0]` and its validity ('y'/'n') in `is_valid[0]`.
                Sets `input_received` event upon completion or error.
                """
                try:
                    raw_input_value = builtins.input()
                    user_input[0] = raw_input_value.strip().lower()
                    if user_input[0] in ['y', 'n']:
                        is_valid[0] = True
                    else:
                        is_valid[0] = False
                except EOFError:  # Handles Ctrl+D or closed input stream
                    user_input[0] = None
                    is_valid[0] = False
                except Exception:  # Catch any other input-related errors
                    user_input[0] = None
                    is_valid[0] = False
                finally:
                    # Signal that input processing is complete
                    input_received.set()

            input_thread = threading.Thread(target=get_input_thread_target)
            input_thread.daemon = True
            input_thread.start()

            countdown_thread = threading.Thread(target=self.display_countdown,
                                              args=(end_time, stop_countdown_event, timeout_event, input_received_successfully))
            countdown_thread.daemon = True
            countdown_thread.start()

            input_received.wait()  # Wait for input thread to complete

            # Check if we got valid input before the timeout
            if is_valid[0] and user_input[0] is not None and not timeout_event.is_set():
                # Signal that we received valid input
                input_received_successfully.set()

                # Stop countdown and cleanup threads
                stop_countdown_event.set()
                countdown_thread.join(timeout=1.0)

                # Clean up display
                sys.stdout.write(f"\033[{timer_line};1H\033[K")
                sys.stdout.write(f"\033[{error_line};1H\033[K")
                sys.stdout.flush()

                os.system('cls' if os.name == 'nt' else 'clear')
                return user_input[0] == 'y'

            # Stop countdown thread before checking timeout
            stop_countdown_event.set()
            countdown_thread.join(timeout=1.0)

            # ANSI: move to timer_line, clear line. Move to error_line, clear line.
            sys.stdout.write(f"\033[{timer_line};1H\033[K")
            sys.stdout.write(f"\033[{error_line};1H\033[K")
            sys.stdout.flush()

            # Check for timeout
            if timeout_event.is_set() or time.time() >= end_time:
                self.timeout_occurred = True  # Ensure flag is set
                os.system('cls' if os.name == 'nt' else 'clear')
                raise InputTimeoutError("Timeout: No valid input received.")

            # If we reach here, input was invalid
            # ANSI: move to error_line, clear line, print error
            sys.stdout.write(
                f"\033[{error_line};1H\033[K" + \
                "Invalid input. Please enter 'y' or 'n'."
            )
            sys.stdout.flush()
            time.sleep(2)  # Show error briefly
            # The loop will continue for the next attempt

# COMMAND ----------

class FeatureStoreManager:
    """
    Manages the saving and updating of feature tables and Delta tables.

    This class encapsulates the logic for creating, registering, and updating
    tables in the Databricks Feature Store, as well as handling the underlying
    Delta tables, including schema evolution and metadata management.

    Attributes:
        spark (SparkSession): The active Spark session.
        fs (FeatureStoreClient): The Databricks Feature Store client.
        table_name (str): Name of the feature table.
        database (str): Database where the feature table resides.
        path_name (str): Path where the Delta table is stored.
        primary_keys (tp.List[str]): List of primary key column names.
        timestamp_keys (tp.List[str]): List of timestamp key column names.
        entity (str): The entity associated with the features.
        table_description (str): Description of the feature table.
        source_tables (tp.List[str]): List of source tables (format: 'database.table').
        fs_table (str): Fully qualified name of the feature store table
            (format: 'database.table_name').
    """

    def __init__(
            self,
            spark: SparkSession,
            fs: FeatureStoreClient,
            table_name: str,
            database: str,
            path_name: str,
            source_tables: tp.List[str],
            primary_keys: tp.List[str],
            timestamp_keys: tp.List[str],
            entity: str,
            table_description: str,
        ):
        """
        Initializes a new instance of FeatureStoreManager.

        Args:
            spark (SparkSession): The active Spark session for DataFrame and SQL operations.
            fs (FeatureStoreClient): The Feature Store client for interacting with
                the feature registry.
            table_name (str): The name of the feature table. Stripped of leading/trailing
                whitespace.
            database (str): The database where the feature table will reside. Stripped.
            path_name (str): The HDFS or S3 path where the Delta table data will be stored.
                Stripped.
            source_tables (tp.List[str]): A list of source table names that contribute to this
                feature table. Expected format for each item is 'database.table'.
            primary_keys (tp.List[str]): A list of column names that serve as primary keys for the
                feature table.
            timestamp_keys (tp.List[str]): A list of column names that serve as timestamp keys,
                used for create time-series features and enables point-in-time lookups.
            entity (str): The name of the entity these features describe
                (e.g., 'customer', 'product'). Stripped.
            table_description (str): A human-readable description of the feature table
                and its purpose. Stripped.
        """
        self.spark = spark
        self.fs = fs
        self.table_name = table_name.strip()
        self.database = database.strip()
        self.path_name = path_name.strip()
        self.primary_keys = primary_keys
        self.timestamp_keys = timestamp_keys
        self.entity = entity.strip()
        self.table_description = table_description.strip()
        self.source_tables = source_tables
        self.validate_attributes()
        self.source_tables = self.validate_database_and_table(self.source_tables)
        self.fs_table = f"{self.database}.{self.table_name}"

    def validate_attributes(self):
        """
        Validates that essential instance attributes have valid, non-empty values.

        This method inspects attributes that were set via the constructor arguments.
        It checks for:
        - String attributes: Must not be empty or None after stripping whitespace.
        - List attributes: Must not be empty, and their elements must not be None
          or empty strings (if the elements are strings).
        - Other attributes: Must not be None.

        Raises:
            ValueError: If any of the validated attributes do not meet the criteria,
                listing all attributes that failed validation and the reason.
        """
        # Get attributes declared as input in constructor
        constructor_params = inspect.signature(self.__init__).parameters
        instance_attrs = {
            name: getattr(self, name)
            for name in constructor_params.keys()
            if name != 'self' and hasattr(self, name)
        }

        # List to store validation errors
        validation_errors = []

        # Validate each attribute based on its type
        for attr_name, value in instance_attrs.items():
            # Validate strings
            if isinstance(value, str):
                if not value or value.strip() == '':
                    validation_errors.append(f"'{attr_name}': empty string or None")

            # Validate lists
            elif isinstance(value, list):
                if not value:
                    validation_errors.append(f"'{attr_name}': empty list")
                else:
                    # Check all elements in list
                    if not all(
                        item is not None and (not isinstance(item, str) or item.strip() != '')
                        for item in value
                    ):
                        validation_errors.append(
                            f"'{attr_name}': contains None or empty string values"
                        )

            # Validate objects
            elif value is None:
                validation_errors.append(f"'{attr_name}': None value")

        # If there are errors, raise exception with all validation errors
        if validation_errors:
            error_message = "The following attributes failed validation:"
            error_message += "\n\t- " + "\n\t- ".join(validation_errors)
            raise ValueError(error_message)

    def validate_database_and_table(self, db_tables: tp.List[str]) -> tp.List[str]:
        """
        Validates format, existence, content, and permissions of specified database tables.

        Each table name in the input list is expected to be in 'database.table' format.
        The method checks for:
        - Correct format.
        - Existence of the database and table.
        - Presence of columns in the table (i.e., not completely empty).
        - Read permissions for the table.

        Args:
            db_tables (tp.List[str]): A list of strings, where each string is a
                table name in 'database.table' format.

        Returns:
            tp.List[str]: A list of validated and cleaned (stripped whitespace)
                table names if all checks pass.

        Raises:
            Exception: If any table fails validation due to incorrect format,
                non-existence, being empty (no columns), or lacking read permissions.
                The error message aggregates all failures.

        Examples:
            (Assuming `self` is an instance of `FeatureStoreManager` and `spark` is available)
            >>> self.validate_database_and_table([
            ...     "feature_store.fs_cus_demographic",
            ...     "   feature_store. fs_cus_demographic" # Spaces are stripped
            ... ])
            >>> # if feature_store.fs_cus_demographic exists and is readable.
            ['feature_store.fs_cus_demographic', 'feature_store.fs_cus_demographic']

            >>> self.validate_database_and_table(["invalid_format", "db.bad_table_name!"])
            >>> # Raises Exception with messages like:
            Error validating the following source_tables:
                - For 'invalid_format', input format is incorrect. Please use 'database.table' format.
                - For 'db.bad_table_name!', input format is incorrect. No alphanumeric characters are in your table or database name.

            >>> self.validate_database_and_table(["db.no_permission_table"])
            Raises Exception: Error validating the following source_tables:
                - For 'db.no_permission_table', no read permission were given
        """
        errors = []
        tables_validated = []
        for db_table in db_tables:
            # Validate format
            if "." not in db_table:
                if db_table.isalpha():
                    table_info = [
                        (t, b)
                        for _, t, b in source_datasets
                        if t==db_table
                    ]
                    if len(table_info)==0:
                        errors.append(
                            f"For '{db_table}', there is no table referenced in `source_dataset`."
                        )
                        continue
                    _, db_name = table_info[0]
                    if db_name is not None:
                        errors.append(
                            f"For '{db_table}', input format is incorrect. " \
                            + "Please use 'database.table'. " \
                            + "You forgot to put the database, it's '{db_name}'."
                        )
                    # if db_name is None then it will be omited as a source_table
                    # to register in the Feature Store because it's path is already captured
                    # by the FeatureStoreClient
                    continue
                else:
                    errors.append(
                        f"For '{db_table}', input format is incorrect. " \
                        + "Please use 'database.table' format."
                    )
                    continue
            table_sections = db_table.split(".")
            if len(table_sections)!=2 or not all(table_sections):
                errors.append(f"For '{db_table}', input format is incorrect. Please use 'database.table' format.")
                continue

            # Validate the existence
            database, table = table_sections
            try:
                if not self.spark.catalog._jcatalog.databaseExists(database):
                    errors.append(f"For '{db_table}', database '{database}' does not exist.")
                    continue
                if not self.spark.catalog._jcatalog.tableExists(f'{database}.{table}'):
                    errors.append(f"For '{db_table}', table '{table}' does not exist in database '{database}'.")
                    continue
            except ParseException:
                errors.append(
                    f"For '{db_table}', input format is incorrect. " + \
                    "No alphanumeric characters are in your table or database name."
                )
                continue

            # Validate if is full empty
            if len(self.spark.table(db_table).columns)==0:
                errors.append(f"For '{db_table}', is full empty and has no columns")
                continue

            # Validate read permissions
            db_table = f"{database.strip()}.{table.strip()}"
            try:
                # Intenta leer solo la primera fila para verificar permisos
                self.spark.table(db_table).limit(1).collect()
                tables_validated.append(db_table)
            except Exception:
                errors.append(f"For '{db_table}', no read permission were given")
                continue

        if errors:
            raise Exception(
                "Error validating the following source_tables:" + \
                "\n\t- " + "\n\t- ".join(errors)
            )
        return tables_validated

    def get_fs_conf(self) -> tp.Dict[str, tp.Any]:
        """
        Generates a configuration dictionary for a Databricks Feature Store table.

        This configuration is typically used when creating or updating a feature table
        using the FeatureStoreClient. It includes a description, primary keys,
        timestamp keys, and relevant tags derived from the instance attributes.

        Returns:
            tp.Dict[str, tp.Any]: A dictionary with the following structure:
                &nbsp;- description (str): A descriptive string for the feature table.
                &nbsp;- primary_keys (tp.List[str]): List of primary key column names.
                &nbsp;- timestamp_keys (tp.List[str]): List of timestamp key column names.
                &nbsp;- tags (tp.Dict[str, str]): A dictionary of tags including:
                    &nbsp;- database (str): The database of the feature table.
                    &nbsp;- based_on (str): The entity the features are based on.
                            This "entity" must be in the primary keys.
                    &nbsp;- table (str): The description of the table.
        """
        return {
            "description": f"Features for {self.entity} {self.table_description}.",
            "primary_keys": self.primary_keys,
            "timestamp_keys": self.timestamp_keys,
            "tags": {
                'database': self.database,
                'based_on': self.entity,
                'table': self.table_description
            },
        }

    def fs_table_exists(self, fs_table: str) -> bool:
        """
        Checks if a table exists in the Databricks Feature Store.

        Args:
            fs_table (str): The fully qualified name of the feature table
                (e.g., 'database_name.table_name').

        Returns:
            bool: True if the table exists in the Feature Store, False otherwise.
        """
        try:
            self.fs.get_table(fs_table)
            return True
        except Exception:
            return False

    def db_table_has_no_columns(self, table:str):
        """
        Checks if a given Delta table has no columns.

        Executes a SQL query to select all columns from the specified table. 
        If the table exists but has no columns, catches the AnalysisException 
        and returns True. Otherwise, returns False or raises the exception.

        Args:
            table (str): The name of the table to check.

        Returns:
            bool: True if the table exists but has no columns, False otherwise.

        Raises:
            AnalysisException: If an error occurs that is not related to the table having no columns.
        """
        try:
            spark.sql(f"select * from {table}")
            return False
        except AnalysisException as e:
            if "[DELTA_READ_TABLE_WITHOUT_COLUMNS]" in str(e):
                error_message = "You are trying to read a Delta table `{table}` that does not have any columns."
                logger.warning(error_message, exc_info=True)
                return True
            else:
                raise e

    def _preprocess_table_path(self, table_path: str) -> str:
        """
        Preprocesses a table path string to ensure it's in a valid format for Delta Lake
        or Spark SQL.

        - If the path starts with 'dbfs:' or '/mnt', it's assumed to be a direct Delta path
          and is enclosed in backticks with 'delta.`...`'.
        - If the path contains a dot ('.'), it's assumed to be a 'database.table' reference
          and is returned as is.
        - Otherwise, the format is considered invalid.

        Args:
            table_path (str): The raw table path or name string.

        Returns:
            str: The processed table path string, ready for use in Spark operations.

        Raises:
            ValueError: If the `table_path` format is not recognized as a valid
                DBFS path, mount path, or 'database.table' identifier.
        """
        if table_path.startswith("dbfs") or table_path.startswith("/mnt"):
            return f"delta.`{table_path}`"
        elif "." in table_path:
            return table_path
        else:
            raise ValueError(
                f"Invalid table path format: {table_path}. " + \
                "Path must start with 'dbfs' or '/mnt' or contain a dot (.) " + \
                "for database.table format"
            )

    # ---------------------- Saving in the DataLake -----------------------
    def _get_metadata(self, table_path: str) -> tp.Dict[str, tp.Any]:
        """
        Retrieves metadata for a Delta table specified by its path.

        This function reads details from a Delta table, including its description,
        primary keys, timestamp keys, tags, partition columns, and source tables
        by querying the table's properties and detail.

        Args:
            table_path (str): The path to the Delta table. This can be a path
                (e.g., '/path/to/table', 'dbfs:/path/to/table') or a table identifier
                ('database.table'). It will be preprocessed by `_preprocess_table_path`.

        Returns:
            tp.Dict[str, tp.Any]: A dictionary containing the table's metadata.
                The structure includes keys like 'description', 'primary_keys',
                'timestamp_keys', 'tags', 'source_tables', 'partition_columns',
                and 'path'.
                Example structure:
                ```
                {
                    "description": "Table description.",
                    "primary_keys": ["pk_col1", "pk_col2"],
                    "timestamp_keys": ["ts_col1"],
                    "tags": {"tag_key": "tag_value", "user_fs_source_tables": "db.table1,db.table2"},
                    "source_tables": ["db.table1", "db.table2"],  // Extracted from tags
                    "partition_columns": ["part_col1"],
                    "path": "/mnt/path/to/delta_table"
                }
                ```

        Notes:
            - The table at `table_path` must be a valid Delta table.
            - Custom metadata like 'primary_keys', 'timestamp_keys', and 'source_tables'
              are expected to be stored in the Delta table's properties/tags,
              often prefixed (e.g., 'user_fs_primary_keys').
            - The 'source_tables' in the return dictionary is parsed from a
              comma-separated string in the 'user_fs_source_tables' tag.
        """
        metadata = {}
        path_for_sql = self._preprocess_table_path(table_path)

        table_descriptions_json = self.spark.sql(f"DESCRIBE DETAIL {path_for_sql}")\
            .toJSON()\
            .collect()[0]
        table_descriptions = json.loads(table_descriptions_json)

        metadata["partition_columns"] = table_descriptions.get("partitionColumns", [])
        metadata["path"] = table_descriptions["location"]
        properties = table_descriptions.get("properties", {})
        metadata["description"] = properties.get("description", None)
        metadata["primary_keys"] = json.loads(properties.get("primary_keys", json.dumps([])))
        metadata["timestamp_keys"] = json.loads(properties.get("timestamp_keys", json.dumps([])))
        metadata["tags"] = json.loads(properties.get("tags", json.dumps({})))
        metadata["source_tables"] = json.loads(properties.get("source_tables", json.dumps([])))
        return metadata

    def _update_metadata(self, table_path: str, properties: tp.Dict[str, tp.Any]) -> None:
        """
        Updates the table properties (TBLPROPERTIES) of a Delta table.

        This method compares the provided `properties` with the existing metadata
        of the table at `table_path`. It only applies updates for properties
        whose values have changed. If no properties need updating, it logs a
        message and returns.

        Args:
            table_path (str): The path or identifier of the Delta table whose
                metadata is to be updated. This will be processed by
                `_preprocess_table_path`.
            properties (tp.Dict[str, tp.Any]): A dictionary where keys are property
                names and values are the new property values. Values are typically
                strings, or lists/dicts that will be JSON-stringified before storing.
                Examples of properties include 'description', 'primary_keys',
                'timestamp_keys', 'tags', 'source_tables'.

        Notes:
            - The table at `table_path` must be a valid Delta table.
            - Property values (like lists or dicts) are stored as JSON strings in
              `TBLPROPERTIES`.
        """
        current_metadata = self._get_metadata(table_path)
        for prop, value in list(properties.items()):
            if current_metadata.get(prop) == value:
                properties.pop(prop)
        if not properties:
            logger.info("DATALAKE METADATA: No properties to save or update")
            return

        properties_str = ", ".join([
            f"'{k}' = '{json.dumps(v) if isinstance(v, list) or isinstance(v, dict) else v}'"
            for k, v in properties.items()
        ])
        logger.info("DATALAKE METADATA: Properties to save or update")

        path_for_sql  = self._preprocess_table_path(table_path)
        logger.info("DATALAKE METADATA: Properties to save or update:")
        print(properties_str)
        try:
            sql_query = f"""
            ALTER TABLE {path_for_sql}
            SET TBLPROPERTIES ({properties_str})
            """
            self.spark.sql(sql_query)
            logger.info("DATALAKE METADATA: Updated successfully")
        except Exception as e:
            logger.error(f"DATALAKE METADATA: Error updating - {e}")
            raise

    def _validate_schema(self, df: DataFrame) -> tp.Dict[str, tp.Any]:
        """
        Validates the schema of a DataFrame against an existing Delta table.

        Compares the schema of the input DataFrame (`df`) with the schema of the
        Delta table located at `self.path_name`. It identifies new columns,
        missing columns, and columns whose data types have changed.

        Args:
            df (DataFrame): The PySpark DataFrame whose schema is to be validated.

        Returns:
            tp.Dict[str, tp.Any]: A dictionary containing the validation results:
                &nbsp;- 'NEW_COLUMNS' (tp.List[str]): Columns present in `df` but
                not in the Delta table.
                &nbsp;- 'MISSING_COLUMNS' (tp.List[str]): Columns present in the Delta table
                but not in `df`.
                &nbsp;- 'DATA_TYPE_CHANGED' (tp.Dict[str, str]): Columns present in both
                but with different data types. The dictionary has the column names as keys
                and as values an explanation of the difference between the data types.

        Notes:
            - If the Delta table at `self.path_name` does not exist, all columns in `df`
              are considered new, and 'NEW_COLUMNS', 'MISSING_COLUMNS' and 'DATA_TYPE_CHANGED'
              will be empty.
        """
        validations = {}
        if not DeltaTable.isDeltaTable(self.spark, self.path_name):
            logger.info(
                f"VALIDATION SCHEMA: Delta table doesn't exist at '{self.path_name}', " + \
                "all columns will be considered new"
            )
            validations["NEW_COLUMNS"] = []
            validations["MISSING_COLUMNS"] = []
            validations["DATA_TYPE_CHANGED"] = {}
        else:
            try:
                existing_df = self.spark.read.format("delta").load(self.path_name)
                existing_schema = existing_df.schema
                new_schema = df.schema
                existing_columns = set(existing_schema.names)
                df_columns = set(new_schema.names)

                new_cols = list(df_columns.difference(existing_columns))
                validations["NEW_COLUMNS"] = new_cols

                missing_cols = list(existing_columns.difference(df_columns))
                validations["MISSING_COLUMNS"] = missing_cols

                data_type_changes = {}
                for field in new_schema.fields:
                    if field.name in existing_columns:
                        existing_field = existing_schema[field.name]
                        if field.dataType.simpleString() != existing_field.dataType.simpleString():
                            data_type_changes[field.name] = (
                                f"Expected: {existing_field.dataType.simpleString()}, " +
                                f"found: {field.dataType.simpleString()}"
                            )
                validations["DATA_TYPE_CHANGED"] = data_type_changes
            except Exception as e:
                logger.error(f"VALIDATION SCHEMA: Error during validation for {self.path_name}")
                pprint(validations)
                raise e

        feature_name_msg = (
            f"`{self.fs_table}`"
            if self.database and self.table_name
            else f"table at `{self.path_name}`"
        )
        if DeltaTable.isDeltaTable(self.spark, self.path_name) and validations.get("NEW_COLUMNS"):
            message = f"New columns found for existing {feature_name_msg}:\n\t- " + \
                "\n\t- ".join(validations["NEW_COLUMNS"])
            warnings.warn(message)
        if validations.get("DATA_TYPE_CHANGED"):
            message = (
                f"Data type changes found for {feature_name_msg}:\n\t- " +
                "\n\t- ".join([
                    f"{col}: {mnsg}"
                    for col, mnsg in validations["DATA_TYPE_CHANGED"].items()
                ])
            )
            warnings.warn(message)
        if validations.get("MISSING_COLUMNS"):
            message = (
                f"For {feature_name_msg}, the following columns are missing " +
                "in the new dataframe (present in existing table):\n\t- " +
                "\n\t- ".join(validations["MISSING_COLUMNS"])
            )
            warnings.warn(message)
        return validations

    def save_in_datalake(
        self,
        df: DataFrame,
        types_changes_detected: dict[str, str],
        new_columns_detected: list[str],
        missing_columns_detected: list[str],
        properties: tp.Dict = None,
        force_overwrite: bool = False,
        overwriteSchema: bool = False
    ) -> bool:
        """
        Saves or updates a DataFrame as a Delta table, handling schema evolution and partitioning.

        This method manages writing a PySpark DataFrame to a Delta Lake table specified by
        `self.path_name`. It handles schema validation results (type changes, new/missing columns),
        applies partitioning based on `self.timestamp_keys` (year, month, day from the first key),
        and can operate in overwrite or merge/append mode.

        Args:
            df (DataFrame): The PySpark DataFrame to be saved or merged into the Delta table.
            types_changes_detected (tp.Dict[str, str]): A dictionary indicating columns with changed
                data types, typically from `_validate_schema`.
            new_columns_detected (tp.List[str]): A list of new column names not present in the
                existing Delta table, from `_validate_schema`.
            missing_columns_detected (tp.List[str]): A list of column names present in the existing
                Delta table but missing in the input `df`, from `_validate_schema`.
            properties (tp.Dict[str, tp.Any], optional): Additional table properties to set or update
                on the Delta table. Defaults to None.
            force_overwrite (bool, optional): If True, forces a complete overwrite of the Delta table,
                even if it exists and has data. Defaults to False.
            overwriteSchema (bool, optional): If True, allows overwriting the schema of the Delta
                table when `force_overwrite` is also True. Defaults to False.

        Returns:
            bool: True if the schema of the Delta table was altered during the save operation
                (e.g., due to new columns or `overwriteSchema`).

        Raises:
            ValueError: If the primary timestamp key (`self.timestamp_keys[0]`) is not found in the
                input DataFrame `df`.
            DeltaError (or other Spark/Delta exceptions): If errors occur during Delta table
                operations (e.g., schema merge conflicts not automatically resolvable, I/O issues).

        Notes:
            - The method first checks if the primary timestamp key exists in the DataFrame.
            - It determines if the target Delta table already exists. If it's empty, `force_overwrite`
              is enabled automatically.
            - Partitioning is automatically applied using 'year', 'month', 'day' derived from the
              first timestamp key if `self.timestamp_keys` is not empty.
            - Schema evolution (`mergeSchema=True`) is used for appends/merges if new columns are
              detected and `overwriteSchema` is False.
            - If `force_overwrite` is True, the table is written in 'overwrite' mode. Otherwise,
              a merge operation (update existing, insert new) is attempted based on
              `self.primary_keys`.
            - Table properties are updated after the write/merge operation.
            - We are note defining partitionBy keys in the write operation because we are creating
              time-series features. It's necessary to create time-series features because the
              datascience team needs to use point-in-time joins when they are creating their
              master tables.
                (reference: https://docs.databricks.com/aws/en/machine-learning/feature-store/workspace-feature-store/feature-tables#store-past-values-of-daily-features)
              And the documentation stablish that a time-series feature table must have
              one timestamp key and cannot have any partition columns.
                (reference: https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series?language=Workspace%C2%A0Feature%C2%A0Store%C2%A0client%C2%A0v0.13.4%C2%A0and%C2%A0above#create-a-time-series-feature-table-in-local-workspace)
        """
        if not properties:
            properties = {}

        schema_changed_by_operation = False

        if self.timestamp_keys[0] not in df.columns:
            raise ValueError(
                f"DATALAKE PREP: Main timestamp_key `{self.timestamp_keys[0]}` not in dataframe"
            )

        is_existing_delta_table = DeltaTable.isDeltaTable(self.spark, self.path_name)

        if is_existing_delta_table:
            num_registers = self.spark.read.format("delta").load(self.path_name).count()
            if num_registers == 0:
                logger.info("DATALAKE PREP: Table has 0 registers, forcing overwrite")
                force_overwrite = True

        if is_existing_delta_table and not force_overwrite:
            try:
                if self.spark.read.format("delta").load(self.path_name).count() == 0:
                    logger.info("DATALAKE PREP: Existing table has 0 registers, forcing overwrite")
                    force_overwrite = True
            except Exception as e:
                logger.warning(
                    "DATALAKE PREP: Count error, will merge if not force_overwrite. " + \
                    f"Error: {e}"
                )

        # Execute the saving logic
        # TODO: explicit drop columns without a force_overwrite. It will help to just drop columns
        # and after that insert the new data or upsert with new columns
        # If implemented, change the code after the TimedInput
        if not is_existing_delta_table or force_overwrite:
            logger.info(
                "DATALAKE WRITE: Starting full write (non-existent or force_overwrite=True)"
            )
            save_op = df.write.format("delta")

            if not is_existing_delta_table:
                logger.info("DATALAKE WRITE: Creating new table")

            if overwriteSchema and is_existing_delta_table:
                logger.info("DATALAKE WRITE: Overwrite Schema enabled")
                save_op = save_op.option("overwriteSchema", "true")
                if types_changes_detected or new_columns_detected or missing_columns_detected:
                    schema_changed_by_operation = True
            elif (new_columns_detected or types_changes_detected) and is_existing_delta_table:
                logger.info("DATALAKE WRITE: Merge Schema for new columns on existing table")
                save_op = save_op.option("mergeSchema", "true")
                schema_changed_by_operation = True
            elif missing_columns_detected and is_existing_delta_table:
                schema_changed_by_operation = True
            elif is_existing_delta_table:
                save_op = save_op.option("mergeSchema", "false")
                schema_changed_by_operation = False

            save_op = save_op.mode("overwrite").save(self.path_name)
            logger.info("DATALAKE WRITE: Full write completed")
        else:
            logger.info("DATALAKE MERGE: Starting update/merge operation")
            columns_to_update = []
            if new_columns_detected:
                logger.info(f"DATALAKE MERGE: Adding new columns: {new_columns_detected}")
                add_clauses = [
                    f"`{c}` {df.schema[c].dataType.simpleString()}"
                    for c in new_columns_detected
                ]
                self.spark.sql(
                    f"ALTER TABLE delta.`{self.path_name}` ADD COLUMNS ({', '.join(add_clauses)})"
                )
                # reference: https://spark.apache.org/docs/latest/sql-ref-syntax-ddl-alter-table.html
                columns_to_update += new_columns_detected
                schema_changed_by_operation = True
            if types_changes_detected and overwriteSchema:
                warning_msg = (
                    "DATALAKE MERGE: overwriteSchema=True but merge mode active. " +
                    f"Type changes: {types_changes_detected}. " +
                    "Merge won't alter types. Consider force_overwrite=True or manual evolution."
                )
                warnings.warn(warning_msg)
                logger.warning(warning_msg)

            delta_table_obj = DeltaTable.forPath(self.spark, self.path_name)
            merge_cond = " AND ".join([f"target.`{k}` = source.`{k}`" for k in self.primary_keys])
            merger = delta_table_obj.alias("target").merge(df.alias("source"), merge_cond)
            if columns_to_update:
                update_expr = {
                    update_col: f"source.{update_col}"
                    for update_col in columns_to_update
                }
                merger = merger.whenMatchedUpdate(set=update_expr)
                logger.info(f"DATALAKE MERGE: Updated the following columns {columns_to_update}")
            merger.whenNotMatchedInsertAll().execute()
            logger.info("DATALAKE MERGE: Merge completed")

        # Optimize the table saved for future readings
        try:
            logger.info("DATALAKE OPTIMIZE: Starting optimization")
            DeltaTable.forPath(self.spark, self.path_name).optimize().executeCompaction()
            logger.info("DATALAKE OPTIMIZE: Optimization completed")
        except Exception as e:
            logger.warning(f"DATALAKE OPTIMIZE: Failed - {e}")

        # Save / Update the table's TBLPROPERTIES
        if properties:
            self._update_metadata(self.path_name, properties)
        else:
            logger.info("DATALAKE METADATA: No custom properties to update")

        return schema_changed_by_operation

    # ----------------- Saving in Feature Store ---------------------
    def _register_feature_table(self) -> None:
        """
        Registers the underlying Delta table with the Databricks Feature Store.

        This method uses the `fs.register_table` API to make the Delta table (specified by
        `self.fs_table` which is `self.database`.`self.table_name`) known to the Feature Store.
        It uses configuration details (primary keys, timestamp keys, description, tags)
        obtained from `self.get_fs_conf()`.

        If the table is already registered but has a schema different from the underlying
        Delta table at `self.path_name`, this method will:
        1. Log the schema difference (cluster logs and prints, not in monitoring logs).
        2. Drop the existing Feature Store table registration (`fs.drop_table`).
        3. Recreate the Databricks catalog entry for the table using the Delta table
        at `self.path_name`. It's necessary to do because the `fs.drop_table` drops
        the table from the Databricks catalog.
        4. Retry the `fs.register_table` operation.

        Raises:
            ValueError: If `fs.register_table` fails for reasons other than a schema mismatch
                that can be resolved by dropping and recreating (e.g., invalid configuration).
            pyspark.sql.utils.AnalysisException: If `spark.sql` commands fail (e.g., table
                creation or if `self.path_name` does not point to a valid Delta table).

        Notes:
            - The method relies on `self.fs_table` (format: 'database.table_name') for the Feature
              Store table name and `self.path_name` for the location of the Delta table data.
            - `self.get_fs_conf()` provides essential metadata like primary keys, timestamp keys,
              description, and tags required for registration.
            - Logging is performed throughout the process to indicate success, schema differences,
              and retries.
        """
        fs_conf = self.get_fs_conf()
        try:
            self.fs.register_table(
                delta_table=self.fs_table,
                **fs_conf
            )
            logger.info(
                "FEATURE_STORE REGISTER: Feature Table registered without schema differences"
            )
        except ValueError as e:
            if "already exists with a different schema" in str(e).lower():
                logger.info(
                    "FEATURE_STORE REGISTER: Dropping Feature Table due to schema differences"
                )
                self.fs.drop_table(self.fs_table)
                self.spark.sql(f"""
                    CREATE TABLE {self.fs_table}
                    USING DELTA
                    LOCATION '{self.path_name}'
                """)
                self.fs.register_table(
                    delta_table=self.fs_table,
                    **fs_conf
                )
            else:
                raise e
        logger.info("FEATURE_STORE REGISTER: Feature Table registered successfully")

    def save_in_feature_store(self, schema_changed: bool) -> None:
        """
        Ensures a Delta table is properly registered and configured in the Databricks Feature Store.

        This method handles the final steps of integrating a Delta table
        (located at `self.path_name`) into the Feature Store. It performs the following actions:
        1. Validates that `self.path_name` points to an existing, non-empty Delta table.
        2. If the `schema_changed` flag is True, or if the underlying Delta table is empty (which
        can happen on first save or if data was cleared), it drops any existing Feature Store
        table registration to ensure a clean slate.
        3. If the Databricks catalog does not already contain an entry for `self.fs_table` (i.e.,
        `self.database`.`self.table_name`), or if the Feature Store table was just dropped,
        it creates the Databricks catalog table linked to the Delta location.
        4. Calls `self._register_feature_table()` to register the table with the Feature Store,
        applying metadata like primary keys, timestamp keys, and descriptions.

        Args:
            schema_changed (bool): A flag indicating whether the schema of the underlying
                Delta table has changed since it was last registered or saved.
                This triggers a drop and re-registration if True.

        Raises:
            FileNotFoundError: If `self.path_name` does not point to a Delta table or
                does not exist.
            pyspark.sql.utils.AnalysisException: If the Delta table at `self.path_name` is empty
                and has no columns (cannot register an empty, schemaless table), or if other
                Spark SQL operations fail.
            ValueError: Propagated from `_register_feature_table` if registration fails for
                reasons other than a resolvable schema mismatch.

        Notes:
            - This method assumes that the Delta table at `self.path_name` has already been created
              or updated by a preceding operation (e.g., `save_in_datalake`).
            - It uses `self.fs_table` for the Feature Store table name and `self.path_name` for the
              Delta table's physical location.
        """
        if not DeltaTable.isDeltaTable(self.spark, self.path_name):
            raise FileNotFoundError(
                f"FEATURE_STORE: The path '{self.path_name}' is not a Delta table or does not exist"
            )

        if len(self.spark.read.format("delta").load(self.path_name).columns)==0:
            raise AnalysisException(
                f"FEATURE_STORE: The table at '{self.path_name}' " + \
                "is empty and has no columns to register"
            )

        # Handle Possible Errors with Feature Registering
        # If we need to modify the schema we'll need to drop the table from the Feature Store
        if self.fs_table_exists(self.fs_table):
            logger.info("FEATURE_STORE CHECK: Feature Table already exists")
            if (
                self.fs.read_table(self.fs_table).count()==0
                or self.db_table_has_no_columns(self.fs_table)
                or schema_changed
                ):
                try:
                    self.fs.drop_table(self.fs_table)
                    spark.sql(f"""
                    DROP TABLE IF EXISTS {self.fs_table};
                    """) # Ensure drop in databricks catalog
                    logger.info("FEATURE_STORE DROP: Executed successfully")
                except Exception as e:
                    logger.error(f"FEATURE_STORE DROP: Error - {e}")
            else:
                logger.info("FEATURE_STORE DROP: Skipped")
        # In case table does not exist in the catalog or feature table was dropped
        if not self.spark.catalog._jcatalog.tableExists(self.fs_table):
            logger.info(f"FEATURE_STORE CREATE: {self.fs_table} does not exist, creating table")
            self.spark.sql(f"""
                CREATE TABLE {self.fs_table}
                USING DELTA
                LOCATION '{self.path_name}'
            """)
        else:
            logger.info(f"FEATURE_STORE CREATE: {self.fs_table} already exists")

        # Execute the Register
        self._register_feature_table()

    # ----------------- main process ---------------------
    def save(
            self,
            df: DataFrame,
            force_overwrite: bool = False,
            overwriteSchema: bool = False,
            omit_data_validation_errors: bool = False,
        ) -> None:
        """
        Orchestrates the full process of saving a DataFrame as a feature table.

        This is the main public method for saving data. It handles:
        1.  Ensuring the target database exists.
        2.  Validating the input DataFrame's schema against the existing Delta table
        using `_validate_schema`.
        3.  Handling schema differences (type changes, new/missing columns) with options to
        either raise errors, attempt to alter types, or proceed with schema evolution.
        4.  Optionally allowing users to interactively decide whether to proceed if missing
        columns are detected (unless `omit_data_validation_errors` or `force_overwrite` is True).
        5.  Calling `save_in_datalake` to write the DataFrame to a Delta table at `self.path_name`,
        managing partitioning, overwrite logic, and schema evolution options.
        6.  Calling `save_in_feature_store` to register/update the Delta table in the Databricks
        Feature Store and manage its data source lineage.

        Args:
            df (DataFrame): The PySpark DataFrame to be saved as a feature table.
            force_overwrite (bool, optional): If True, completely overwrites the existing Delta
                table and Feature Store table. Defaults to False.
            overwriteSchema (bool, optional): If True (and `force_overwrite` is True), allows
                the schema of the Delta table to be overwritten. Defaults to False.
            omit_data_validation_errors (bool, optional): If True, suppresses errors related to
                data type changes or missing columns during schema validation, and attempts to
                proceed. If False (default), errors will be raised for these issues unless
                `force_overwrite` is also True for missing columns.

        Raises:
            Exception: If critical validation errors occur (e.g., missing columns when not
                overwriting and not omitting errors, data type alteration failures) or if
                underlying Delta or Feature Store operations fail.
            InputTimeoutError: If user interaction for missing columns times out.

        Notes:
            - The method performs extensive logging of its operations and decisions.
            - It uses `TimedInput` for interactive prompts if `omit_data_validation_errors` is False
              and missing columns are detected.
            - Schema evolution (adding new columns or merging schema for type changes) is
              delegated to `save_in_datalake` and controlled by `overwriteSchema` and the
              nature of detected changes.
            - If `self.alter_types_when_saving` is True, it will attempt to use
              `_alter_column_data_types` for detected type changes before saving to Datalake.
        """
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        feature_name_log = (
            f"`{self.database}.{self.table_name}`"
            if self.database and self.table_name
            else f"table at `{self.path_name}`"
        )

        # ------------------------- Validate DataFrame ------------------------------
        schema_validation_results = self._validate_schema(df)
        types_changes_detected = schema_validation_results.get("DATA_TYPE_CHANGED", {})
        new_columns_detected = schema_validation_results.get("NEW_COLUMNS", [])
        missing_columns_detected = schema_validation_results.get("MISSING_COLUMNS", [])
        throw_missing_columns_error = False
        # Handle when there are missing columns
        if missing_columns_detected:
            missing_columns_message = (
                f"VALIDATION MISSING_COLS: For feature `{feature_name_log}` " + \
                "missing columns in dataframe:" + \
                "\n\t- " + "\n\t- ".join(missing_columns_detected)
            )
            # Give the opportunity to the user to decide if overwrite with
            # the missing columns or raise the exception
            try:
                logger.warning(missing_columns_message)
                timed_input = TimedInput()
                response = timed_input.get_yes_no_input(
                    timeout_seconds=60*5,  # 5 minutes waiting the user answer
                    input_message=(
                        "VALIDATION USER_INPUT: Do you want to OVERWRITE " + \
                        "the Feature Table despite the missing columns? (y/n): "
                    )
                )
                print() # adds a line break after the "input" finish
                if response is True:
                    force_overwrite = True
                    overwriteSchema = True
                    throw_missing_columns_error = False
                    logger.info(
                        "VALIDATION USER_INPUT: User confirmed overwrite with missing columns"
                    )
                else:
                    throw_missing_columns_error = True
                    logger.info("VALIDATION USER_INPUT: User declined overwrite")
            except InputTimeoutError:
                throw_missing_columns_error = True
                logger.warning("VALIDATION USER_INPUT: Timeout - rejecting save operation")

        # Evaluate if the DataFrame passed the validations
        if not omit_data_validation_errors and throw_missing_columns_error:
            message = (
                "VALIDATION: Failed - missing columns in dataframe: \n\t- " +
                "\n\t- ".join(missing_columns_detected)
            )
            raise AnalysisException(message)

        logger.info("VALIDATION: ✅ Passed all validations")

        # ------------------------- Save table in DataLake --------------------------
        # Only if passed the validations, save the table as delta in the DataLake
        properties = self.get_fs_conf()
        properties["source_tables"] = self.source_tables
        schema_changed = self.save_in_datalake(
            df=df,
            properties=properties,
            force_overwrite=force_overwrite,
            overwriteSchema=overwriteSchema,
            types_changes_detected=types_changes_detected,
            new_columns_detected=new_columns_detected,
            missing_columns_detected=missing_columns_detected,
        )

        # ------------------------ Register in Feature Store ------------------------
        self.save_in_feature_store(schema_changed=schema_changed)

        # ------------------------ Process completed successfully ------------------------
        logger.info("PROCESS: ✅ Save operation completed successfully")
        logger.info(f"PROCESS: Feature {feature_name_log} is ready for use")
