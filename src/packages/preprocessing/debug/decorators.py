import functools
import logging

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def print_shape(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, DataFrame):
            shape = (result.count(), len(result.columns))
            logger.info(f"Function: {func.__name__}, Shape of DataFrame: {shape}")
        return result

    return wrapper


def print_column_names(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, DataFrame):
            to_print = list(result.columns)
            logger.info(
                f"Function: {func.__name__}, Columns of DataFrame:\n\t\t{to_print}"
            )
        return result

    return wrapper
