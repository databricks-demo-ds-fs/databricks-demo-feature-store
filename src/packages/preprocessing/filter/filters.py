import logging
import typing as tp

import pyspark.sql.functions as f
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def filter_by_fixed_conditions(
    df: DataFrame, conditions: tp.Optional[tp.List[str]]
) -> DataFrame:
    """
    Filters the DataFrame based on a list of conditions.

    Args:
        df (DataFrame): The DataFrame to filter.
        conditions (Optional[List[str]]): A list of conditions as strings to filter the DataFrame.
            Each condition should be a valid Spark SQL expression.

    Returns:
        DataFrame: Filtered Spark DataFrame.

    Example:
        filter_by_fixed_conditions:
            - "purchase_amount > 100"
            - "status = 'completed'"
    """
    if not conditions:
        return df

    filter_cond = f.lit(True)
    for cond in conditions:
        filter_cond = filter_cond & (f.expr(cond))
    logger.info(f"Initial Columns in DataFrame: ({df.columns})")
    logger.info(f"Initial table filtered shape: ({df.count()}, {len(df.columns)})")
    df = df.filter(filter_cond)
    logger.info(f"New table filtered shape: ({df.count()}, {len(df.columns)})")

    return df


def filter_by_max_date(df: DataFrame, params) -> DataFrame:
    """
    Filters the DataFrame by the maximum date in a specific column.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - date_column (str): Name of the date column to filter by.

    Returns:
        DataFrame: Filtered Spark DataFrame.

    Example:
        filter_by_max_date:
            date_column: transaction_date
    """
    date_column = params["date_column"]

    max_date = df.select(f.max(f.col(date_column))).collect()[0][0]
    df = df.filter(f.col(date_column) == max_date)

    return df
