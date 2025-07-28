import logging
import typing as tp

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def select_columns(df: DataFrame, columns: tp.List[str]) -> DataFrame:
    """
    Selects a subset of columns from the DataFrame.

    Args:
        df (DataFrame): Input Spark DataFrame.
        columns (list): List of column names to select.

    Returns:
        DataFrame: Spark DataFrame with the selected columns.

    Example:
        select_columns:
          - customer_id
          - purchase_amount
          - purchase_date
    """
    if columns is None:
        return df
    if columns.__len__() == 0:
        return df

    # Validate columns exist in the Dataframe
    invalid_columns = set(columns).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Columns not found in DataFrame: {invalid_columns}")

    logger.info(f"Selecting: original dataframe columns: {df.columns}")

    return df.select(*columns)


def select_distinct(
    df: DataFrame,
    columns: tp.List[str],
) -> DataFrame:
    """
    Selects distinct rows from a DataFrame based on specified columns.

    Args:
        df (DataFrame): The input DataFrame.
        columns (List[str]): A list of column names to select and ensure distinct
                             values across.

    Returns:
        df: A DataFrame with distinct rows based on the specified columns.

    Example:
        distinct_op:
          - cod_cliente
          - cod_operacion

    Notes:
        If the columns list is empty, the original DataFrame is returned.
    """
    if not columns:
        return df

    df = df.select(columns).distinct()
    return df
