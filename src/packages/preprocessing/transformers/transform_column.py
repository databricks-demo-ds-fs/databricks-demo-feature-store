import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import *

import typing as tp
import logging

logger = logging.getLogger(__name__)


def concat_columns(df: DataFrame, params) -> DataFrame:
    """
    Concatenates multiple columns into a single column, with optional string concatenation.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): Name of the new column.
            - columns (list): List of columns to concatenate.
            - lit_str (str, optional): A string to concatenate with the columns (only used if a single column is provided).

    Returns:
        DataFrame: Spark DataFrame with the new concatenated column.

    Example:
        concat_columns:
            new_column_name: full_name
            columns: [first_name, last_name]
            lit_str: "prefix_"  # Optional, used only when concatenating with a single column.
    """
    new_column_name = params["new_column_name"]
    columns = params.get("columns", [])
    if len(columns) == 0:
        logger.info(f"¡¡ALERT!! Missing `columns` into the parameters")
        return df

    # Validate that columns exist in the DataFrame
    invalid_columns = set(columns).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Columns not found in DataFrame: {invalid_columns}")

    if len(columns) == 1:  # Concatenate with string
        lit_str = params.get("lit_str", None)
        if lit_str is None:
            raise ValueError(
                "Two posible causes:\n\t1. Not enought columns in `columns`\n\t2. Missing `lit_str` parameter"
            )
        df = df.withColumn(new_column_name, f.concat(f.lit(lit_str), f.col(columns[0])))
        logger.info(f"Concated column: '{columns[0]}', with string: '{lit_str}'")
    else:  # Concatenate between columns
        df = df.withColumn(
            new_column_name, f.concat(*[f.col(col_name) for col_name in columns])
        )
        logger.info(f"Concated columns: {columns}")
    return df


def rename_columns(df: DataFrame, params: tp.Dict[str, str]) -> DataFrame:
    """
    Renames columns in the DataFrame according to the specified mappings.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary mapping old column names to new names.

    Returns:
        DataFrame: Spark DataFrame with renamed columns.

    Example:
        rename_columns:
            old_column1: new_column1
            old_column2: new_column2
    """
    logger.info(f"Renaming - original dataframe columns: {df.columns}")

    # Validate columns exist in the DataFrame
    actual_column_names = params.keys()
    invalid_columns = set(actual_column_names).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Columns not found in DataFrame: {invalid_columns}")

    df = df.withColumnsRenamed(colsMap=params)
    logger.info(f"Renaming, columns to rename: {actual_column_names}")
    logger.info(f"Renaming, new column names: {params.values()}")
    logger.info(f"Renaming - final dataframe columns renamed: {df.columns}")

    return df


def impute_dataframe(
    df: DataFrame,
    value: tp.Union[str, tp.Dict] = 0,
) -> DataFrame:
    """
    Imputes missing values in the DataFrame.

    Args:
        df (DataFrame): Input Spark DataFrame.
        value (str or dict): Value to use for imputing missing data.
            - If a single value is provided, it is applied to all columns.
            - If a dictionary is provided, it specifies values per column.

    Returns:
        DataFrame: Spark DataFrame with missing values imputed.

    Example:
        impute_dataframe:
            impute_op1:
              column_name: value_to_impute

            or

            impute_op1: "unknown"
    """
    # Impute all the DataFrame by a given value `value`
    if type(value) is not dict:
        return df.na.fill(value=value)

    # Impute by especific columns
    columns_to_impute = list(value.keys())
    # Validate columns exist
    invalid_columns = set(columns_to_impute).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Columns not found in DataFrame: {invalid_columns}")

    logger.info(f"Columns to impute: {columns_to_impute}")
    df = df.na.fill(value)

    return df


def decimals_to_floats(df: DataFrame):
    """
    Takes a spark dataframe and casts all decimal columns to floats to avoid
    inconsistencies when dealing with aggregations and computations with floats

    Args:
        df (DataFrame): Input Spark DataFrame.

    Returns:
        DataFrame: Spark DataFrame with decimal columns casted to float.
    """
    decimals_to_cast = [c for c, t in df.dtypes if t.startswith("decimal")]
    for colname in decimals_to_cast:
        df = df.withColumn(colname, df[colname].cast(FloatType()))
    return df

