import logging
import typing as tp

import pyspark.sql.functions as f
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def standardize_na_values(
    df: DataFrame, na_values: tp.Optional[tp.Dict[str, tp.List[str]]]
) -> DataFrame:
    """
    Standardizes specified NA values in the DataFrame.

    Args:
        df (DataFrame): The DataFrame to standardize NA values in.
        na_values (Optional[Dict[str, List[str]]]): A dictionary where keys are column names
            and values are lists of strings representing NA values to standardize.

    Returns:
        DataFrame: The DataFrame with standardized NA values.

    Example:
        na_values = {'column1': ['NA', 'N/A'], 'column2': ['NULL']}
        standardized_df = standardize_na_values(df, na_values)
    """
    if not na_values:
        logger.info("No NA values to standardize.")
        return df
    logger.info(f"Standardizing NA values for columns: {list(na_values.keys())}")
    replacements = {
        field: f.when(f.col(field).isin(na_vals), None).otherwise(f.col(field))
        for field, na_vals in na_values.items() if na_vals
    }
    if replacements:
        df = df.withColumns(replacements)
    return df


def drop_na(
    df: DataFrame, subset: tp.Optional[tp.Union[str, tp.List[str]]]
) -> DataFrame:
    """
    Drops rows with NA values in specified columns of the DataFrame.

    Args:
        df (DataFrame): The DataFrame to drop NA values from.
        subset (Optional[List[str]]): A list of column names where NA values will be dropped.

    Returns:
        DataFrame: The DataFrame with rows containing NA values dropped.

    Raises:
        KeyError: If any column in `subset` is not found in the DataFrame.

    Example:
        subset = ['column1', 'column2']
        cleaned_df = drop_na(df, subset)
    """
    if not subset:
        logger.info("No columns specified for dropping NA values.")
        return df

    # Make drop to all fields in the DataFrames
    drop_all = subset.lower() == "all" if type(subset) is str else False
    if drop_all:
        logger.info(f"Dropna for all the columns")
        return df.na.drop()

    invalid_columns = set(subset).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Columns not found in DataFrame: {invalid_columns}")
    logger.info(f"Dropping NA values for columns: {subset}")
    return df.na.drop(subset=subset)


def drop_duplicates(
    df: DataFrame, subset: tp.Optional[tp.Union[str, tp.List[str]]]
) -> DataFrame:
    """
    Drops duplicate rows based on specified columns in the DataFrame.

    Args:
        df (DataFrame): The DataFrame to drop duplicate rows from.
        subset (Optional[List[str]]): A list of column names to consider when identifying duplicates.
            If not specified, all columns will be used.

    Returns:
        DataFrame: The DataFrame with duplicate rows dropped.

    Example:
        subset = ['column1', 'column2']
        cleaned_df = drop_duplicates(df, subset)
    """
    # If missing values in `subset`, don't modify
    if not subset:
        logger.info("No columns specified for dropping duplicates.")
        return df

    drop_duplicates_for_all = subset.lower() == "all" if type(subset) is str else False
    if drop_duplicates_for_all:
        logger.info("DropDuplicates is applied to all the columns.")
        return df.dropDuplicates()

    # Validate if columns exist in DataFrame
    invalid_columns = set(subset).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Columns not found in DataFrame: {invalid_columns}")

    logger.info(f"Dropping duplicates for columns: {subset}")
    return df.dropDuplicates(subset)
