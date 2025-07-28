import logging
import typing as tp

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def make_join(
    df_left: DataFrame,
    df_right: DataFrame,
    params: tp.Dict[str, tp.Union[str, tp.List[str]]],
):
    """
    Performs a join on two DataFrames based on specified parameters.

    Args:
        df_left: The left DataFrame.
        df_right: The right DataFrame.
        params: A dictionary containing join parameters:
            - method (str): The join method (e.g., "inner", "left", "right").
            - left_columns (List[str]): List of columns from the left DataFrame for the join.
            - right_columns (List[str]): List of columns from the right DataFrame for the join.
            - drop_na (bool, optional): Whether to drop rows with null values after the join (default: False).

    Returns:
        A DataFrame resulting from the join operation, without column aliases.

    Raises:
        ValueError: If any required parameters (method, left_columns, right_columns) are missing.
        KeyError: If any specified join columns do not exist in their respective DataFrames.

        Returns:
        DataFrame: The joined Spark DataFrame.

    Example:
        join_op3:
            left_columns:
              - cod_deudor
            right_columns:
              - cod_deudor
            method: inner
            drop_na: True
    """
    method = params.get("method", None)
    left_columns = params.get("left_columns", None)
    right_columns = params.get("right_columns", None)
    if (method is None) or (not left_columns) or (not right_columns):
        raise ValueError(
            "Missing one of the following parameters: method, left_columns, right_columns."
        )

    # Validate key columns have the same lenght
    if len(left_columns) != len(right_columns):
        raise ValueError(
            f"The lenght of the key join values is different. They must be the same. "
            + f"Left lenght: {len(left_columns)}, Right lenght: {len(right_columns)}"
        )

    # Validate that columns are in the DataFrame
    invalid_left_columns = set(left_columns).difference(set(df_left.columns))
    invalid_right_columns = set(right_columns).difference(set(df_right.columns))
    if invalid_left_columns or invalid_right_columns:
        raise KeyError(
            f"Columns not found in DataFrame:"
            + f"\n\tcolumns invalid for df_left: {invalid_left_columns}"
            + f"\n\tcolumns invalid for df_right: {invalid_right_columns}"
        )

    logger.info(
        f"make_join - Number of registers in the input DataFrames: left({df_left.count()}), right({df_right.count()})"
    )
    logger.info("JOIN columns...")
    logger.info(f"df_left columns {df_left.columns}")
    logger.info(f"df_right columns {df_right.columns}")

    # Rename right columns - in case at least one name is different
    if left_columns != right_columns:
        df_right = df_right.withColumnsRenamed(
            {
                right_name: left_name
                for left_name, right_name in zip(left_columns, right_columns)
            }
        )
        logger.info(f"After Renaming: {df_right.columns}")

    # Avoid duplicate columns
    left_df_columns = list(set(df_left.columns))
    right_df_columns = list(set(df_right.columns).difference(left_df_columns))
    # If a column other than a join key is in both dataframes,
    # it is kept in the left dataframe and removed from the right dataframe
    right_df_columns = left_columns + right_df_columns

    logger.info(f"keys for join: {left_columns}")
    logger.info(f"Columns before join - left: {left_df_columns}")
    logger.info(f"Columns before join - right: {right_df_columns}")
    # join
    df_joined = df_left.select(*left_df_columns).join(
        df_right.select(*right_df_columns), left_columns, method
    )
    logger.info(f"Columns after join: {df_joined.columns}")

    # If you want to drop rows with na values
    drop_na = params.get("drop_na", False)
    if drop_na:
        df_joined = df_joined.dropna()

    logger.info(f"make_join - Cantidad de registros de output: {df_joined.count()}")

    return df_joined


def point_in_time_join(
    df_left: DataFrame,
    df_right: DataFrame,
    params: tp.Dict[str, tp.Union[str, tp.List[str]]],
) -> DataFrame:
    """
    Perform a point-in-time join between two DataFrames based on specific time keys and other join keys.

    This function performs a point-in-time (PIT) join, which is typically used to join two datasets
    based on a time dimension while ensuring that the join respects temporal consistency. The left dataset
    contains the "current" state, while the right dataset contains the "historical" data to be joined
    up to a certain point in time.

    Args:
        df_left: The "current" or main DataFrame to join, often representing the more recent data.
        df_right: The "historical" DataFrame containing past records to be joined with the current data.
        params: A dictionary containing the following configuration parameters:
            - point_in_time_keys (dict): A dictionary specifying the time keys for both DataFrames.
                - left (dict): Contains "column" and "alias" for the time column in `df_left`.
                - right (dict): Contains "column" and "alias" for the time column in `df_right`.
            - left_columns_keys (list): A list of key columns in `df_left` to be used for joining.
            - right_columns_keys (list): A list of key columns in `df_right` to be used for joining.
            - aliases (list, optional): A list of aliases for the key columns in both DataFrames.
            - select (dict, optional): A dictionary specifying columns to select after the join for both DataFrames.

    Returns:
        DataFrame: The resulting DataFrame after performing the point-in-time join, with appropriate columns selected
        and renamed according to the provided parameters.

    Example:
        params = {
            "point_in_time_keys": {
                "left": {"column": "event_date", "alias": "current_date"},
                "right": {"column": "history_date", "alias": "past_date"},
            },
            "left_columns_keys": ["customer_id"],
            "right_columns_keys": ["customer_id"],
            "aliases": ["cust_id"],
            "select": {
                "left": ["event_type"],
                "right": ["history_value"]
            }
        }
        result_df = _point_in_time_join(df_current, df_history, params)
    """

    left_columns = params["left_columns_keys"]
    right_columns = params["right_columns_keys"]

    left_time_column = params["point_in_time_keys"]["left"]["column"]
    right_time_column = params["point_in_time_keys"]["right"]["column"]
    left_time_alias = params["point_in_time_keys"]["left"]["alias"]
    right_time_alias = params["point_in_time_keys"]["right"]["alias"]

    select_params = params.get("select", {"left":[], "right": []})
    left_columns_to_select = set(df_left.columns).difference(
        set(
            (select_params["left"] if select_params.get("left", []) else left_columns)
            + [left_time_column]
        )
    )
    right_columns_to_select = set(df_right.columns).difference(
        set(
            (select_params["right"] if select_params.get("right", []) else right_columns)
            + [right_time_column]
        )
    )
    right_columns_to_select = right_columns_to_select.difference(left_columns_to_select)
    logger.info(f"left columns to select in join: {left_columns_to_select}")
    logger.info(f"right columns to select in join: {right_columns_to_select}")
    columns_to_select = [f.col(f"s.{col}").alias(col) for col in left_columns]
    columns_to_select += [
        f.col(f"s.{left_time_column}").alias(left_time_alias),
        f.col(f"f.{right_time_column}").alias(right_time_alias),
    ]
    columns_to_select += [
        f.col(f"s.{col}").alias(col) for col in list(left_columns_to_select)
    ]
    columns_to_select += [
        f.col(f"f.{col}").alias(col) for col in list(right_columns_to_select)
    ]

    # rename key dataframes' columns by their aliases
    key_aliases = params.get("aliases", None)
    if key_aliases:
        df_left = df_left.withColumnsRenamed(
            {
                right_name: left_name
                for left_name, right_name in zip(left_columns, key_aliases)
            }
        )
        df_right = df_right.withColumnsRenamed(
            {
                right_name: left_name
                for left_name, right_name in zip(right_columns, key_aliases)
            }
        )

    join_conditions = f.col(f"s.{left_time_column}") >= f.col(f"f.{right_time_column}")
    for col in left_columns:
        join_conditions &= f.col(f"s.{col}") == f.col(f"f.{col}")

    joined_table = (
        df_left.alias("s")
        .join(
            df_right.alias("f"),
            join_conditions,
            "left",
        )
        .select(*columns_to_select)
    )

    logger.info(f"PITJ after join: {joined_table.columns}")

    group_key_columns = left_columns + [left_time_alias]
    window_spec = Window.partitionBy(*group_key_columns).orderBy(
        f.col(right_time_alias).desc()
    )
    result_table = joined_table.withColumn(
        "row_num", f.row_number().over(window_spec)
    ).filter((f.col("row_num") == 1) | (f.col("row_num").isNull()))
    result_table = result_table.drop("row_num")

    return result_table


def exclude_ids_leftjoin(
    df_left: DataFrame,
    df_toExclude: DataFrame,
    params: tp.Dict[str, tp.Union[str, tp.List[str]]],
) -> DataFrame:
    """
    Exclude registers from `df_left` that have the values of the `df_toExclude` columns selected in params.

    Args:
        df_left (DataFrame): pyspark DataFrame to do left anti join.
        df_toExclude (DataFrame): pyspark DataFrame to select columns ids (PK's)
        params (dictionary): Dictionary whit the parameters. It has the following structure,
            {
                columns: Can be a string (unique PK) or a list (multiple PK).
                             These columns will be selected from df_toExclude to obtain the uniques registers.
            }

    Returns:
        df: DataFrame whith antijoin applied considering the columns in `params`.

    """
    if not params:
        return df_left

    df_col_keys = df_toExclude.select(params["columns"]).distinct()
    df = df_left.join(df_col_keys, on=params["columns"], how="leftanti")

    return df
