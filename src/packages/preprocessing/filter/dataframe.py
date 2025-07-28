import logging
import typing as tp
import warnings

import pyspark
import pyspark.sql.functions as f

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _filter_dataframe(
    df: pyspark.sql.DataFrame, conditions: tp.List[str]
) -> pyspark.sql.DataFrame:
    """
    Applies a series of filtering conditions to a PySpark DataFrame and returns
    the filtered DataFrame.

    This function allows for dynamic filtering of a DataFrame based on a list
    of conditions provided as strings. Each condition should be a valid SQL expression
    string understandable by PySpark. The DataFrame is then ordered by '_id' and
    '_observ_end_dt' columns.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame to be processed.
        conditions (List[str]): A list of string conditions to be applied as
        filters on the DataFrame.  Each string should be a valid SQL expression.

    Returns:
        pyspark.sql.DataFrame: The filtered DataFrame, sorted by '_id' and '_observ_end_dt'.

    Example:
        ```python
        from pyspark.sql import SparkSession

        # Initialize a Spark session
        spark = SparkSession.builder.appName("PreProcessing").getOrCreate()

        # Example DataFrame
        data = [(1, "2021-01-01"), (2, "2021-01-02"), (3, "2021-01-03")]
        df = spark.createDataFrame(data, ["_id", "_observ_end_dt"])

        # Example conditions
        conditions = ["_id > 1"]

        # Apply pre-processing
        processed_df = pre_processing(df, conditions)
        processed_df.show()
        ```
    """
    filter_cond = f.lit(True)
    if conditions:
        for cond in conditions:
            filter_cond = filter_cond & (f.expr(cond))

    logger.info(f"Initial table filtered shape: ({df.count()}, {len(df.columns)})")
    df = df.filter(filter_cond)
    logger.info(f"New table filtered shape: ({df.count()}, {len(df.columns)})")

    return df


def filter_max_value_in_column(
    df: pyspark.sql.DataFrame, column: str
) -> pyspark.sql.DataFrame:

    max_value = df.agg(f.max(column)).collect()[0][0]

    logger.info(f"Maximum value in column: {max_value}")

    filtered_df = df.filter(df[column] == max_value)

    return filtered_df


def filter_dataframe_by_fixed_conditions(
    parameters: tp.Dict[str, tp.Dict[str, tp.List[str]]],
    **dfs: tp.Dict[str, pyspark.sql.DataFrame],
) -> tp.Dict[str, pyspark.sql.DataFrame]:
    """
    Apply filters in specified columns from DataFrames based on parameters.

    Args:
        parameters (Dict[str, Dict[str, List[str]]]): A dictionary specifying filters to apply
            for each DataFrame.
        dfs (Dict[str, DataFrame]): Input DataFrames.

    Returns:
        Dict[str, DataFrame]: DataFrames with selected columns.
    """
    selected_dfs = {}

    for df_name, df_spark in dfs.items():
        if df_name in parameters:
            conditions = parameters[df_name].get("conditions", [])

            # Double length and columns definition
            if conditions is None:
                conditions = []

            length_columns = len(conditions)

            if length_columns > 0:

                # Apply filters
                selected_df = _filter_dataframe(df_spark, conditions=conditions)
                selected_dfs[df_name] = selected_df

            else:
                # If no filters specified, keep the DataFrame as is
                selected_dfs[df_name] = df_spark
        else:
            # If DataFrame name not in parameters, keep the DataFrame as is
            selected_dfs[df_name] = df_spark

    return selected_dfs
