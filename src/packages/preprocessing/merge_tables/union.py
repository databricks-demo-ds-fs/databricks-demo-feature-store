import logging

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def union_dataframes(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """
    Union two PySpark DataFrames.

    Args:
        df1 (DataFrame): The first PySpark DataFrame.
        df2 (DataFrame): The second PySpark DataFrame.

    Returns:
        DataFrame: A new DataFrame resulting from the merge operation.
    """

    data = df1.unionByName(df2, allowMissingColumns=True)

    logger.info(f"Unifying dataframes: {data.columns}")
    return data
