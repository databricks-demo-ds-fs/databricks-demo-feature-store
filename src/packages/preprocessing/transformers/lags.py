import logging
import typing as tp

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def create_lags_ind(df: DataFrame, params) -> DataFrame:
    """
    Creates an indicator for lagged rows by partitioning the data
    with a window function.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - by_columns (list): Columns to partition the data.
            - orderby (str): Column to order the data.

    Returns:
        DataFrame: Spark DataFrame with lag indicators.

    Example:
        create_lags_ind:
            by_columns:
              - customer_id
            orderby: period
    """
    by_columns = params["by_columns"]
    orderby = params["orderby"]

    generate_partition = Window.partitionBy(*by_columns).orderBy(orderby)
    df = df.withColumn("seq_filas", f.row_number().over(generate_partition))

    # Calculate number of rows per group
    generate_partition = Window.partitionBy(*by_columns).orderBy([by_columns[0]])
    df = df.withColumn("cant_max_filas", f.max("seq_filas").over(generate_partition))

    # Reorder the values of `seq_filas`
    df = df.withColumn(
        "lags_num", df["cant_max_filas"] - df["seq_filas"].cast(IntegerType())
    )

    # Make the value a string
    df = df.withColumn("lags", f.concat(f.lit("lag_"), f.col("lags_num")))

    df = df.drop("cant_max_filas", "seq_filas")
    return df


def generate_lags(
    df: DataFrame, params: tp.Dict[str, tp.Union[str, tp.List[tp.Union[int, str]]]]
) -> DataFrame:
    """
    Generates lag features for specified columns.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - columns (list): Columns for which to generate lag features.
            - lags (list): List of lag intervals to apply.
            - sliding_windows (dict, optional): Sliding window statistics to calculate.
            - relative_change (list, optional): List of relative change calculations.

    Returns:
        DataFrame: Spark DataFrame with lag features.

    Example:
        generate_lags:
            pk: cod_cliente
            tsk: month_dt # aniomes
            columns:
              - monto_vencido
            lags:
              - 1
              - 3
              - 6
    """
    sdf_with_lags = df
    columns = params["columns"]
    columns = columns if type(columns) is list else [columns]

    # Validar que las columnas existan en el DataFrame
    invalid_columns = set(columns).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Columns not found in DataFrame: {invalid_columns}")

    logger.info(f"Columns before lag operation: {df.columns}")

    pk = params["pk"]
    tsk = params.get("tsk", None)
    agrupation = Window.partitionBy(pk)
    if tsk is not None:
        agrupation = agrupation.orderBy(tsk)
    for column in columns:
        # Compute lags
        if "lags" in params:
            for lag in params["lags"]:
                sdf_with_lags = sdf_with_lags.withColumn(
                    f"{column}_lag_{lag}", f.lag(f.col(column), lag).over(agrupation)
                )
        # Compute sliding windows for mean and std
        if "sliding_windows" in params:
            for stat, windows in params["sliding_windows"].items():
                for window in windows:
                    start, end = window
                    window_spec = agrupation.rowsBetween(start, end)
                    if stat == "mean":
                        sdf_with_lags = sdf_with_lags.withColumn(
                            f"{column}_mean_{start}_{end}",
                            f.mean(column).over(window_spec),
                        )
                    elif stat == "std":
                        sdf_with_lags = sdf_with_lags.withColumn(
                            f"{column}_std_{start}_{end}",
                            f.stddev(column).over(window_spec),
                        )
        if "relative_change" in params:
            for actual, reference in params["relative_change"]:
                sdf_with_lags = sdf_with_lags.withColumn(
                    f"{column}_rel_change_{actual}_{reference}", f.when()
                )

    logger.info(f"Columns after lag operation: {sdf_with_lags.columns}")
    return sdf_with_lags
