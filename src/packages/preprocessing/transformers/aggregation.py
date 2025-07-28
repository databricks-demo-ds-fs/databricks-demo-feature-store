import logging
import typing as tp

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def window_operations(
    df: DataFrame,
    params: tp.Dict[str, tp.Union[str, tp.List[tp.Union[str, tp.Dict[str, str]]]]],
) -> DataFrame:
    """
    Applies window functions to create new columns based on specified operations.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - by_columns (list): Columns to partition the window.
            - orderby (str, optional): Column to order within the window.
            - operations (list): List of operations to apply.

    Returns:
        DataFrame: Spark DataFrame with new columns created by applying window functions.

    Example:
        win_op1:
          by_columns:
            - customer_id
          orderby: date
          operations:
            - operation: row_number
              new_column_name: row_num

    Note: The value in `operation` parameter must to be the function's aggregation name
        we want to excecute for the window operation.
    """
    if not params:
        return df

    by_columns = params["by_columns"]
    if "orderby" in params:
        generate_agrupacion = Window.partitionBy(by_columns).orderBy(params["orderby"])
    else:
        generate_agrupacion = Window.partitionBy(by_columns)

    for metadata in params["operations"]:
        column = metadata.get("column", None)
        df = df.withColumn(
            metadata["new_column_name"],
            getattr(f, metadata["operation"])(  # call the aggegation function
                *[] if column is None else [f.col(column)]
            ).over(generate_agrupacion),
        )
    return df


def aggregation_operations(
    df: DataFrame,
    params: tp.Dict[str, tp.Union[bool, tp.List[tp.Dict[str, str]], tp.Dict[str, str]]],
) -> DataFrame:
    """
    Aggregates data by specified columns and operations.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - by_columns (list): Columns to group by.
            - operations (list): List of operations to apply on specified columns.
                The values in this list are a dictionary with teh enought information
                to excecute the aggregation function.
                It's a list because you could excecute multiple functions for the same
                groupBy.
            - pivot (dict, optional): Pivot configuration.
            - join (bool, optional): Whether to join the result back to the original DataFrame.

    Returns:
        DataFrame: Aggregated Spark DataFrame.

    Example:
        agg_op7:
          by_columns:
            - customer_id
          pivot: {} # doesn't make pivot
          operations:
            - column: amount_due
              operation: sum
              alias: total_due
            - column: risk_score
              operation: mean
              alias: mean_risk
          join: True

    Example with using pivot:
        agg_op8:
          by_columns:
            - cod_cliente
            - month_dt
            - event_dt
          join: False
          pivot:
            column: des_tipo_credito
          operations:
            - column: cant_registros
              operation: max
              alias: False

    Note: The value in `operation` parameter must to be the function's aggregation name
        we want to excecute for the window operation.
    """
    if not params:
        return df

    logger.info(f"Initial columns before aggregation: {df.columns}")

    list_funcs = [
        (
            getattr(f, metadata["operation"])(  # call the aggregation function
                *[f.col(metadata["column"])] if metadata.get("column", None) else []
            ).alias(metadata["alias"])
            if metadata.get("alias", None)
            else getattr(f, metadata["operation"])(
                *[f.col(metadata["column"])] if metadata.get("column", None) else []
            )
        )
        for metadata in params["operations"]
    ]

    by_columns = params.get("by_columns", [])
    if len(by_columns) == 0:
        raise ValueError(
            "Missing columns to group the data. `by_columns` is an empty list "
            + "or is not found among the parameters"
        )

    pivot = params.get("pivot", None)
    logger.info(f"Do pivot: {pivot is not None}")
    if pivot:
        logger.info(f"Columns before pivot: {df.columns}")
        df_grouped = df.groupby(*by_columns).pivot(pivot["column"]).agg(*list_funcs)
        logger.info(f"Columns after pivot: {df_grouped.columns}")
    else:
        df_grouped = df.groupby(*by_columns).agg(*list_funcs)

    join = params.get("join", False)
    if join is True:
        aliases = [
            metadata["alias"]
            for metadata in params["operations"]
            if metadata.get("alias", None)
        ]
        select_columns = [col for col in df.columns if col not in aliases]
        # The select is done to avoid having duplicate columns after the join
        return df.select(*select_columns).join(df_grouped, on=by_columns)

    logger.info(f"Columns after aggregation: {df.columns}")
    return df_grouped
