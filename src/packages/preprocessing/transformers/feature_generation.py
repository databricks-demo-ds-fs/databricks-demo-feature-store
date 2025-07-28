import logging
import operator
import typing as tp

import pyspark.sql.functions as f
from pyspark.sql import Column as cf
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def math_operations(
    df: DataFrame,
    params: tp.List[tp.Dict[str, tp.Union[str, tp.List[str]]]],
) -> DataFrame:
    """
    Performs mathematical operations between columns to create new columns.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (list): List of dictionaries, each containing:
            - new_column_name (str): Name of the new column.
            - operator (str): Mathematical operation (e.g., 'add', 'subtract').
            - values (list): List of columns involved in the operation.
            - source_operator (str): Source of the operator ('operator' or 'spark_function').

    Returns:
        DataFrame: Spark DataFrame with new columns created by applying mathematical operations.

    Example:
        math_operations:
            new_column_name: result_column
            operator: add
            values:
              - column1
              - column2
            source_operator: operator
    """
    if not params:
        return df

    for metadata in params:
        new_column_name = metadata["new_column_name"]
        op = metadata["operator"]
        values = metadata["values"]
        source_operator = metadata["source_operator"]

        if source_operator == "operator":
            df = df.withColumn(
                new_column_name,
                getattr(operator, op)(f.col(values[0]), f.col(values[1])),
            )
        elif source_operator == "spark_function":
            if type(values) is dict:
                df = df.withColumn(new_column_name, getattr(f, op)(**values))
            elif type(values) is list:
                df = df.withColumn(new_column_name, getattr(f, op)(*values))
    return df


def spark_function_operations(
    df: DataFrame,
    params: tp.List[tp.Dict[str, tp.Union[str, tp.List[str]]]],
) -> DataFrame:
    """
    Applies Spark functions to create new columns.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (list): List of dictionaries, each containing:
            - new_column_name (str): Name of the new column.
            - source_function (str): Function source ('spark_function' or 'spark_column_function').
            - function (str): Spark function to be applied.
            - values (list or dict): Arguments to pass to the function.

    Returns:
        DataFrame: Spark DataFrame with new columns created by applying the specified functions.

    Example:
        spark_function_operations:
          - new_column_name: col_name
            source_function: spark_function
            function: col
            values:
              - col1
              - col2
    """
    if not params:
        return df

    source_functions_allowed = ["spark_function", "spark_column_function"]
    logger.info(f"Original columns: {df.columns}")

    for i, metadata in enumerate(params):
        new_column_name = metadata["new_column_name"]
        source_function = metadata["source_function"]
        op = metadata["function"]
        values = metadata["values"]
        if source_function == "spark_function":
            module = f
        # TODO: standarize spark_column_function, it isn't working now
        elif source_function == "spark_column_function":
            module = cf
        else:
            raise ValueError(
                f"Incorrect source_function. you put '{source_function}' "
                + f"and only allowed: {source_functions_allowed}"
            )

        # just pyspark.sql.functions is tested, not pyspark.sql.Column
        if type(values) is dict:
            df = df.withColumn(new_column_name, getattr(module, op)(**values))
        elif type(values) is list:
            df = df.withColumn(new_column_name, getattr(module, op)(*values))

        logger.info(f"Columns after modify ({i+1}): {df.columns}")

    return df


def choose_columns_as_value(df: DataFrame, params) -> DataFrame:
    """
    Creates a new column by choosing values from existing columns based on conditions.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): The name of the new column.
            - cases (list): List of conditions and corresponding columns.
            - else_value (str, optional): Default value if no condition is met.
            - cast_type (str, optional): Data type for the new column.

    Returns:
        DataFrame: Spark DataFrame with the newly created column.

    Example:
        choose_columns_as_value:
            new_column_name: selected_column
            cases:
                - condition: "condition1"
                  column: column1
                - condition: "condition2"
                  column: column2
            else_value: "default_value"
            cast_type: STRING
    """
    new_column_name = params["new_column_name"]
    cases = params.get("cases", [])
    cast_type = params.get("cast_type", None)
    else_value = params.get("else_value", None)
    none_values = ("NONE", "NULL", "NA", "N/A", "NAN")
    is_null = else_value.upper() in none_values if type(else_value) is str else False

    # Validate that columns exist in the DataFrame
    columns_in_cases = [case["column"] for case in cases]
    logger.info(f"Columns for case conditions: {columns_in_cases}")
    invalid_columns = set(columns_in_cases).difference(set(df.columns))
    if invalid_columns:
        logger.info(
            f"The following values were not found in DataFrame Columns: {invalid_columns}. "
            + "If your intention is to assign a value and not a column, it's all ok"
        )

    # query construction
    query_content = (
        ["CASE"]
        + [
            f"WHEN {case_values['condition']} " + f"THEN {case_values['column']}"
            for case_values in cases
        ]
        + (
            [f"ELSE NULL END"]
            if (is_null)
            else (
                [f"ELSE {else_value} END"]
                if else_value is not None
                else [f"ELSE '{else_value}' END"]
            )
        )
    )
    # allows to return nulls
    # allows to return the column's value
    # allows to return a fixed value detailed in the yml
    query = " ".join(query_content)
    logger.info(f"query: {query}")

    df = df.withColumn(
        new_column_name,
        f.expr(query) if cast_type is None else f.expr(query).cast(cast_type.lower()),
    )

    return df


def categorize_by_fixed_values(
    df: DataFrame,
    params: tp.Dict[
        str, tp.Union[str, int, tp.Dict[tp.Union[int, str], tp.Union[int, float, str]]]
    ],
) -> DataFrame:
    """
    Categorizes a column based on fixed values and assigns them to a new column.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - column (str): The column to categorize.
            - new_column_name (str): The name of the new categorized column.
            - values (dict): A dictionary of values and conditions.
            - else_value (str, optional): Value for cases not matching any condition.
            - cast_type (str, optional): Data type for the new column.

    Returns:
        DataFrame: Spark DataFrame with the categorized column.

    Example:
        categorize_by_fixed_values:
            column: category_column
            new_column_name: categorized_column
            values:
                1: "= 'A'"
                2: "= 'B'"
            else_value: 0
            cast_type: INT
    """
    if not params:
        return df

    column = params["column"]
    new_column_name = params["new_column_name"]
    values = params["values"]
    else_value = params.get("else_value", None)
    none_values = ("NONE", "NULL", "NA", "N/A", "NAN")
    is_null = else_value.upper() in none_values if type(else_value) is str else False

    # query construction
    query_content = (
        ["CASE"]
        + [
            f"WHEN {column}{ ' '+cond if cond[0].isalpha() else cond } " +
            # If starts with a letter,
            # its a reserved word for an order,
            # for example "LIKE". It needs to add a white space
            f"THEN '{new_value}'"
            for new_value, cond in values.items()
        ]
        + (
            [f"ELSE NULL END"]
            if (is_null)
            else (
                [f"ELSE {else_value} END"]
                if else_value in df.columns
                else [f"ELSE '{else_value}' END"]
            )
        )
    )
    # allows to return nulls
    # allows to return the column's value
    # allows to return a fixed value detailed in the yml
    query = " ".join(query_content)
    logger.info(f"Categorizing to: {column} -> {new_column_name}\n{query}")

    unique_values_before = [
        elem[column] for elem in df.select(column).distinct().collect()
    ]
    logger.info(f"Values before categorized `{column}`: {unique_values_before}")
    logger.info(f"query: {query}")

    df = df.withColumn(
        new_column_name,
        (
            f.expr(query).cast(params["cast_type"].lower())
            if params.get("cast_type", None) is not None
            else f.expr(query)
        ),
    )

    unique_values_after = [
        elem[new_column_name]
        for elem in df.select(new_column_name).distinct().collect()
    ]
    logger.info(f"Values categorized `{column}`: {unique_values_after}")

    return df


def binarize(df: DataFrame, params) -> DataFrame:
    """
    Binarizes specified columns based on provided expressions.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing column names as keys and expressions as values.

    Returns:
        DataFrame: Spark DataFrame with binarized columns.

    Example:
        binarize:
            new_column: "condition_expression"
            another_column: "another_expression"
    """
    # In case there are no params
    if not params:
        return df

    logger.info(f"Columns before binarize: {df.columns}")
    logger.info(f"params of binarize: {params.items()}")

    # Binarize
    for new_column_name, query in params.items():
        df = df.withColumn(new_column_name, f.expr(query))
        if new_column_name not in df.columns:
            logger.info(f"New column created: {new_column_name}")

    logger.info(f"Columns after binarize: {df.columns}")
    return df


def create_unic_value_col(df: DataFrame, params) -> DataFrame:
    """
    Creates a new column with a constant value.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): Name of the new column.
            - data_type (str): Data type of the new column.
            - value (str): Value to be assigned to the new column.

    Returns:
        DataFrame: Spark DataFrame with the new column containing the constant value.

    Example:
        create_unic_value_col:
            new_column_name: constant_value
            data_type: INT
            value: 100
    """
    new_column_name = params["new_column_name"]
    data_type = params["data_type"]
    value = params["value"]

    df = df.withColumn(new_column_name, f.lit(value).cast(data_type))
    return df
