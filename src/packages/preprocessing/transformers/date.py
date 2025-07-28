from packages.storage.storage_operations import get_table
from constants import conf

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

import logging
import typing as tp

logger = logging.getLogger(__name__)


def _format_date_to_yyyymm(df: DataFrame, column_name: str) -> DataFrame:
    """
    Formats a date column in a PySpark DataFrame to yyyyMM format.

    Args:
        df (DataFrame): The PySpark DataFrame containing the date column.
        column_name (str): The name of the date column to be formatted.

    Returns:
        DataFrame: A new PySpark DataFrame with the date column formatted to yyyyMM.
    """
    new_column_name = "monthyear_" + column_name
    df = df.withColumn(
        new_column_name, f.date_format(f.to_date(f.col(column_name)), "yyyyMM")
    )
    return df


def format_dates_on_dataframe_in_format_to_yyyymm(
    df: DataFrame, columns_to_format: tp.List[str]
) -> DataFrame:
    """
    Formats date columns in a PySpark DataFrame to yyyyMM format.

    Args:
        df (DataFrame): The PySpark DataFrame to be processed.
        columns_to_format (List[str]): A list of column names containing date values to be formatted.

    Returns:
        DataFrame: A new PySpark DataFrame with the specified date columns formatted to yyyyMM.
    """
    if columns_to_format is None:
        columns_to_format = []

    if not columns_to_format:
        logger.info("This table has no columns to format to yyyyMM")
        return df

    for column_name in columns_to_format:
        df = _format_date_to_yyyymm(df, column_name)
    return df


def yearmonth_from_columns_separated(
    df: DataFrame, params: tp.Dict[str, str]
) -> DataFrame:
    """
    Combines year and month columns into a single year-month column.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): Name of the new column to be created.
            - year_column (str): Name of the year column.
            - month_column (str): Name of the month column.

    Returns:
        DataFrame: Spark DataFrame with the new year-month column.

    Example:
        yearmonth_from_columns_separated:
            new_column_name: year_month
            year_column: year
            month_column: month
    """
    new_column_name = params["new_column_name"]
    year_column = params["year_column"]
    month_column = params["month_column"]

    # Ensure that the year and month columns are of type string
    df = df.withColumn(year_column, f.col(year_column).cast(StringType()))
    df = df.withColumn(
        month_column, f.lpad(f.col(month_column).cast(StringType()), 2, "0")
    )

    # Create the new column by combining year and month
    df = df.withColumn(
        new_column_name, f.concat(f.col(year_column), f.col(month_column))
    )

    # Convert the new column to date type with format yyyyMM
    df = df.withColumn(new_column_name, f.to_date(f.col(new_column_name), "yyyyMM"))
    return df


def substract_to_date(df: DataFrame, params) -> DataFrame:
    """
    Subtracts specified days, months, or years from a date column.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): Name of the new column.
            - date_column (str): The column with the date to subtract from.
            - days_to_sub (int): Number of days to subtract.
            - months_to_sub (int): Number of months to subtract.
            - years_to_sub (int): Number of years to subtract.

    Returns:
        DataFrame: Spark DataFrame with the adjusted date.

    Example:
        substract_to_date:
            new_column_name: adjusted_date
            date_column:
              name: event_date
              format: yyyyMM
            days_to_sub: 10
            months_to_sub: 2
            years_to_sub: 1
    """
    if not params:
        raise ("Not params found")

    new_column_name = params.get("new_column_name", None)
    date_col = params["date_column"]["name"]
    format_date = params["date_column"]["format"]
    # date_col = params.get("date_col", None)
    if (not new_column_name) or (not date_col) or (not format_date):
        raise (
            "At least one of the following parameters is missing: "
            + "('new_column_name', 'date_col', 'days_to_sub')"
        )

    if date_col not in df.columns:
        raise (
            f"The column '{date_col}' doesn't exist in your dataframe, "
            + f"you have the following columns:\n{df.columns}"
        )

    days_to_sub = params.get("days_to_sub", 0)
    months_to_sub = params.get("months_to_sub", 0)
    years_to_sub = params.get("years_to_sub", 0)
    df = df.withColumn(
        new_column_name,
        f.date_sub(f.to_date(f.col(date_col), format_date).cast("date"), days_to_sub),
    )
    df = df.withColumn(
        new_column_name,
        f.add_months(new_column_name, -(months_to_sub + years_to_sub * 12)),
    )

    return df


def difference_between_dates(df: DataFrame, params: tp.Dict[str, str]) -> DataFrame:
    """
    Calculates the difference between two dates.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): Name of the new column for storing the date difference.
            - date1 (str): The first date column or 'current_date' to use the current date.
            - date2 (str): The second date column.
            - unit (str): Unit of difference ('day' or 'month').

    Returns:
        DataFrame: Spark DataFrame with the new column containing the date difference.

    Example:
        diff_dates:
            new_column_name: days_difference
            date1: start_date
            date2: end_date
            unit: day
    """
    new_column_name = params["new_column_name"]
    date1 = params["date1"]
    date2 = params["date2"]
    date_columns = [date1, date2]

    # Consider operation with current date
    if date1.lower() == "current_date":
        date1 = f.current_date()
        date_columns.pop(0)
    elif date2.lower() == "current_date":
        date2 = f.current_date()
        date_columns.pop(1)

    # Validate columns exist in the DataFrame
    invalid_columns = set(date_columns).difference(set(df.columns))
    if invalid_columns:
        raise KeyError(f"Date Columns not found in DataFrame: {invalid_columns}")

    unit = params["unit"].lower()
    if unit is None:
        raise ("`unit` is not in your params")
    unit = params["unit"].lower()
    if unit in ("day", "days", "dd"):
        df = df.withColumn(new_column_name, f.datediff(date1, date2))
    elif unit in ("month", "months", "mm"):
        df = df.withColumn(
            new_column_name, f.months_between(date1=date1, date2=date2, roundOff=False)
        )
    else:
        raise (
            f"The unit param `{unit}` is not a unit acepted value, "
            + f"please put one of the folowwing:\nfor months: ('month', 'months', 'mm')"
            + f"\nfor days: ('day', 'days', 'dd')"
        )

    return df


def months_between_dates(df: DataFrame, params: tp.Dict[str, str]) -> DataFrame:
    """
    Calculates the number of months between two dates.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): Name of the new column to store the result.
            - date1 (str): The first date column.
            - date2 (str): The second date column.

    Returns:
        DataFrame: Spark DataFrame with the new column containing the month difference.

    Example:
        months_between_dates:
            new_column_name: months_difference
            date1: start_date
            date2: end_date
    """
    new_column_name = params["new_column_name"]
    date1 = params["date1"]
    date2 = params["date2"]

    df = df.withColumn(
        new_column_name, f.months_between(date1=date1, date2=date2, roundOff=False)
    )

    return df


def difference_with_current_date(df: DataFrame, params: tp.Dict[str, str]) -> DataFrame:
    """
    Calculates the difference between a date column and the current date.

    Args:
        df (DataFrame): Input Spark DataFrame.
        params (dict): Dictionary containing:
            - new_column_name (str): Name of the new column to store the result.
            - date_column (str): Name of the date column.

    Returns:
        DataFrame: Spark DataFrame with the new column containing the date difference.

    Example:
        difference_with_current_date:
            new_column_name: days_until_today
            date_column: birth_date
    """
    new_column_name = params["new_column_name"]
    date_column = params["date_column"]

    fec_act = f.current_date()
    df = df.withColumn(new_column_name, f.datediff(fec_act, f.col(date_column)))
    return df


def shift_date(df: DataFrame, params):
    """
    shift_date:
      new_column_name: release_date
      date_column:
        name:
        format:
      days_to_add: 1
      months_to_add: 0
      years_to_add: 1
    """

    if not params:
        return df

    date_column_name = params["date_column"]["name"]
    date_column_format = params["date_column"]["format"]

    new_column_name = params["new_column_name"]

    if date_column_name not in df.columns:
        raise (
            f"The date column name `{date_column_name}` especified was not found in the DataFrame columns."
        )

    days_to_add = params.get("days_to_add", 0)
    months_to_add = params.get("months_to_add", 0)
    years_to_add = params.get("years_to_add", 0)
    df = df.withColumn(
        new_column_name,
        f.date_add(
            f.to_date(f.col(date_column_name), date_column_format).cast("date"),
            days_to_add,
        ),
    )
    df = df.withColumn(
        new_column_name,
        f.add_months(new_column_name, months_to_add + years_to_add * 12),
    )

    return df


def id_date_to_datetime(
    df: DataFrame,
    params: tp.Dict[str, str],
    layer: tp.Literal["source", "ingestion", "preprocessing"] = "preprocessing"
) -> DataFrame:
    if not params:
      return df
    
    dates_per_databases = conf.dates_per_databases
    
    id_date = params.get("id_date", None)
    if id_date is None:
      raise ValueError("The parameter `id_date` is missing.")

    base = params.get("base", None)
    if base is None:
      raise ValueError("The parameter `base` is missing.")

    if base not in dates_per_databases.keys():
      raise ValueError(f"The value of the parameter `base` is not valid. It must be one of the following: {list(dates_per_databases.keys())}")

    dataset_date, date_column_params = dates_per_databases[base]
    id_date_of_date_catalog = date_column_params["original"][0] if layer=="source" else date_column_params["renamed"][0]
    datetime_column_of_date_catalog = date_column_params["original"][1] if layer=="source" else date_column_params["renamed"][1]

    df_date = get_table(dataset_date, layer=layer, format="delta" if layer!="source" else None).select(
      id_date_of_date_catalog, 
      datetime_column_of_date_catalog
    ).withColumnsRenamed({
      id_date_of_date_catalog: id_date,
      datetime_column_of_date_catalog: "event_dt", 
    })

    df = df.join(df_date, on=id_date, how="inner")

    return df
