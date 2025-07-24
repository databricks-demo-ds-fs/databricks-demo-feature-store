# Databricks notebook source
from pyspark.sql import functions as f, DataFrame
from pyspark.sql.window import Window
from typing import List, Dict, Tuple, Union, Optional
import gc
from datetime import datetime
import typing as tp

# COMMAND ----------

# MAGIC %run /Shared/databricks-demo-feature-store/notebooks/utils $env="dev"

# COMMAND ----------

notebook_inputs = dict(dbutils.widgets.getAll())

# COMMAND ----------

source_tables = [
    "demo_db.transacciones",
]

# COMMAND ----------

# Validate save parameters
for input_key in ["force_overwrite", "overwriteSchema", "omit_data_validation_errors"]:
    input_value = notebook_inputs.get(input_key, "false").lower()
    if input_value not in ["true", "false"]:
        raise ValueError(f"{input_key} must be 'true' or 'false'. Value given: '{input_value}'.")

# filter by release_dt
filter_date = []
end_date_filter = None
for date_key in ["start_date", "end_date"]:
    date_input = notebook_inputs.get(date_key)
    if date_input is not None:
        if date_key=="start_date":
            filter_date.append(f"tpk_event_dt >= '{date_input}'")
        elif date_key=="end_date":
            end_date_filter = f"tpk_release_dt = '{date_input}'"
            filter_date.append(f"tpk_event_dt <= '{date_input}'")


# COMMAND ----------

# MAGIC %md
# MAGIC # Feature `fs_cus_transactions`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingesta

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from demo_db.transacciones

# COMMAND ----------

1

# COMMAND ----------

df = spark.sql("""
SELECT
    id_cliente AS id_customer,
    periodo AS event_dt,
    numero_transacciones AS trx_cnt,
    monto_total_transacciones AS amt_trx,
    recencia_ultima_transaccion AS last_transaction_recall
FROM demo_db.transacciones
WHERE
    id_cliente IS NOT NULL
    AND id_cliente != '999999999'
""")
params = {
    "new_column_name": "release_dt",
    "date_column":{
        "name": "event_dt",
        "format": "yyyy-MM",
    },
    "days_to_add": 0,
    "months_to_add": 1,
    "years_to_add": 0,
}
df = shift_date(df, params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocesar texto

# COMMAND ----------

df = preprocessing_ingesting_tables(df)

# COMMAND ----------

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Generation

# COMMAND ----------

# MAGIC %md
# MAGIC **Funciones Auxiliares**
# MAGIC
# MAGIC Estas funciones auxiliares permiten realizar operaciones comunes como el cálculo de lags, estadísticas móviles, ratios, e indicadores técnicos.

# COMMAND ----------

# Función auxiliar para calcular lags
def lag_column(df: DataFrame, col: str, lag: int, window_spec: Window):
    """Calculate the lag value of a column using a specific window.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col (str): Column name to calculate lag for.
        lag (int): Number of periods to shift.
        window_spec (pyspark.sql.window.Window): Window specification.

    Returns:
        pyspark.sql.Column: Column with lag values.
    """
    return f.lag(f.col(col), lag).over(window_spec)

# Función genérica para calcular estadísticas móviles
def calculate_rolling_stat(df: DataFrame, col_name: str, window_spec: Window, stat_func, stat_name: str, window_size: int):
    """Calculate a rolling statistic over a specific window.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Column name.
        window_spec (pyspark.sql.window.Window): Window specification.
        stat_func: Statistical function to apply (f.avg, f.sum, etc.).
        stat_name (str): Name of the statistic.
        window_size (int): Size of the rolling window.

    Returns:
        pyspark.sql.Column: Column with calculated rolling statistic.
    """
    return stat_func(f.col(col_name)).over(window_spec.rowsBetween(-window_size + 1, 0))

# Función auxiliar para crear ratios
def ratio_column(df: DataFrame, numerator: str, denominator: str):
    """Calculate ratio between two columns handling division by zero.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        numerator (str): Numerator column name.
        denominator (str): Denominator column name.

    Returns:
        pyspark.sql.Column: Column with calculated ratio, 0 if denominator is zero.
    """
    return f.when(f.col(denominator) != 0, f.col(numerator) / f.col(denominator)).otherwise(f.lit(0))

# Función RSI (Relative Strength Index)
def calculate_rsi(df: DataFrame, col_name: str, window_spec: Window, window_size: int = 14):
    """Calculate the Relative Strength Index (RSI) of a numeric column.

    RSI is a technical indicator that measures the speed and change of price movements,
    oscillating between 0 and 100. Values above 70 are considered overbought and
    below 30 oversold.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Column name to calculate RSI for.
        window_spec (pyspark.sql.window.Window): Ordered window specification.
        window_size (int, optional): Period for RSI calculation. Defaults to 14.

    Returns:
        pyspark.sql.Column: Column with calculated RSI.

    Notes:
        - Requires at least `window_size + 1` records to calculate correctly.
        - Formula used: RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss.
    """
    # Calculamos los cambios (diferencias) entre las filas
    delta = f.col(col_name) - f.lag(f.col(col_name), 1).over(window_spec)

    # Calculamos ganancias y pérdidas
    gain = f.when(delta > 0, delta).otherwise(0)
    loss = f.abs(f.when(delta < 0, delta).otherwise(0))

    # Media móvil exponencial para ganancias y pérdidas
    avg_gain = f.avg(gain).over(window_spec.rowsBetween(-window_size + 1, 0))
    avg_loss = f.avg(loss).over(window_spec.rowsBetween(-window_size + 1, 0))

    # RSI: 100 - (100 / (1 + RS)), donde RS = avg_gain / avg_loss
    rs = avg_gain / f.coalesce(avg_loss, f.lit(1))
    rsi = 100 - (100 / (1 + rs))

    return rsi

# COMMAND ----------

# MAGIC %md
# MAGIC **Agregaciones Básicas**
# MAGIC
# MAGIC Calculamos las agregaciones básicas de las columnas numéricas y unimos el DataFrame de calificaciones con los datos principales.

# COMMAND ----------

def calculate_aggregations(
        df: DataFrame,
        groupby_columns: tp.List[str],
        numeric_ops: Optional[Dict[str, List[str]]] = None,
        pivot_column: Optional[str] = None,
        pivot_ops: Optional[tp.List[tp.Dict[str, str]]] = None,
        cats_to_pivot: Optional[tp.Dict[str, str]] = None,
        fillna_value: Optional[tp.Union[int, float, str]] = None,
    ) -> DataFrame:
    """Perform aggregations on a DataFrame with grouping, numeric operations, and optional pivoting.

    This advanced aggregation function supports both standard numeric aggregations and
    categorical pivoting operations, providing flexibility for complex feature engineering
    scenarios in transaction data analysis.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame to aggregate.
        groupby_columns (List[str]): Columns to group by for aggregation.
        numeric_ops (Dict[str, List[str]], optional): Dictionary mapping column names to
            list of aggregation operations. Example: {"amount": ["sum", "avg"]}.
        pivot_column (str, optional): Categorical column to pivot on.
        pivot_ops (List[Dict[str, str]], optional): List of pivot operations, each containing:
            &nbsp;- operation: Aggregation function name
            &nbsp;- column: Column to aggregate 
            &nbsp;- alias: Suffix for the resulting column name
        cats_to_pivot (Dict[str, str], optional): Dictionary mapping pivot values to
            column name prefixes. Example: {"online": "onl", "offline": "off"}.
        fillna_value (Union[int, float, str], optional): Value to fill nulls after aggregation.

    Returns:
        pyspark.sql.DataFrame: Aggregated DataFrame with numeric operations and pivot columns.

    Examples:
        ```python
        # Basic numeric aggregation
        numeric_ops = {"trx_cnt": ["sum"], "amt_trx": ["sum", "avg"]}
        result = calculate_aggregations(df, ["customer_id", "month"], numeric_ops=numeric_ops)

        # With pivoting
        pivot_ops = [{"operation": "sum", "column": "amount", "alias": "amt"}]
        cats_to_pivot = {"online": "onl", "store": "str"}
        result = calculate_aggregations(
            df, ["customer_id"],
            pivot_column="channel",
            pivot_ops=pivot_ops,
            cats_to_pivot=cats_to_pivot
        )

        # Expected output columns: customer_id, onl_amt, str_amt
        ```

    Raises:
        ValueError: If pivot_column, cats_to_pivot, and pivot_ops are inconsistently defined.

    Notes:
        - All pivot parameters must be provided together or all omitted.
        - 'LIT' operation creates indicator columns (binary 0/1) for categorical presence.
        - Pivot operations use CASE WHEN logic for conditional aggregation.
    """
    if not numeric_ops:
        numeric_ops = {}
    if not cats_to_pivot:
        cats_to_pivot = {}
    if not pivot_ops:
        pivot_ops = []

    # Validación de consistencia entre pivot_column, cats_to_pivot y pivot_ops
    pivot_vars = [pivot_column, cats_to_pivot, pivot_ops]
    empty_or_none = [v is None or (isinstance(v, (dict, list, str)) and len(v) == 0) for v in pivot_vars]
    if 0 < sum(empty_or_none) < len(pivot_vars):
        raise ValueError("`pivot_column`, `cats_to_pivot` and `pivot_ops` must all be defined or all empty/None.")

    df = df.groupBy(*groupby_columns).agg(
        # pivoteado
        *[
            f.expr(f"""
            {op['operation'] if op['operation'].upper()!='LIT' else 'SUM'}(
                CASE WHEN {pivot_column}='{value}' THEN {op['column']} ELSE 0 END
            ) { '>0' if op['operation'].upper()=='LIT' else ''}
            """).alias(
            ("ind_" if op['operation'].upper()=='LIT' else "") +
            f"{renaming}_{op['alias']}"
            )
            for op in pivot_ops
            for value, renaming in cats_to_pivot.items()
        ],
        # total
        *[
            f.expr(f"{op}({col})").alias(f"{col}_{op}_m")
            for col, ops in numeric_ops.items()
            for op in ops
        ],
        # f.count("*").alias("operation_cnt"),
    )

    if fillna_value is not None:
        df = df.na.fill(fillna_value)

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC **Lags y features basadas en lags**
# MAGIC
# MAGIC Para las columnas numéricas calculamos diferencias, tasas de cambio cambios porcentuales entre los valores actuales y los lags.
# MAGIC Para las columnas categóricas calculamos estabilidad, frecuencia de cambio y tendencias entre los valores actuales y los lags.

# COMMAND ----------

def calculate_lags_and_features(df: DataFrame, numeric_cols: List[str], lags: List[int], window_spec: Window) -> DataFrame:
    """Calculate lag-based features for numeric transaction columns.

    For each numeric column and each specified lag period, calculates:
    - Lag value (shifted)
    - Absolute difference (current value - lag value) 
    - Rate of change ((current value - lag) / lag)

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame with numeric transaction columns.
        numeric_cols (List[str]): List of numeric column names to process.
        lags (List[int]): List of lag periods to calculate (e.g., [1, 3, 6]).
        window_spec (pyspark.sql.window.Window): Window specification ordered by customer.

    Returns:
        pyspark.sql.DataFrame: Original DataFrame (no modifications, only calculates transformations).

    Notes:
        - Transformations are built but not applied in this function.
        - Transformations need to be applied later using withColumns.
        - Rate of change handles division by zero by returning NULL.

    Examples:
        ```python
        numeric_cols = ['trx_cnt_m']
        lags = [1, 3]
        window_spec = Window.partitionBy("id_customer").orderBy("release_dt")

        df_with_lags = calculate_lags_and_features(df, numeric_cols, lags, window_spec)

        # Generates transformations for:
        # - trx_cnt_m_lag_1m, trx_cnt_m_lag_3m
        # - trx_cnt_m_diff_1m, trx_cnt_m_diff_3m 
        # - trx_cnt_m_roc_1m, trx_cnt_m_roc_3m
        ```
    """
    transformations = []
    # Cálculos para columnas numéricas (montos)
    for col in numeric_cols:
        for lag in lags:
            lag_col_name = f"{col}_lag_{lag}m" # Columna de lag
            diff_col_name = f"{col}_diff_{lag}m" # Diferencias (valor actual - valor desplazado)
            roc_col_name = f"{col}_roc_{lag}m" # Tasa de cambio ((valor actual - lag) / lag)

            # Construir las transformaciones para todas las columnas necesarias
            transformations.extend([
                (lag_col_name, lag_column(df, col, lag, window_spec)),
                (diff_col_name, f.col(col) - f.col(lag_col_name)),
                (roc_col_name, f.round((f.col(col) - f.col(lag_col_name)) / f.col(lag_col_name), 4)),
            ])

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC **Estadísticas Móviles**
# MAGIC
# MAGIC Calculamos media, desviación estándar, mínimo y máximo en ventanas móviles.

# COMMAND ----------

# Función para aplicar rolling stats a columnas numéricas
def numeric_rolling_stats(df: DataFrame, col_name: str, window_spec: Window, window_sizes: List[int]) -> List:
    """Generate rolling statistics operations for a numeric transaction column.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Numeric column name.
        window_spec (pyspark.sql.window.Window): Window specification.
        window_sizes (List[int]): List of window sizes (e.g., [3, 6]).

    Returns:
        List[pyspark.sql.Column]: List of columns with rolling average and standard deviation.

    Examples:
        ```python
        operations = numeric_rolling_stats(df, 'trx_cnt_m', window_spec, [3, 6])
        # Generates: trx_cnt_m_ravg_3m, trx_cnt_m_rstd_3m,
        #           trx_cnt_m_ravg_6m, trx_cnt_m_rstd_6m
        ```
    """
    operations = list()
    for window in window_sizes:
        # Promedio y desviación estándar
        avg_col = f.round(
            calculate_rolling_stat(df, col_name, window_spec, f.avg, "avg", window),
            4
        ).alias(f"{col_name}_ravg_{window}m")
        std_col = f.round(
            calculate_rolling_stat(df, col_name, window_spec, f.stddev, "std", window),
            4
        ).alias(f"{col_name}_rstd_{window}m")
        operations.extend([avg_col, std_col])

    return operations

# Función para aplicar rolling stats a columnas categóricas
def categorical_rolling_stats(df: DataFrame, col_name: str, window_spec: Window, window_sizes: List[int]) -> List:
    """Generate rolling statistics operations for a categorical transaction column.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Categorical column name.
        window_spec (pyspark.sql.window.Window): Window specification.
        window_sizes (List[int]): List of window sizes (e.g., [3, 6]).

    Returns:
        List[pyspark.sql.Column]: List of columns with rolling minimum and maximum values.

    Examples:
        ```python
        operations = categorical_rolling_stats(df, 'channel_type', window_spec, [3, 6])
        # Generates: channel_type_rmin_3m, channel_type_rmax_3m,
        #           channel_type_rmin_6m, channel_type_rmax_6m
        ```
    """
    operations = list()
    for window in window_sizes:
        # Mínimo y máximo
        min_col = calculate_rolling_stat(df, col_name, window_spec, f.min, "min", window).alias(f"{col_name}_rmin_{window}m")
        max_col = calculate_rolling_stat(df, col_name, window_spec, f.max, "max", window).alias(f"{col_name}_rmax_{window}m")
        operations.extend([min_col, max_col])

    return operations

# Función principal para calcular estadísticas móviles
def calculate_rolling_statistics(df: DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                               window_sizes: List[int], window_spec: Window) -> DataFrame:
    """Calculate rolling statistics for numeric and categorical transaction columns.

    Applies specific rolling statistics based on column type:
    - Numeric: rolling average and standard deviation
    - Categorical: rolling minimum and maximum values

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame with transaction data.
        numeric_cols (List[str]): List of numeric columns to process.
        categorical_cols (List[str]): List of categorical columns to process.
        window_sizes (List[int]): Rolling window sizes (e.g., [3, 6]).
        window_spec (pyspark.sql.window.Window): Ordered window specification.

    Returns:
        pyspark.sql.DataFrame: DataFrame with all original columns plus new
            rolling statistics with nomenclature:
            &nbsp;- {col}_ravg_{window}m: Rolling average for numeric
            &nbsp;- {col}_rstd_{window}m: Rolling standard deviation for numeric 
            &nbsp;- {col}_rmin_{window}m: Rolling minimum for categorical
            &nbsp;- {col}_rmax_{window}m: Rolling maximum for categorical

    Examples:
        ```python
        numeric_cols = ['trx_cnt_m', 'amt_trx_sum_m']
        categorical_cols = ['channel_type']
        window_sizes = [3, 6]

        df_with_stats = calculate_rolling_statistics(df, numeric_cols, categorical_cols,
                                                   window_sizes, window_spec)
        # Adds 12 new columns: 4 per numeric col + 4 per categorical col
        ```
    """
    # Aplicar estadísticas móviles a columnas numéricas
    num_ops = []
    for col_name in numeric_cols:
        num_ops.extend(numeric_rolling_stats(df, col_name, window_spec, window_sizes))

    # Aplicar estadísticas móviles a columnas categóricas
    cat_ops = []
    for col_name in categorical_cols:
        cat_ops.extend(categorical_rolling_stats(df, col_name, window_spec, window_sizes))

    df = df.select("*", *num_ops, *cat_ops)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC **Ratios y Proporciones**
# MAGIC
# MAGIC Calculamos ratios entre variables clave, como monto_vencido / monto_vencer.

# COMMAND ----------

def calculate_percentages(df: DataFrame, percentages: List[Tuple[str, str]]) -> DataFrame:
    """Calculate percentages between transaction column pairs.

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame.
        percentages (List[Tuple[str, str]]): List of tuples (numerator, denominator).

    Returns:
        pyspark.sql.DataFrame: DataFrame with percentage columns added as {col1}_pct.
    """
    df = df.withColumns({
        f"{col1}_pct": f.when(f.col(col2) != 0, f.round((f.col(col1) / f.col(col2)) * 100, 4)).otherwise(None)
        for col1, col2 in percentages
    })
    return df

def calculate_ratios(df: DataFrame, ratios: List[Tuple[str, str]]) -> DataFrame:
    """Calculate ratios between transaction column pairs.

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame.
        ratios (List[Tuple[str, str]]): List of tuples (numerator, denominator).

    Returns:
        pyspark.sql.DataFrame: DataFrame with ratio columns added as {col1}_to_{col2}_ratio.
    """
    df = df.withColumns({
        f"{col1}_to_{col2}_ratio": f.when(f.col(col2) != 0, f.round(f.col(col1) / f.col(col2), 4)).otherwise(None)
        for col1, col2 in ratios
    })
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC **Integración de Módulos**
# MAGIC
# MAGIC Finalmente, integramos todos los módulos para construir la tabla de features.

# COMMAND ----------

def calculate_advanced_features(df: DataFrame) -> DataFrame:
    """Calculate advanced transaction features integrating multiple analytical techniques.

    Processes a transaction data DataFrame applying sequential transformations:
    1. Basic aggregations by customer and period with transaction metrics
    2. Transaction ratio calculations (amount per transaction)
    3. Lag-based features (1, 3, 6 months) 
    4. Rolling statistics (3 and 6 month windows)

    Args:
        df (pyspark.sql.DataFrame): DataFrame with transaction data that must contain:
            &nbsp;- id_customer: Unique customer identifier
            &nbsp;- release_dt: Period release date 
            &nbsp;- trx_cnt: Number of transactions
            &nbsp;- amt_trx: Total transaction amount
            &nbsp;- last_transaction_recall: Recency of last transaction

    Returns:
        pyspark.sql.DataFrame: DataFrame with advanced transaction features including:
            &nbsp;- trx_cnt_m: Monthly transaction count
            &nbsp;- amt_trx_sum_m: Monthly total transaction amount
            &nbsp;- last_transaction_recall_sum_m: Monthly transaction recency sum
            &nbsp;- amt_trx_sum_m_to_trx_cnt_m_ratio: Average amount per transaction
            &nbsp;- Rolling statistics for all numeric features (*_ravg_*m, *_rstd_*m)

    Examples:
        ```python
        # Input DataFrame
        df_input = spark.createDataFrame([
            ("1001", "2024-01", 15, 75000, 2),
            ("1001", "2024-02", 20, 100000, 1),
            ("1002", "2024-01", 8, 40000, 5)
        ], ["id_customer", "release_dt", "trx_cnt", "amt_trx", "last_transaction_recall"])

        features_df = calculate_advanced_features(df_input)

        # Expected output includes columns like:
        # - trx_cnt_m: 15, 20, 8
        # - amt_trx_sum_m: 75000, 100000, 40000 
        # - amt_trx_sum_m_to_trx_cnt_m_ratio: 5000.0, 5000.0, 5000.0
        # - trx_cnt_m_ravg_3m, amt_trx_sum_m_rstd_6m
        ```

    Notes:
        - Requires chronologically ordered data for accurate window calculations.
        - Rolling statistics need sufficient historical periods to be meaningful.
        - Ratio calculations automatically handle division by zero cases.
        - No categorical columns are processed in this transaction-focused pipeline.
    """
    window_spec = Window.partitionBy("id_customer").orderBy("release_dt")

    # Paso 1: Calculamos las agregaciones y pivots básicas agrupadas por `cod_cliente` y `periodo`
    numeric_ops = {
        "trx_cnt": ["sum"],
        "amt_trx": ["sum"],
        "last_transaction_recall": ["sum"],
    }
    df = calculate_aggregations(
        df,
        groupby_columns = ["id_customer", "release_dt"],
        numeric_ops = numeric_ops,
        fillna_value = 0,
    ).withColumnRenamed(
        "trx_cnt_sum_m", "trx_cnt_m"
    )

    # Paso 2: Calcular ratios
    ratios = [
        ("amt_trx_sum_m", "trx_cnt_m"), # Proporcion entre el monto total transaccionado y la cantidad de transacciones del mes
    ]
    df = calculate_ratios(df, ratios)

    numeric_cols = [
        "trx_cnt_m", # cantidad de transacciones en el mes
        "amt_trx_sum_m", # monto total transaccionado en el mes
        "last_transaction_recall_sum_m", # monto total de recencia_ultima_transaccion
        "amt_trx_sum_m_to_trx_cnt_m_ratio", # ratio entre (monto transaccionado / cantidad de transacciones)
    ]
    categorical_cols = []

    # Paso 3: Aplicamos las operaciones de ventana (lags y diferencias)
    lags = [1, 3, 6]
    df = calculate_lags_and_features(df, numeric_cols, lags, window_spec)

    # Paso 4: Calculamos las estadísticas móviles
    window_sizes = [3, 6]  # Definimos los tamaños de ventana móviles
    df = calculate_rolling_statistics(df, numeric_cols, categorical_cols, window_sizes, window_spec)


    return df

# COMMAND ----------

def feature_output_standarize(df: DataFrame, pks_mapper: Dict[str, str]) -> DataFrame:
    """Standardize transaction feature output format applying feature store conventions.

    Selects specific columns, applies primary key renaming, adds prefixes
    to features and normalizes data types for feature store compatibility.

    Args:
        df (pyspark.sql.DataFrame): DataFrame with calculated transaction features that must contain
            expected transaction columns.
        pks_mapper (Dict[str, str]): Dictionary mapping for renaming primary keys.
            Example: {"id_customer": "pk_customer", "release_dt": "tpk_release_dt"}

    Returns:
        pyspark.sql.DataFrame: Standardized DataFrame with:
            &nbsp;- Selected columns in specific order
            &nbsp;- Primary keys renamed according to pks_mapper
            &nbsp;- "ctrx_" prefix added to all features (except primary keys)
            &nbsp;- DECIMAL types converted to DOUBLE 
            &nbsp;- LONG types converted to INTEGER

    Examples:
        ```python
        pks_mapper = {
            "id_customer": "pk_customer",
            "release_dt": "tpk_release_dt"
        }

        standardized_df = feature_output_standarize(features_df, pks_mapper)

        # Expected output:
        # Columns: pk_customer, tpk_release_dt, ctrx_trx_cnt_m, ctrx_amt_trx_sum_m, ...
        ```

    Notes:
        - Uses auxiliary functions decimals_to_floats() and longs_to_integers().
        - The "ctrx_" prefix identifies customer transaction features.
        - Selects a specific subset of 22 feature columns.

    Raises:
        AttributeError: If DataFrame doesn't contain expected columns.
    """
    df = df.select(
        "id_customer",
        "release_dt",
        "trx_cnt_m",
        "amt_trx_sum_m",
        "last_transaction_recall_sum_m",
        "amt_trx_sum_m_to_trx_cnt_m_ratio",
        "trx_cnt_m_ravg_3m",
        "trx_cnt_m_rstd_3m",
        "trx_cnt_m_ravg_6m",
        "trx_cnt_m_rstd_6m",
        "amt_trx_sum_m_ravg_3m",
        "amt_trx_sum_m_rstd_3m",
        "amt_trx_sum_m_ravg_6m",
        "amt_trx_sum_m_rstd_6m",
        "last_transaction_recall_sum_m_ravg_3m",
        "last_transaction_recall_sum_m_rstd_3m",
        "last_transaction_recall_sum_m_ravg_6m",
        "last_transaction_recall_sum_m_rstd_6m",
        "amt_trx_sum_m_to_trx_cnt_m_ratio_ravg_3m",
        "amt_trx_sum_m_to_trx_cnt_m_ratio_rstd_3m",
        "amt_trx_sum_m_to_trx_cnt_m_ratio_ravg_6m",
        "amt_trx_sum_m_to_trx_cnt_m_ratio_rstd_6m",
    )

    # Define primary keys
    df = df.withColumnsRenamed(pks_mapper)

    # Añade prefijo por feature table
    pks = list(pks_mapper.values())
    features_prefix = "ctrx_"
    dict_for_renames = {col: f"{features_prefix}{col}" for col in df.columns if col not in pks }
    df = df.withColumnsRenamed(dict_for_renames)

    # DECIMAL -> DOUBLE
    df = decimals_to_floats(df)

    # LONG -> INT
    df = longs_to_integers(df)

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Excecute Feature

# COMMAND ----------

features_transactions = calculate_advanced_features(df)

# COMMAND ----------

pks_mapper = {
    "id_customer": "pk_customer",
    "release_dt": "tpk_release_dt",
}
features_transactions = feature_output_standarize(features_transactions, pks_mapper)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

features_transactions.cache()

# COMMAND ----------

database = fs_database
table = 'fs_cus_transactions'
path_name = fs_base_path + f"/{table}"

entity = "customer"
table_description = "transactions"

# COMMAND ----------

execute_saving = True

# COMMAND ----------

force_overwrite = notebook_inputs.get("force_overwrite", "false").lower() == "true"
overwriteSchema = notebook_inputs.get("overwriteSchema", "false").lower() == "true"
omit_data_validation_errors = notebook_inputs.get("omit_data_validation_errors", "false").lower() == "true"

successfully_saved = False
if execute_saving:
    fsm = FeatureStoreManager(
        spark=spark,
        fs=FeatureStoreClient(),
        table_name=table,
        database=database,
        path_name=path_name,
        source_tables=source_tables,
        primary_keys=['pk_customer', 'tpk_release_dt'],
        timestamp_keys=['tpk_release_dt'],
        entity=entity,
        table_description=table_description,
    )
    fsm.save(
        df=features_transactions,
        force_overwrite=force_overwrite,
        overwriteSchema=overwriteSchema,
        omit_data_validation_errors=omit_data_validation_errors,
    )
    successfully_saved = True

# COMMAND ----------

spark.catalog.clearCache()
features_transactions.unpersist(True)

del features_transactions
gc.collect()

# COMMAND ----------

if not successfully_saved:
    failure_reason_message = "\n\t- " + "\n\t- ".join(failure_reason.split("\n"))
    message = f"Error saving Feature Table '{database}.{table}':" + failure_reason_message
    raise Exception(message)
