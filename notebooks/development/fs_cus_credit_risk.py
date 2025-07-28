# Databricks notebook source
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from typing import List, Dict, Tuple
import gc
from datetime import datetime

# COMMAND ----------

# MAGIC %run /Shared/databricks-demo-feature-store/notebooks/utils $env="dev"

# COMMAND ----------

notebook_inputs = dict(dbutils.widgets.getAll())

# COMMAND ----------

source_tables = [
    "demo_db.buro_credito",
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
# MAGIC # Feature `fs_cus_credit_risk`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingesta

# COMMAND ----------

spark.table("demo_db.buro_credito").schema.fields

# COMMAND ----------

df = spark.sql("""
SELECT
    id_cliente AS id_customer,
    periodo AS event_dt,
    -- calificacion_sistema AS rating,
    buro_score AS bureau_score,
    (
        CASE
            WHEN calificacion_sistema = 'A' THEN 1
            WHEN calificacion_sistema = 'B' THEN 2
            WHEN calificacion_sistema = 'C' THEN 3
            WHEN calificacion_sistema = 'D' THEN 4
            WHEN calificacion_sistema = 'E' THEN 5
            ELSE 0
        END
    ) AS rating_num, -- Mapeo de calificación a número
    consultas_buro_12m AS bureau_inquiries_12m,
    deuda_total_bancos AS total_banking_debt
FROM demo_db.buro_credito
WHERE
    id_cliente IS NOT NULL
    AND id_cliente != '999999999'
    AND buro_score IS NOT NULL
    AND calificacion_sistema IS NOT NULL
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
        pyspark.sql.Column: Column with calculated RSI, rounded to 4 decimal places.

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
    rsi = f.round(100 - (100 / (1 + rs)), 4)

    return rsi

# COMMAND ----------

# MAGIC %md
# MAGIC **Agregaciones Básicas**
# MAGIC
# MAGIC Calculamos las agregaciones básicas de las columnas numéricas y unimos el DataFrame de calificaciones con los datos principales.

# COMMAND ----------

def calculate_aggregations(df: DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> DataFrame:
    """Calculate basic aggregations grouped by customer and release period.

    Performs specific aggregations for numeric columns (sum) and categorical columns (maximum),
    plus counts total credits per customer and period.

    Args:
        df (pyspark.sql.DataFrame): DataFrame with credit data that must contain
            'id_customer' and 'release_dt' columns.
        numeric_cols (List[str]): List of numeric column names to aggregate.
        categorical_cols (List[str]): List of categorical column names to aggregate.

    Returns:
        pyspark.sql.DataFrame: Grouped DataFrame with the following columns:
            &nbsp;- id_customer: Customer identifier.
            &nbsp;- release_dt: Release date.
            &nbsp;- credits_cnt: Total count of credits.
            &nbsp;- {col}_sum: Sum of each numeric column.
            &nbsp;- {col}: Maximum value of each categorical column.

    Examples:
        ```python
        numeric_cols = ['bureau_score', 'total_debt']
        categorical_cols = ['rating']
        result = calculate_aggregations(df, numeric_cols, categorical_cols)

        # Expected output:
        # +----------+----------+-----------+----------------+----------+------+
        # |id_customer|release_dt|credits_cnt|bureau_score_sum|total_debt_sum|rating|
        # +----------+----------+-----------+----------------+----------+------+
        # |    1001  | 2024-01  |     3     |       660      |   150000 |  A   |
        # +----------+----------+-----------+----------------+----------+------+
        ```
    """
    agg_exprs = []

    # Total de creditos
    agg_exprs.append(f.count("*").alias("credits_cnt"))

    # Agregaciones para columnas numéricas
    for col in numeric_cols:
        agg_exprs.append(f.sum(col).alias(f"{col}_sum")) # Total

    # Agregaciones para columnas categóricas
    for col in categorical_cols:
        agg_exprs.append(f.max(col).alias(f"{col}"))

    # Agrupar por cliente y período
    return df.groupBy("id_customer", "release_dt").agg(*agg_exprs)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lags y features basadas en lags**
# MAGIC
# MAGIC Para las columnas numéricas calculamos diferencias, tasas de cambio cambios porcentuales entre los valores actuales y los lags.
# MAGIC Para las columnas categóricas calculamos estabilidad, frecuencia de cambio y tendencias entre los valores actuales y los lags.

# COMMAND ----------

def calculate_lags_and_features(df: DataFrame, numeric_cols: List[str], lags: List[int], window_spec: Window) -> DataFrame:
    """Calculate lag-based features for numeric columns.

    For each numeric column and each specified lag period, calculates:
    - Lag value (shifted)
    - Absolute difference (current value - lag value) 
    - Rate of change ((current value - lag) / lag)

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame with numeric columns.
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
        numeric_cols = ['bureau_score_sum']
        lags = [1, 3]
        window_spec = Window.partitionBy("id_customer").orderBy("release_dt")

        df_with_lags = calculate_lags_and_features(df, numeric_cols, lags, window_spec)

        # Generates transformations for:
        # - bureau_score_sum_lag_1m, bureau_score_sum_lag_3m
        # - bureau_score_sum_diff_1m, bureau_score_sum_diff_3m 
        # - bureau_score_sum_roc_1m, bureau_score_sum_roc_3m
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
    """Generate rolling statistics operations for a numeric column.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Numeric column name.
        window_spec (pyspark.sql.window.Window): Window specification.
        window_sizes (List[int]): List of window sizes (e.g., [3, 6]).

    Returns:
        List[pyspark.sql.Column]: List of columns with rolling average and standard deviation.

    Examples:
        ```python
        operations = numeric_rolling_stats(df, 'bureau_score_sum', window_spec, [3, 6])
        # Generates: bureau_score_sum_ravg_3m, bureau_score_sum_rstd_3m,
        #           bureau_score_sum_ravg_6m, bureau_score_sum_rstd_6m
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
    """Generate rolling statistics operations for a categorical column.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Categorical column name.
        window_spec (pyspark.sql.window.Window): Window specification.
        window_sizes (List[int]): List of window sizes (e.g., [3, 6]).

    Returns:
        List[pyspark.sql.Column]: List of columns with rolling minimum and maximum values.

    Examples:
        ```python
        operations = categorical_rolling_stats(df, 'rating_num', window_spec, [3, 6])
        # Generates: rating_num_rmin_3m, rating_num_rmax_3m,
        #           rating_num_rmin_6m, rating_num_rmax_6m
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
    """Calculate rolling statistics for numeric and categorical columns.

    Applies specific rolling statistics based on column type:
    - Numeric: rolling average and standard deviation
    - Categorical: rolling minimum and maximum values

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame with data.
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
        numeric_cols = ['bureau_score_sum', 'total_debt_sum']
        categorical_cols = ['rating_num']
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
    """Calculate percentages between column pairs.

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
    """Calculate ratios between column pairs.

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
    """Calculate advanced credit risk features integrating multiple analytical techniques.

    Processes a credit data DataFrame applying sequential transformations:
    1. Basic aggregations by customer and period
    2. Lag-based features (1, 3, 6 months) 
    3. Rolling statistics (3 and 6 month windows)
    4. Financial ratios between key variables

    Args:
        df (pyspark.sql.DataFrame): DataFrame with credit data that must contain:
            &nbsp;- id_customer: Unique customer identifier
            &nbsp;- release_dt: Period release date 
            &nbsp;- bureau_score: Credit bureau score
            &nbsp;- total_banking_debt: Total banking debt
            &nbsp;- bureau_inquiries_12m: Bureau inquiries in 12 months
            &nbsp;- rating_num: Numeric risk rating

    Returns:
        pyspark.sql.DataFrame: DataFrame with advanced features including:
            &nbsp;- All basic aggregation columns (*_sum, credits_cnt)
            &nbsp;- Rolling statistics (*_ravg_*m, *_rstd_*m, *_rmin_*m, *_rmax_*m)
            &nbsp;- Financial ratios (*_to_*_ratio)
            &nbsp;- Time series derived variables

    Examples:
        ```python
        # Input DataFrame
        df_input = spark.createDataFrame([
            ("1001", "2024-01", 750, 850, 3, 2),
            ("1001", "2024-02", 760, 850, 2, 2),
            ("1002", "2024-01", 680, 915, 5, 3)
        ], ["id_customer", "release_dt", "bureau_score", "total_banking_debt",
            "bureau_inquiries_12m", "rating_num"])

        features_df = calculate_advanced_features(df_input)

        # Expected output includes columns like:
        # - bureau_score_sum, total_banking_debt_sum
        # - bureau_score_sum_ravg_3m, bureau_score_sum_rstd_6m 
        # - total_banking_debt_sum_to_bureau_score_sum_ratio
        ```

    Notes:
        - Requires chronologically ordered data for accurate window calculations.
        - Rolling statistics need sufficient historical periods to be meaningful.
        - Ratios automatically handle division by zero cases by returning None.
    """
    window_spec = Window.partitionBy("id_customer").orderBy("release_dt")

    numeric_cols = [
        'bureau_score',
        'total_banking_debt',
        'bureau_inquiries_12m',
    ]
    categorical_cols = [
        'rating_num', # A partir de ahora se entiende como peor calificacion
    ]

    # Paso 1: Calculamos las agregaciones básicas agrupadas por `cod_cliente` y `periodo`
    df = calculate_aggregations(df, numeric_cols, categorical_cols)

    numeric_cols = [
        'bureau_score_sum',
        'total_banking_debt_sum',
        'bureau_inquiries_12m_sum',
    ]

    # Paso 2: Aplicamos las operaciones de ventana (lags y diferencias)
    lags = [1, 3, 6]
    df = calculate_lags_and_features(df, numeric_cols, lags, window_spec)

    numeric_cols = [
        'bureau_score_sum',
        'total_banking_debt_sum',
        'bureau_inquiries_12m_sum',
    ]

    # # Paso 3: Calculamos las estadísticas móviles
    window_sizes = [3, 6]  # Definimos los tamaños de ventana móviles
    df = calculate_rolling_statistics(df, numeric_cols, categorical_cols, window_sizes, window_spec)

    # Paso 4: Calculamos ratios
    ratios = [
        ('total_banking_debt_sum', 'bureau_score_sum'),       # Ratio entre total de deuda y score
    ]
    df = calculate_ratios(df, ratios)

    return df

# COMMAND ----------

def feature_output_standarize(df: DataFrame, pks_mapper: Dict[str, str]) -> DataFrame:
    """Standardize feature output format applying feature store conventions.

    Selects specific columns, applies primary key renaming, adds prefixes
    to features and normalizes data types for feature store compatibility.

    Args:
        df (pyspark.sql.DataFrame): DataFrame with calculated features that must contain
            expected credit risk columns.
        pks_mapper (Dict[str, str]): Dictionary mapping for renaming primary keys.
            Example: {"id_customer": "pk_customer", "release_dt": "tpk_release_dt"}

    Returns:
        pyspark.sql.DataFrame: Standardized DataFrame with:
            &nbsp;- Selected columns in specific order
            &nbsp;- Primary keys renamed according to pks_mapper
            &nbsp;- "chr_" prefix added to all features (except primary keys)
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
        # Columns: pk_customer, tpk_release_dt, chr_credits_cnt, chr_bureau_score_sum, ...
        ```

    Notes:
        - Uses auxiliary functions decimals_to_floats() and longs_to_integers().
        - The "chr_" prefix identifies credit risk features.
        - Selects a specific subset of 25 feature columns.

    Raises:
        AttributeError: If DataFrame doesn't contain expected columns.
    """
    df = df.select(
        "id_customer",
        "release_dt",
        "credits_cnt",
        "bureau_score_sum",
        "total_banking_debt_sum",
        "bureau_inquiries_12m_sum",
        "rating_num",
        "bureau_score_sum_ravg_3m",
        "bureau_score_sum_rstd_3m",
        "bureau_score_sum_ravg_6m",
        "bureau_score_sum_rstd_6m",
        "total_banking_debt_sum_ravg_3m",
        "total_banking_debt_sum_rstd_3m",
        "total_banking_debt_sum_ravg_6m",
        "total_banking_debt_sum_rstd_6m",
        "bureau_inquiries_12m_sum_ravg_3m",
        "bureau_inquiries_12m_sum_rstd_3m",
        "bureau_inquiries_12m_sum_ravg_6m",
        "bureau_inquiries_12m_sum_rstd_6m",
        "rating_num_rmin_3m",
        "rating_num_rmax_3m",
        "rating_num_rmin_6m",
        "rating_num_rmax_6m",
        "total_banking_debt_sum_to_bureau_score_sum_ratio",
    )

    # Define primary keys
    df = df.withColumnsRenamed(pks_mapper)

    # Añade prefijo por feature table
    pks = list(pks_mapper.values())
    features_prefix = "chr_"
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

features_external_risk = calculate_advanced_features(df)

# COMMAND ----------

pks_mapper = {
    "id_customer": "pk_customer",
    "release_dt": "tpk_release_dt",
}
features_external_risk = feature_output_standarize(features_external_risk, pks_mapper)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

features_external_risk.cache()

# COMMAND ----------

database = fs_database
table = 'fs_cus_credit_risk'
path_name = fs_base_path + f"/{table}"

entity = "customer"
table_description = "credit risk"

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
        df=features_external_risk,
        force_overwrite=force_overwrite,
        overwriteSchema=overwriteSchema,
        omit_data_validation_errors=omit_data_validation_errors,
    )
    successfully_saved = True

# COMMAND ----------

spark.catalog.clearCache()
features_external_risk.unpersist(True)

del features_external_risk
gc.collect()

# COMMAND ----------

if not successfully_saved:
    failure_reason_message = "\n\t- " + "\n\t- ".join(failure_reason.split("\n"))
    message = f"Error saving Feature Table '{database}.{table}':" + failure_reason_message
    raise Exception(message)
