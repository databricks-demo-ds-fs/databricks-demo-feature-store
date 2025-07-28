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
    "demo_db.productos",
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
# MAGIC # Feature `fs_cus_holding_products`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingesta

# COMMAND ----------

df = spark.sql("""
SELECT
    id_cliente AS id_customer,
    fecha AS event_dt,
    producto AS product,
    monto AS amount,
    total_productos AS products_cnt
FROM demo_db.productos
WHERE
    id_cliente IS NOT NULL
    AND id_cliente != '999999999'
    AND monto IS NOT NULL
""")
params = {
    "new_column_name": "release_dt",
    "date_column":{
        "name": "event_dt",
        "format": "yyyy-MM-dd",
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
def calculate_rolling_stat(
        df: DataFrame, col_name: str,
        window_spec: Window, stat_func, stat_name: str, window_size: int
    ):
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
    scenarios in product holding analysis.

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
            column name prefixes. Example: {"credito_vehicular": "vehicle_loan"}.
        fillna_value (Union[int, float, str], optional): Value to fill nulls after aggregation.

    Returns:
        pyspark.sql.DataFrame: Aggregated DataFrame with numeric operations and pivot columns.

    Examples:
        ```python
        # Product holding with pivoting
        cats_to_pivot = {"credito_vehicular": "vehicle_loan", "tarjeta_de_credito": "credit_card"}
        pivot_ops = [
            {"operation": "sum", "column": "amount", "alias": "amount_sum_m"},
            {"operation": "LIT", "column": 1, "alias": "product"}
        ]
        numeric_ops = {"products_cnt": ["max"]}

        result = calculate_aggregations(
            df, ["customer_id", "month"],
            numeric_ops=numeric_ops,
            pivot_column="product",
            pivot_ops=pivot_ops,
            cats_to_pivot=cats_to_pivot
        )

        # Expected output columns:
        # customer_id, month, vehicle_loan_amount_sum_m, credit_card_amount_sum_m,
        # ind_vehicle_loan_product, ind_credit_card_product, products_cnt_max_m
        ```

    Raises:
        ValueError: If pivot_column, cats_to_pivot, and pivot_ops are inconsistently defined.

    Notes:
        - All pivot parameters must be provided together or all omitted.
        - 'LIT' operation creates indicator columns (binary 0/1) for product holding.
        - Pivot operations use CASE WHEN logic for conditional aggregation.
        - Useful for creating product-specific features from categorical product data.
    """
    if not numeric_ops:
        numeric_ops = {}
    if not cats_to_pivot:
        cats_to_pivot = {}
    if not pivot_ops:
        pivot_ops = []

    # Validación de consistencia entre pivot_column, cats_to_pivot y pivot_ops
    pivot_vars = [pivot_column, cats_to_pivot, pivot_ops]
    empty_or_none = [
        v is None or (isinstance(v, (dict, list, str)) and len(v) == 0)
        for v in pivot_vars
    ]
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

def calculate_lags_and_features(
        df: DataFrame, numeric_cols: List[str],
        lags: List[int], window_spec: Window
    ) -> DataFrame:
    """Calculate lag-based features for numeric product holding columns.

    For each numeric column and each specified lag period, calculates:
    - Lag value (shifted)
    - Absolute difference (current value - lag value) 
    - Rate of change ((current value - lag) / lag)

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame with numeric product holding columns.
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
        numeric_cols = ['vehicle_loan_amount_sum_m']
        lags = [1, 3]
        window_spec = Window.partitionBy("id_customer").orderBy("release_dt")

        df_with_lags = calculate_lags_and_features(df, numeric_cols, lags, window_spec)

        # Generates transformations for:
        # - vehicle_loan_amount_sum_m_lag_1m, vehicle_loan_amount_sum_m_lag_3m
        # - vehicle_loan_amount_sum_m_diff_1m, vehicle_loan_amount_sum_m_diff_3m 
        # - vehicle_loan_amount_sum_m_roc_1m, vehicle_loan_amount_sum_m_roc_3m
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
def numeric_rolling_stats(
        df: DataFrame, col_name: str,
        window_spec: Window, window_sizes: List[int]
    ) -> List:
    """Generate rolling statistics operations for a numeric product holding column.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Numeric column name.
        window_spec (pyspark.sql.window.Window): Window specification.
        window_sizes (List[int]): List of window sizes (e.g., [3, 6]).

    Returns:
        List[pyspark.sql.Column]: List of columns with rolling average and standard deviation.

    Examples:
        ```python
        operations = numeric_rolling_stats(df, 'vehicle_loan_amount_sum_m', window_spec, [3, 6])
        # Generates: vehicle_loan_amount_sum_m_ravg_3m, vehicle_loan_amount_sum_m_rstd_3m,
        #           vehicle_loan_amount_sum_m_ravg_6m, vehicle_loan_amount_sum_m_rstd_6m
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
def categorical_rolling_stats(
        df: DataFrame, col_name: str,
        window_spec: Window, window_sizes: List[int]
    ) -> List:
    """Generate rolling statistics operations for a categorical product holding column.

    Args:
        df (pyspark.sql.DataFrame): PySpark DataFrame.
        col_name (str): Categorical column name.
        window_spec (pyspark.sql.window.Window): Window specification.
        window_sizes (List[int]): List of window sizes (e.g., [3, 6]).

    Returns:
        List[pyspark.sql.Column]: List of columns with rolling minimum and maximum values.

    Examples:
        ```python
        operations = categorical_rolling_stats(df, 'ind_vehicle_loan_product', window_spec, [3, 6])
        # Generates: ind_vehicle_loan_product_rmin_3m, ind_vehicle_loan_product_rmax_3m,
        #           ind_vehicle_loan_product_rmin_6m, ind_vehicle_loan_product_rmax_6m
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
def calculate_rolling_statistics(
        df: DataFrame, numeric_cols: List[str], categorical_cols: List[str],
        window_sizes: List[int], window_spec: Window
    ) -> DataFrame:
    """Calculate rolling statistics for numeric and categorical product holding columns.

    Applies specific rolling statistics based on column type:
    - Numeric: rolling average and standard deviation
    - Categorical: rolling minimum and maximum values

    Args:
        df (pyspark.sql.DataFrame): Base DataFrame with product holding data.
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
        numeric_cols = ['vehicle_loan_amount_sum_m', 'amount_sum_m']
        categorical_cols = ['ind_vehicle_loan_product', 'products_cnt_m']
        window_sizes = [3, 6]

        df_with_stats = calculate_rolling_statistics(df, numeric_cols, categorical_cols,
                                                   window_sizes, window_spec)
        # Adds rolling statistics for product holding patterns over time
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
    """Calculate percentages between product holding column pairs.

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
    """Calculate ratios between product holding column pairs.

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
    """Calculate advanced product holding features integrating multiple analytical techniques.

    Processes a product holding data DataFrame applying sequential transformations:
    1. Product-specific aggregations with pivoting by product type
    2. Lag-based features (1, 3, 6 months) for product amounts
    3. Rolling statistics (3 and 6 month windows) for holding patterns
    4. Product holding indicators and amount statistics

    Args:
        df (pyspark.sql.DataFrame): DataFrame with product holding data that must contain:
            &nbsp;- id_customer: Unique customer identifier
            &nbsp;- release_dt: Period release date 
            &nbsp;- product: Product type (credito_vehicular, credito_hipotecario, etc.)
            &nbsp;- amount: Product amount/balance
            &nbsp;- products_cnt: Total product count

    Returns:
        pyspark.sql.DataFrame: DataFrame with advanced product holding features including:
            &nbsp;- Product-specific amounts (vehicle_loan_amount_sum_m, mortgage_loan_amount_sum_m, etc.)
            &nbsp;- Product holding indicators (ind_vehicle_loan_product, ind_mortgage_loan_product, etc.)
            &nbsp;- Aggregate amount statistics (amount_sum_m, amount_avg_m, amount_min_m, amount_max_m)
            &nbsp;- Products count (products_cnt_m)
            &nbsp;- Rolling statistics for all numeric and categorical features

    Examples:
        ```python
        # Input DataFrame
        df_input = spark.createDataFrame([
            ("1001", "2024-01", "credito_vehicular", 50000, 2),
            ("1001", "2024-01", "tarjeta_de_credito", 15000, 2),
            ("1002", "2024-01", "credito_hipotecario", 200000, 1)
        ], ["id_customer", "release_dt", "product", "amount", "products_cnt"])

        features_df = calculate_advanced_features(df_input)

        # Expected output includes columns like:
        # - vehicle_loan_amount_sum_m: 50000, 0
        # - credit_card_amount_sum_m: 15000, 0 
        # - mortgage_loan_amount_sum_m: 0, 200000
        # - ind_vehicle_loan_product: 1, 0
        # - amount_sum_m: 65000, 200000
        # - products_cnt_m: 2, 1
        # - Plus rolling statistics for temporal analysis
        ```

    Notes:
        - Requires chronologically ordered data for accurate window calculations.
        - Rolling statistics capture product holding stability and changes over time.
        - Product types are mapped to English names for consistency.
        - No percentage calculations are applied in the current implementation.
    """
    window_spec = Window.partitionBy("id_customer").orderBy("release_dt")

    # Paso 1: Calculamos las agregaciones y pivots básicas agrupadas por `cod_cliente` y `periodo`
    cats_to_pivot = {
        "credito_vehicular": "vehicle_loan",
        "credito_hipotecario": "mortgage_loan",
        "tarjeta_de_credito": "credit_card",
        "credito_productivo": "productive_credit",
    }
    pivot_ops = [
        {
            "column": "amount",
            "operation": "SUM",
            "alias": "amount_sum_m",
        },
        { # Para tenencia
            "column": 1,
            "operation": "LIT", #se termina aplicando: SUM(CASE WHEN ... THEN 1 ELSE 0) > 0
            "alias": "product",
        },
    ]
    numeric_ops = {
        "products_cnt": ["max"],
        "amount": ["sum", "avg", "min", "max"],
    }
    df = calculate_aggregations(
        df,
        groupby_columns = ["id_customer", "release_dt"],
        numeric_ops = numeric_ops,
        pivot_column = "product",
        pivot_ops = pivot_ops,
        cats_to_pivot = cats_to_pivot,
        fillna_value = 0,
    ).withColumnRenamed(
        "products_cnt_max_m", "products_cnt_m"
    )

    numeric_cols = [
        "vehicle_loan_amount_sum_m", # monto total de los credito_vehicular del cliente en el mes
        "mortgage_loan_amount_sum_m", #  monto total de los credito_hipotecario del cliente en el mes
        "credit_card_amount_sum_m", #  monto total de los tarjeta_de_credito del cliente en el mes
        "productive_credit_amount_sum_m", #  monto total de los credito_productivo del cliente en el mes
        "amount_sum_m", # suma del monto de todos los productos
        "amount_avg_m", # promedio del monto de todos los productos
        "amount_min_m", # el monto minimo entre todos los productos
        "amount_max_m", # el monto máximo entre todos los productos
    ]
    categorical_cols = [
        "ind_vehicle_loan_product", # tiene credito_vehicular en el mes?
        "ind_mortgage_loan_product", # tiene credito_hipotecario en el mes?
        "ind_credit_card_product", # tiene tarjeta_de_credito en el mes?
        "ind_productive_credit_product", # tiene credito_productivo en el mes?
        "products_cnt_m", # cantidad de productos que tiene el cliente en dicho mes
    ]

    # Paso 2: Aplicamos las operaciones de ventana (lags y diferencias)
    lags = [1, 3, 6]
    df = calculate_lags_and_features(df, numeric_cols, lags, window_spec)


    # Paso 3: Calculamos las estadísticas móviles
    window_sizes = [3, 6]  # Definimos los tamaños de ventana móviles
    df = calculate_rolling_statistics(df, numeric_cols, categorical_cols, window_sizes, window_spec)

    # # Calcular proporciones
    # percentages = [
    #     ("amt_paid_sum", "amt_total_credit_sum"),                 # Proporcion entre el monto pagado y el total del credito
    #     ("amt_total_balance_sum", "amt_total_credit_sum"),        # Proporcion entre el saldo pendiente y el total del credito
    #     ('time_to_close_in_months', 'credit_duration_in_months'), # Proporción entre tiempo para que se cierre el credito y duracion total
    # ]
    # df = calculate_percentages(df, percentages)

    return df

# COMMAND ----------

def feature_output_standarize(df: DataFrame, pks_mapper: Dict[str, str]) -> DataFrame:
    """Standardize product holding feature output format applying feature store conventions.

    Selects specific columns, applies primary key renaming, adds prefixes
    to features and normalizes data types for feature store compatibility.

    Args:
        df (pyspark.sql.DataFrame): DataFrame with calculated product holding features that must contain
            expected product holding columns.
        pks_mapper (Dict[str, str]): Dictionary mapping for renaming primary keys.
            Example: {"id_customer": "pk_customer", "release_dt": "tpk_release_dt"}

    Returns:
        pyspark.sql.DataFrame: Standardized DataFrame with:
            &nbsp;- Selected columns in specific order
            &nbsp;- Primary keys renamed according to pks_mapper
            &nbsp;- "chp_" prefix added to all features (except primary keys)
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
        # Columns: pk_customer, tpk_release_dt, chp_vehicle_loan_amount_sum_m,
        #          chp_ind_vehicle_loan_product, chp_products_cnt_m, ...
        ```

    Notes:
        - Uses auxiliary functions decimals_to_floats() and longs_to_integers().
        - The "chp_" prefix identifies customer holding products features.
        - Selects a comprehensive set of 63 product holding feature columns.
        - Includes both product-specific and aggregate features with rolling statistics.

    Raises:
        AttributeError: If DataFrame doesn't contain expected columns.
    """
    df = df.select(
        "id_customer",
        "release_dt",
        "vehicle_loan_amount_sum_m",
        "mortgage_loan_amount_sum_m",
        "credit_card_amount_sum_m",
        "productive_credit_amount_sum_m",
        "ind_vehicle_loan_product",
        "ind_mortgage_loan_product",
        "ind_credit_card_product",
        "ind_productive_credit_product",
        "products_cnt_m",
        "amount_sum_m",
        "amount_avg_m",
        "amount_min_m",
        "amount_max_m",
        "vehicle_loan_amount_sum_m_ravg_3m",
        "vehicle_loan_amount_sum_m_rstd_3m",
        "vehicle_loan_amount_sum_m_ravg_6m",
        "vehicle_loan_amount_sum_m_rstd_6m",
        "mortgage_loan_amount_sum_m_ravg_3m",
        "mortgage_loan_amount_sum_m_rstd_3m",
        "mortgage_loan_amount_sum_m_ravg_6m",
        "mortgage_loan_amount_sum_m_rstd_6m",
        "credit_card_amount_sum_m_ravg_3m",
        "credit_card_amount_sum_m_rstd_3m",
        "credit_card_amount_sum_m_ravg_6m",
        "credit_card_amount_sum_m_rstd_6m",
        "productive_credit_amount_sum_m_ravg_3m",
        "productive_credit_amount_sum_m_rstd_3m",
        "productive_credit_amount_sum_m_ravg_6m",
        "productive_credit_amount_sum_m_rstd_6m",
        "amount_sum_m_ravg_3m",
        "amount_sum_m_rstd_3m",
        "amount_sum_m_ravg_6m",
        "amount_sum_m_rstd_6m",
        "amount_avg_m_ravg_3m",
        "amount_avg_m_rstd_3m",
        "amount_avg_m_ravg_6m",
        "amount_avg_m_rstd_6m",
        "amount_min_m_ravg_3m",
        "amount_min_m_rstd_3m",
        "amount_min_m_ravg_6m",
        "amount_min_m_rstd_6m",
        "amount_max_m_ravg_3m",
        "amount_max_m_rstd_3m",
        "amount_max_m_ravg_6m",
        "amount_max_m_rstd_6m",
        "ind_vehicle_loan_product_rmin_3m",
        "ind_vehicle_loan_product_rmax_3m",
        "ind_vehicle_loan_product_rmin_6m",
        "ind_vehicle_loan_product_rmax_6m",
        "ind_mortgage_loan_product_rmin_3m",
        "ind_mortgage_loan_product_rmax_3m",
        "ind_mortgage_loan_product_rmin_6m",
        "ind_mortgage_loan_product_rmax_6m",
        "ind_credit_card_product_rmin_3m",
        "ind_credit_card_product_rmax_3m",
        "ind_credit_card_product_rmin_6m",
        "ind_credit_card_product_rmax_6m",
        "ind_productive_credit_product_rmin_3m",
        "ind_productive_credit_product_rmax_3m",
        "ind_productive_credit_product_rmin_6m",
        "ind_productive_credit_product_rmax_6m",
        "products_cnt_m_rmin_3m",
        "products_cnt_m_rmax_3m",
        "products_cnt_m_rmin_6m",
        "products_cnt_m_rmax_6m",
    )

    # Define primary keys
    df = df.withColumnsRenamed(pks_mapper)

    # Añade prefijo por feature table
    pks = list(pks_mapper.values())
    features_prefix = "chp_"
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

features_holding_prod = calculate_advanced_features(df)

# COMMAND ----------

pks_mapper = {
    "id_customer": "pk_customer",
    "release_dt": "tpk_release_dt",
}
features_holding_prod = feature_output_standarize(features_holding_prod, pks_mapper)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

features_holding_prod.cache()

# COMMAND ----------

database = fs_database
table = 'fs_cus_holding_products'
path_name = fs_base_path + f"/{table}"

entity = "customer"
table_description = "holding products"

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
        df=features_holding_prod,
        force_overwrite=force_overwrite,
        overwriteSchema=overwriteSchema,
        omit_data_validation_errors=omit_data_validation_errors,
    )
    successfully_saved = True

# COMMAND ----------

spark.catalog.clearCache()
features_holding_prod.unpersist(True)

del features_holding_prod
gc.collect()

# COMMAND ----------

if not successfully_saved:
    failure_reason_message = "\n\t- " + "\n\t- ".join(failure_reason.split("\n"))
    message = f"Error saving Feature Table '{database}.{table}':" + failure_reason_message
    raise Exception(message)
