# Databricks notebook source
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import *
from datetime import datetime, date, timedelta

import gc

# COMMAND ----------

# MAGIC %run /Shared/databricks-demo-feature-store/notebooks/utils $env="dev"

# COMMAND ----------

notebook_inputs = dict(dbutils.widgets.getAll())

# COMMAND ----------

source_tables = [
  "demo_db.clientes",
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
            filter_date.append(f"release_dt >= '{date_input}'")
        elif date_key=="end_date":
            end_date_filter = f"tpk_release_dt = '{date_input}'"
            filter_date.append(f"release_dt <= '{date_input}'")


# COMMAND ----------

# MAGIC %md
# MAGIC # Feature fs_demographic

# COMMAND ----------

def get_actual_date():
    end_date = date.today().replace(day=1)
    return end_date.strftime('%Y-%m-%d')

notebook_inputs = dict(dbutils.widgets.getAll())
start_date = notebook_inputs.get("start_date", "2019-01-01")
end_date = notebook_inputs.get("end_date", get_actual_date())
timestamp = "release_dt"

# generate months
date_range = pd.concat(
    [
        pd.date_range(
            start=start_date, end=end_date, freq="M"
        ).to_frame(index=False, name=timestamp),
        pd.DataFrame({ # Include `end_date` (current month)
            timestamp: [pd.to_datetime(end_date).to_period('M').to_timestamp()]
        }),
    ],
    ignore_index=True
)

# create reference date and make pyspark DataFrame
spark = SparkSession.builder.getOrCreate()
date_range = (
    spark.createDataFrame(date_range)
    .withColumn(timestamp, f.trunc(f.col(timestamp).cast("date"), "month"))
)

date_range.createOrReplaceTempView("release_dates")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingesta

# COMMAND ----------

# Tabla principal de Demograficos
df = spark.sql("""
SELECT
    cust.id_cliente AS id_customer,
    dates.release_dt AS release_dt,
    cust.fecha_nacimiento AS birth_date,
    cust.genero AS gender,
    cust.estado_civil AS marital_status,
    cust.nivel_educativo AS educational_level,
    cust.ingresos_mensuales AS avg_income_month,
    cust.zona_residencia AS zone_residence,
    cust.ciudad AS city,
    CAST(date_format(cust.fecha_apertura, 'yyyy-MM-dd') AS DATE) AS customer_code_creation_date,
    --cust.fecha_apertura AS customer_code_creation_date,
    cust.segmento_cliente AS segment
FROM demo_db.clientes as cust
CROSS JOIN (
    SELECT release_dt
    FROM release_dates
) dates
WHERE release_dt>=cust.fecha_apertura
AND id_cliente IS NOT NULL
AND id_cliente != '999999999'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocesar texto

# COMMAND ----------

df = preprocessing_ingesting_tables(df)


# COMMAND ----------

# MAGIC %md
# MAGIC **Categorical Feature Engineering**
# MAGIC
# MAGIC Functions to process and engineer categorical demographic features.

# COMMAND ----------

def process_categorical_features(df: DataFrame) -> DataFrame:
    """Process and engineer categorical demographic features.

    Creates simplified and standardized versions of categorical variables:
    - Educational level: Groups into 'superior', 'secundaria', 'primaria', 'ninguna'
    - Marital status: Groups into 'soltero' (single/divorced) and 'casado' (married/others)

    Args:
        df (pyspark.sql.DataFrame): DataFrame with raw categorical features containing:
            &nbsp;- educational_level: Original education level
            &nbsp;- marital_status: Original marital status

    Returns:
        pyspark.sql.DataFrame: DataFrame with additional processed categorical features:
            &nbsp;- educational_lvl1: Simplified education categories
            &nbsp;- educational_lvl2: Original education level (renamed)
            &nbsp;- marital_status_lvl1: Simplified marital status
            &nbsp;- marital_status_lvl2: Original marital status (renamed)

    Examples:
        ```python
        df_processed = process_categorical_features(df)

        # Educational mapping:
        # 'universitario', 'postgrado' -> 'superior'
        # 'secundaria' -> 'secundaria'
        # 'primaria' -> 'primaria'
        # others -> 'ninguna'

        # Marital mapping:
        # 'divorciado', 'soltero' -> 'soltero'
        # others -> 'casado'
        ```

    Notes:
        - Creates both simplified (lvl1) and detailed (lvl2) versions.
        - Handles null values appropriately for marital status.
        - Uses SQL CASE WHEN expressions for categorical mapping.
    """
    query_edu = """
    CASE WHEN educational_level IN ('universitario', 'postgrado') THEN 'superior'
        WHEN educational_level IN ('secundaria', 'primaria') THEN educational_level
        ELSE 'ninguna' END
    """
    query_marital = """
    CASE WHEN marital_status IN ('divorciado', 'soltero') THEN 'soltero'
        WHEN marital_status NOT IN ('divorciado', 'soltero') THEN 'casado'
        ELSE NULL END
    """

    df = df.withColumns({
        "educational_lvl1": f.expr(query_edu),
        "marital_status_lvl1": f.expr(query_marital),
    }).withColumnsRenamed({
        "educational_level": "educational_lvl2",
        "marital_status": "marital_status_lvl2",
    })

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC **Temporal Feature Engineering**
# MAGIC
# MAGIC Functions to calculate time-based features from demographic data.

# COMMAND ----------

def calculate_temporal_features(df: DataFrame) -> DataFrame:
    """Calculate temporal features based on customer account creation.

    Computes time-based features that indicate customer tenure and lifecycle stage:
    - Customer code creation time in months from release date

    Args:
        df (pyspark.sql.DataFrame): DataFrame containing:
            &nbsp;- release_dt: Reference date for the period
            &nbsp;- customer_code_creation_date: Date when customer account was created

    Returns:
        pyspark.sql.DataFrame: DataFrame with additional temporal feature:
            &nbsp;- cust_code_creation_time_months: Months since account creation

    Examples:
        ```python
        df_with_time = calculate_temporal_features(df)

        # If customer created account in Jan 2024 and release_dt is Apr 2024:
        # cust_code_creation_time_months = 3
        ```

    Notes:
        - Uses months_between function for accurate month calculations.
        - Truncates customer creation date to month for consistency.
        - Useful for customer lifecycle analysis and tenure-based segmentation.
    """
    df = df.withColumns({
        "cust_code_creation_time_months": f.months_between(
            date1="release_dt",
            date2=f.date_trunc("month", "customer_code_creation_date"),
            roundOff=True
        ),
    })

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC **Feature Selection and Standardization**
# MAGIC
# MAGIC Functions to select relevant features and apply standardization.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Integración de Módulos**
# MAGIC
# MAGIC Finalmente, integramos todos los módulos para construir la tabla de features.

# COMMAND ----------

def calculate_advanced_features(df: DataFrame) -> DataFrame:
    """Calculate advanced demographic features integrating all processing steps.

    Processes demographic data through a standardized pipeline:
    1. Categorical feature engineering (education and marital status grouping)
    2. Temporal feature calculation (account age)

    Args:
        df (pyspark.sql.DataFrame): DataFrame with raw demographic data that must contain:
            &nbsp;- id_customer: Unique customer identifier
            &nbsp;- release_dt: Period release date
            &nbsp;- educational_level: Raw education level
            &nbsp;- marital_status: Raw marital status
            &nbsp;- customer_code_creation_date: Account creation date
            &nbsp;- Other demographic fields (gender, income, location, segment)

    Returns:
        pyspark.sql.DataFrame: DataFrame with processed demographic features including:
            &nbsp;- All original demographic fields (gender, income, zone, city, segment)
            &nbsp;- Processed categorical features (educational_lvl1/2, marital_status_lvl1/2)
            &nbsp;- Temporal features (cust_code_creation_time_months)
            &nbsp;- Ready for feature store standardization

    Examples:
        ```python
        # Input DataFrame with raw demographic data
        demo_features = calculate_advanced_features(df_raw)

        # Expected output includes:
        # - gender: 'M', 'F'
        # - educational_lvl1: 'superior', 'secundaria', 'primaria', 'ninguna'
        # - marital_status_lvl1: 'soltero', 'casado'
        # - cust_code_creation_time_months: 24, 36, etc.
        ```

    Notes:
        - Unlike other feature files, demographic features are mostly static.
        - No rolling statistics or lag features are calculated as demographics don't change frequently.
        - Focuses on categorical processing and tenure calculation.
        - Feature selection is handled in the standardization step.
    """
    # Paso 1: Procesar características categóricas
    df = process_categorical_features(df)

    # Paso 2: Calcular características temporales
    df = calculate_temporal_features(df)

    return df

# COMMAND ----------

def feature_output_standarize(df: DataFrame, pks_mapper: Dict[str, str]) -> DataFrame:
    """Standardize demographic feature output format applying feature store conventions.

    Selects specific columns, applies primary key renaming, adds prefixes
    to features and normalizes data types for feature store compatibility.

    Args:
        df (pyspark.sql.DataFrame): DataFrame with processed demographic features that must contain
            expected demographic columns.
        pks_mapper (Dict[str, str]): Dictionary mapping for renaming primary keys.
            Example: {"id_customer": "pk_customer", "release_dt": "tpk_release_dt"}

    Returns:
        pyspark.sql.DataFrame: Standardized DataFrame with:
            &nbsp;- Selected columns in specific order
            &nbsp;- Primary keys renamed according to pks_mapper
            &nbsp;- "cdmg_" prefix added to all features (except primary keys)
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
        # Columns: pk_customer, tpk_release_dt, cdmg_gender, cdmg_educational_lvl1, ...
        ```

    Notes:
        - Uses auxiliary functions decimals_to_floats() and longs_to_integers().
        - The "cdmg_" prefix identifies customer demographic features.
        - Selects a specific subset of 12 demographic feature columns.

    Raises:
        AttributeError: If DataFrame doesn't contain expected columns.
    """
    df = df.select(
        "id_customer",
        "release_dt",
        "gender",
        "educational_lvl1",
        "educational_lvl2",
        "marital_status_lvl1",
        "marital_status_lvl2",
        "avg_income_month",
        "zone_residence",
        "city",
        "segment",
        "cust_code_creation_time_months",
    )

    # Define primary keys
    df = df.withColumnsRenamed(pks_mapper)

    # Añade prefijo por feature table
    pks = list(pks_mapper.values())
    features_prefix = "cdmg_"
    dict_for_renames = {col: f"{features_prefix}{col}" for col in df.columns if col not in pks }
    df = df.withColumnsRenamed(dict_for_renames)

    # DECIMAL -> DOUBLE
    df = decimals_to_floats(df)

    # LONG -> INT
    df = longs_to_integers(df)

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Feature

# COMMAND ----------

# Calculate advanced features
df_customer_features = calculate_advanced_features(df)

# COMMAND ----------

# Standardize output
pks_mapper = {
    "id_customer": "pk_customer",
    "release_dt": "tpk_release_dt",
}
df_customer_features = feature_output_standarize(df_customer_features, pks_mapper)


# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

df_customer_features = df_customer_features.cache()

# COMMAND ----------

database = fs_database
table = 'fs_cus_demographic'
path_name = fs_base_path + f"/{table}"

entity = "customer"
table_description = "demographic"

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
        df=df_customer_features,
        force_overwrite=force_overwrite,
        overwriteSchema=overwriteSchema,
        omit_data_validation_errors=omit_data_validation_errors,
    )
    successfully_saved = True


# COMMAND ----------

df_customer_features.unpersist(True)
del df_customer_features

spark.catalog.clearCache()
gc.collect()

# COMMAND ----------

if not successfully_saved:
    failure_reason_message = "\n\t- " + "\n\t- ".join(failure_reason.split("\n"))
    message = f"Error saving Feature Table '{database}.{table}':" + failure_reason_message
    raise Exception(message)