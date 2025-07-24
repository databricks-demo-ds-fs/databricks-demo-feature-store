import gc
import time
import typing as tp
# from datetime import datetime # Not directly used in new docstrings, time is used
from pprint import pprint # Used in print statements in original save_table
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.utils import AnalysisException
from delta.tables import DeltaTable

from constants import conf
# Assuming get_dbutils remains in packages.utils as per prior discussion
from packages.utils import get_dbutils

spark = SparkSession.builder.getOrCreate()
logger = logging.getLogger(__name__)

def get_table(
    dataset: str,
    layer: tp.Literal["source", "ingestion", "preprocessing"],
    format: tp.Literal["parquet", "delta"] = None,
) -> DataFrame:
    """Reads a table from the specified layer and format.

    Args:
        dataset (str): The alias of the table name as defined in the configuration.
        layer (tp.Literal["source", "ingestion", "preprocessing"]): The data layer
            from which to retrieve the table.
        format (tp.Literal["parquet", "delta"], optional): The file format of the
            table. Required for 'ingestion' and 'preprocessing' layers.
            Defaults to None.

    Returns:
        DataFrame: A Spark DataFrame representing the loaded table.

    Raises:
        AnalysisException: If the format is not provided for 'ingestion' or
            'preprocessing' layers.
        KeyError: If the dataset or path_key is not found in `conf.dataset_paths`.
    """
    path_key = layer if layer == "source" else f"{layer}_path"
    path = conf.dataset_paths[dataset][path_key]
    logger.info(f"Reading table for dataset '{dataset}', layer '{layer}', from path: {path}")
    if layer == "source":
        if path.startswith("dbfs:/mnt"): # Specific handling for Databricks mount paths
            # GESTION folder must have only delta tables, no parquet or csv
            df = spark.read.format("delta").load(path)
        else: # Assumed to be a catalog table name
            df = spark.read.table(path)
    else: # For 'ingestion' or 'preprocessing' layers
        if not format:
            msg = f"`format` must be provided for layer '{layer}' (dataset: '{dataset}')"
            logger.error(msg)
            raise AnalysisException(msg)
        df = spark.read.format(format).load(path)
    return df

def _table_exist(path: str) -> bool:
    """Checks if a table or path exists using DBUtils.

    Args:
        path (str): The path to check for existence.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    dbutils = get_dbutils(spark) # Relies on get_dbutils from packages.utils
    try:
        dbutils.fs.ls(path) # This call raises an exception if the path does not exist
        return True
    except Exception:
        return False

def save_table(
    df: DataFrame,
    dataset: str,
    layer: tp.Literal["ingestion", "preprocessing"],
    file_format: tp.Literal["parquet", "delta"] = "delta",
    mode: tp.Literal["overwrite", "append", "errorifexists", "ignore", "merge"] = "overwrite",
    historical_key: tp.Optional[str] = None, # Made explicit with Optional
) -> None:
    """Saves a Spark DataFrame as an external table.

    This function handles saving Spark DataFrames to a specified layer (ingestion
    or preprocessing) using different file formats and save modes. It includes
    logic for historical data handling using a `historical_key` for partitioning
    and Delta Lake merges or dynamic partition overwrites.

    Args:
        df (DataFrame): The Spark DataFrame to be saved.
        dataset (str): The alias of the table name, used to determine the save path
            from the project configuration (`conf.dataset_paths`).
        layer (tp.Literal["ingestion", "preprocessing"]): The data layer where
            the table will be saved.
        file_format (tp.Literal["parquet", "delta"], optional): The file format
            for saving the table. Defaults to "delta".
        mode (tp.Literal["overwrite", "append", "errorifexists", "ignore", "merge"], optional):
            The Spark save mode.
            &nbsp;- "overwrite": Overwrites the existing table or data within partitions.
            &nbsp;- "append": Appends data to the existing table.
            &nbsp;- "errorifexists": Throws an error if the table already exists.
            &nbsp;- "ignore": Does nothing if the table already exists.
            &nbsp;- "merge": (Delta format only) Merges new data based on the `historical_key`.
            Defaults to "overwrite".
        historical_key (str, optional): The column name used for partitioning
            if the table contains historical data. This enables specialized handling
            like merges for Delta tables or dynamic partition overwrites.
            Defaults to None.

    Raises:
        ValueError: If `historical_key` is provided but not found in the DataFrame's columns,
            or if `mode="merge"` is used with a non-Delta `file_format`.
        KeyError: If the dataset or path_key is not found in `conf.dataset_paths`.
        AnalysisException: Can be raised by underlying Spark operations for various
            issues like invalid paths or schema incompatibilities not handled by options.
    """
    start_time = time.time()
    save_path = conf.dataset_paths[dataset][f"{layer}_path"]
    logger.info(f"Saving table for dataset '{dataset}', layer '{layer}', to path: {save_path}")
    logger.info(f"Params: format='{file_format}', mode='{mode}', historical_key='{historical_key}'")

    table_exist_check = _table_exist(save_path)

    columns_to_add = False # Default placeholder, original logic for this was complex

    # Logic from original save_table regarding schema comparison for 'columns_to_add'
    if table_exist_check and file_format=="delta":
        logger.info("Comparing schema for existing Delta table to find new columns.")
        try:
            target_df_for_schema = get_table(dataset, layer, "delta")

            current_target_columns = set(target_df_for_schema.columns)
            source_columns = set(df.columns)

            cols_to_add_list = list(source_columns - current_target_columns)

            if cols_to_add_list:
                columns_to_add = True
                logger.info(f"New columns to add to Delta table '{save_path}': {cols_to_add_list}")
                cols_with_types_str = ", ".join([
                    f"`{col_name}` {df.schema[col_name].dataType.simpleString()}"
                    for col_name in cols_to_add_list
                ])
                alter_table_query = f"ALTER TABLE delta.`{save_path}` ADD COLUMNS ({cols_with_types_str})"
                logger.info(f"Executing alter query: {alter_table_query}")
                spark.sql(alter_table_query)
            else:
                logger.info(f"No new columns to add to Delta table '{save_path}'.")
        except Exception as e:
            logger.warning(f"Could not perform schema comparison or alter table for '{save_path}': {e}")

    if historical_key and not columns_to_add:
        logger.info(f"Processing as historical table with key: '{historical_key}'")
        if historical_key not in df.columns:
            msg = f"The historical_key '{historical_key}' is not in the DataFrame columns: {df.columns}."
            logger.error(msg)
            raise ValueError(msg)

        partition_filter = None
        if table_exist_check and mode not in ["overwrite", "errorifexists", "ignore"]:
            if file_format == "delta":
                try:
                    source_partitions_distinct_df = df.select(historical_key).distinct()
                    source_partitions_list = [
                        row[historical_key].strftime("%Y-%m-%d")
                        for row in source_partitions_distinct_df.collect()
                        if row[historical_key] is not None
                    ]
                    if source_partitions_list:
                        partition_filter = f"`{historical_key}` IN ({', '.join(f'{repr(p)}' for p in source_partitions_list)})"
                        logger.info(f"Generated partition_filter for replaceWhere from source data: {partition_filter}")
                    else:
                        logger.info("No distinct, non-null partitions found in source data for historical_key; replaceWhere filter not generated.")
                except Exception as e:
                    logger.warning(f"Could not determine partition filter from source data for historical key '{historical_key}': {e}")

        if file_format == "delta" and mode == "merge":
            if not DeltaTable.isDeltaTable(spark, save_path):
                logger.info("Target for merge is not a Delta table or does not exist. Saving as new Delta table.")
                df.write.format("delta").partitionBy(historical_key).save(save_path)
            else:
                logger.info("Merging data into existing Delta table.")
                delta_table_obj = DeltaTable.forPath(spark, save_path)
                delta_table_obj.alias("target").merge(
                    df.alias("source"),
                    f"target.`{historical_key}` = source.`{historical_key}`"
                ).whenNotMatchedInsertAll().execute()
        else:
            schema_option = "overwriteSchema" if mode == "overwrite" else "mergeSchema"
            print(f"schema mode: {schema_option}")
            writer = df.write.format(file_format).option(schema_option, "true").partitionBy(historical_key)

            if mode != "overwrite" and table_exist_check and partition_filter and file_format == "delta":
                logger.info(f"Using replaceWhere for Delta dynamic partition overwrite with filter: {partition_filter}")
                writer = writer.option("replaceWhere", partition_filter).mode("overwrite")
            elif mode != "overwrite" and table_exist_check and file_format == "parquet":
                logger.info("Using dynamic partition overwrite for Parquet.")
                writer = writer.option("partitionOverwriteMode", "dynamic").mode(mode)
            else:
                writer = writer.mode(mode)
            writer.save(save_path)
    else:
        if columns_to_add:
            logger.info("Saving table with added columns (implies overwrite logic).")
        elif not historical_key:
            logger.info("Processing as non-historical table or initial save (implies overwrite logic).")
        schema_option = "overwriteSchema" if mode == "overwrite" else "mergeSchema"
        logger.info(f"schema mode: {schema_option}")
        print(f"schema mode: {schema_option}")
        df.write.format(file_format).option(schema_option, "true").mode("overwrite").save(save_path)


    if file_format == "delta":
        try:
            if DeltaTable.isDeltaTable(spark,save_path):
                logger.info(f"Optimizing Delta table at: {save_path}")
                DeltaTable.forPath(spark, save_path).optimize().executeCompaction()
            else:
                logger.info(f"Skipping optimize as path {save_path} is not a Delta table after save (possibly empty write or non-delta format).")
        except Exception as e:
            logger.warning(f"Could not optimize Delta table at {save_path}. It might be empty or an issue occurred: {e}")

    if df.is_cached:
        df.unpersist(True)
    spark.catalog.clearCache()
    gc.collect()

    duration_time = time.time() - start_time
    logger.info(f"Table saving for '{dataset}' layer '{layer}' completed. Duration: {duration_time:.2f} seconds.")
