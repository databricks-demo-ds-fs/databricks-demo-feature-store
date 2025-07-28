from constants import conf
from packages.preprocessing.transformers.date import id_date_to_datetime

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
import typing as tp

import logging

logger = logging.getLogger(__name__)


def schemas_validation(df: DataFrame, dataset, schema_params) -> DataFrame:
    """Validate schemas.

    Preprocesses the DataFrame from the raw layer to the intermediate layer by
    applying column casting, and renaming.

    Args:
        df (DataFrame): The input DataFrame.
        dataset (str): It's the alias table name given in config folder.
        schema_params (Dict[str, Dict[str, Union[str, Dict[str, str]]]]):
            Dictionary with schema parameters for casting, and renaming columns.
            Each key is the original column name, and each value is a dictionary with 'name',
            'data_type', and optionally 'format_source'.

    Returns:
        DataFrame: The processed DataFrame.
    """
    transformed_cols = []
    
    for old_name, new_schema in schema_params["fields"].items():
        new_name = new_schema.get("name")
        data_type = new_schema.get("data_type")

        if not (new_name and data_type):
            raise ValueError(
                f"Missing 'name' or 'data_type' for column '{old_name}' in schema_params."
            )

        if data_type.upper() == "DATE":
            date_format = new_schema.get("format_source")
            if not date_format:
                raise ValueError(f"Parameter `format_source` was not found in the params of '{old_name}' column")
            
            if date_format.lower() == "yyyy-mm-qq":
                # Handle quarter date format using regex_replace
                col = f.regexp_replace(f.lower(f.col(old_name)), "q2", "16")
                col = f.regexp_replace(col, "q1", "01")
                col = f.to_date(col, "yyyy-MM-dd")
            else:
                col = f.to_date(f.col(old_name), date_format)
            
            transformed_cols.append(col.alias(new_name))
        else:
            # Handle regular type casting
            transformed_cols.append(f.col(old_name).cast(data_type.lower()).alias(new_name))
    
    # Apply all transformations in a single operation
    df = df.select(*transformed_cols)

    logger.info(f"DataFrame output columns: {df.columns}")
    return df

def get_historical_key(ingestion_params:tp.Dict[str, tp.Union[str, tp.Dict]]):
    """
    Get historical key parameters from ingestion parameters.
    This function extracts historical key information from ingestion parameters, specifically
    looking for date column filter details. It returns a dictionary with the historical key
    name and format, or an empty dictionary if no date column filter is specified.

    Args:
        ingestion_params (Dict[str, str | Dict]): Dictionary containing ingestion 
            parameters. Expected to potentially contain 'date_col_filter' and 'fields' keys.

    Returns:
        Dict(str, str): A dictionary containing:
            - 'original_name' (str): Name of the historical key column (name of silver tables), defaults to 'event_dt'
            - 'renamed_name' (str): Name of the historical key column (renamed), defaults to 'event_dt'
            - 'format_source' (str): Format of the date, defaults to 'yyyy-MM-dd'
            Returns empty dict if no date_col_filter is specified.
    """
    date_col_filter = ingestion_params.get("date_col_filter")
    if date_col_filter:
        historical_key_params = ingestion_params["fields"].get(date_col_filter, {})
        historical_key = historical_key_params.get("name", "event_dt") # default 'event_dt' becuase `id_to_date`
        format_source = historical_key_params.get("format_source", "yyyy-MM-dd") # default 'yyyy-MM-dd' because `event_dt` has that format
        # ¿Cuando puede `params["fields"].get(source_date_col, {})` retornar un diccionario  vacío `{}` ?
        # Pasa cuando la columna no está dentro de las columnas en "fields".
        # Esto sucede si `source_date_col` si es de una tabla que solo tenía id de la fecha,
        # cuando pasa eso el proceso de id_to_date une la columna tipo date de formato yyyy-MM-dd 
        # de un catalogo de fechas, lo que hace que no necesite una validación de esquema 
        # como el resto de columnas en "fields".
        # En dicho caso, el proceso de id_to_date siempre retorna la fecha como "event_dt".
        return {
            "original_name": date_col_filter,
            "renamed_name": historical_key,
            "format_source": format_source,
        }
    return {}
