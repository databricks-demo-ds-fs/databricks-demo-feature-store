from constants import conf
from packages.storage.storage_operations import get_table
from packages.data_validation.outdateness import validate_outdatedness
from packages.data_validation.schemas import get_historical_key

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, functions as f
from pyspark.sql.utils import AnalysisException
from delta.tables import DeltaTable
from pyspark.sql.types import *
import typing as tp
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import time
import inspect

from pprint import pprint
import warnings
import logging

logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()


# monitoring_path prod = "dbfs:/mnt/SANDBOX/WORKSPACE/AA_PROJECTS/MONITORING/data_validation"
# monitoring_path dev = "dbfs:/mnt/SANDBOX/WORKSPACE/AA_PROJECTS/DATA_ENGINEERING_DEV/MONITORING/data_validation"
summary_path = f"{conf.config_globals['monitoring_path']}/fs_data_load/validation_summary"
detail_path = f"{conf.config_globals['monitoring_path']}/fs_data_load/validation_detail"


def get_last_validation(database, table_name) -> dict:
    """
    Obtiene un diccionario de los valores del registro más recientes de la tabla de validación de carga mensual.
    La tabla se filtra por database y table_name.
    Tener en cuenta que si un valor es NULL en ese registro, no existirá en el diccionario a retornar.
    """
    data_validation_folder = "dbfs:/mnt/SANDBOX/WORKSPACE/AA_PROJECTS/MONITORING/data_validation"
    monthly_validation_summary_path = f"{data_validation_folder}/monthly_data_load/validation_summary"

    df = spark.read.format('delta').load(monthly_validation_summary_path)

    db_operator = ' IS ' if database is None else '='
    database = f"'{database}'" if database is not None else 'NULL'
    table_filter = f"base_datos{db_operator}{database} AND tabla='{table_name}'"
    df_filtered = df.filter(table_filter)\
        .orderBy(f.desc("fecha_ejecucion"))\
        .limit(1)
    json_register = df_filtered.toJSON().collect()
    if json_register:
        return json.loads(json_register[0])
    else:
        return {}


def get_dataframe_to_compare(path, force_last_version:bool=False) -> DataFrame:
    """
    Obtiene el dataframe a comparar con el que se quiere guardar
    """
    history_df = DeltaTable.forPath(spark, path).history()
    initial_day_of_month = datetime.now().replace(day=1)
    before_month_versions = history_df.filter(history_df.timestamp<initial_day_of_month)\
                                    .orderBy(f.desc("timestamp")).limit(1)
    if before_month_versions.count()==0 or force_last_version:
        # Doesn't exist version of the before month
        # We're going to take the last version of the dataset (probably saved in this month)
        return spark.read.format("delta").load(path)
    else:
        version_to_compare = before_month_versions.select("version").collect()[0][0]
        return spark.sql(f"SELECT * FROM delta.`{path}` VERSION AS OF {version_to_compare}")


def feature_validator(
        df_source:DataFrame, 
        dataset:str, 
        ignore_validations:tp.List[str]=[],
        layer:tp.Literal["source", "ingestion", "preprocessing", "feature"]=None,
    ) -> tp.Tuple[bool, dict]:
    layer = "preprocessing"
    table_name = conf.source_datasets[dataset]["table_name"]
    database = conf.source_datasets[dataset]["database"]
    table = f"{database}.{table_name}" if database is not None else table_name
    
    source_date_col = conf.ingestion_params["tables"][dataset].get("date_col_filter")
    is_historical = source_date_col is not None
    
    execute_saving = True
    validations = {}
    if DeltaTable.isDeltaTable(spark, summary_path): # existe
        df_summary = spark.read.format("delta").load(summary_path)
        # "fecha_ejecucion", "base_datos", "tabla", "tipo_tabla", "estado", "motivo_fallo", "validaciones_ignoradas"
        df_filtered = df_summary\
            .filter(f"base_datos='{layer}' AND tabla='{table_name}'")\
            .orderBy(f.desc("fecha_ejecucion")).limit(1)
        json_register = json.loads(df_filtered.toJSON().collect()[0])
        table_status = json_register.get("estado", None) # Si no encuentra registros, se lo considera como un FAIL (no se guardó)
        # fail_reason = json_register.get("motivo_fallo", "Tabla no encontrada en registros de validación.")

        validations["TABLE_TYPE"] = "historical" if is_historical else "dimensional"
        
        validations[f"{layer.upper()}_SAVED"] = {
            "status": "SUCCESS" if table_status=="SUCCESS" else "FAIL",
            "message": None if table_status=="SUCCESS" else (
                "Tabla no encontrada en registros de validación." 
                if table_status is None 
                else "Tabla no pasó las validaciones."
            ),
        }
        if f"{layer.upper()}_SAVED" not in ignore_validations:
            execute_saving &= table_status=="SUCCESS"
        
        if is_historical:
            # Validación de que se haya cargado la tabla hasta donde el shift date está configurado
            outdated_tables = validate_outdatedness(
                source_tables=[table],
                dfs = [df_source],
                base_path = conf.config_globals["base_path"],
                layer = layer,
            ).get(table_name, {}) # Vacío si la tabla está actualizada
            print("Validate Outdatedness")
            pprint(outdated_tables)
            max_date = None if outdated_tables.__len__()==0 else outdated_tables.get("max_date")
            month_needed = None if outdated_tables.__len__()==0 else outdated_tables.get("month_needed")
            months_difference = None if outdated_tables.__len__()==0 else outdated_tables.get("months_difference")
            message = None if outdated_tables.__len__()==0 else (
                f"The table has {months_difference} months missing. " + 
                f"It has until {max_date} and it's needed until {month_needed.strftime('%Y-%m')}."
            )
            validations["OUTDATENESS"] = {
                "n_months": months_difference,
                "month_found": None if outdated_tables.__len__()==0 else max_date.strftime("%Y-%m-%d"),
                "month_needed": None if outdated_tables.__len__()==0 else month_needed.strftime("%Y-%m-%d"),
                "status": "SUCCESS" if outdated_tables.__len__()==0 else "FAIL",
                "message": message,
            }
            if "OUTDATENESS" not in ignore_validations:
                execute_saving &= (outdated_tables.__len__()==0)

        return True, {}
    else:
        print("No hay una previa version de la tabla. Recien se va a guardar.")
        return False, {}


def preprocessing_validator(
        df_source:DataFrame, 
        dataset:str, 
        params:dict,
        ignore_validations:tp.List[str]=[],
        layer:tp.Literal["source", "ingestion", "preprocessing", "feature"]=None,
        verbose:bool=False,
    ) -> tp.Tuple[bool, dict]:
    """
    
    """
    layer = "ingestion"
    table_name = conf.source_datasets[dataset]["table_name"]
    database = conf.source_datasets[dataset]["database"]
    table = f"{database}.{table_name}" if database is not None else table_name
    print(f"preprocessing_validator table: {table}")
    
    source_date_col = conf.ingestion_params["tables"][dataset].get("date_col_filter")
    is_historical = source_date_col is not None
    print(f"preprocessing_validator. source_date_col: {source_date_col}, is_historical: {is_historical}")
    print(f"Does summary table exist?: {DeltaTable.isDeltaTable(spark, summary_path)}")
    
    execute_saving = True
    validations = {}
    validations["TABLE_TYPE"] = "historical" if is_historical else "dimensional"
    if DeltaTable.isDeltaTable(spark, summary_path): # existe
        df_summary = spark.read.format("delta").load(summary_path)
        # "fecha_ejecucion", "base_datos", "tabla", "tipo_tabla", "estado", "motivo_fallo", "validaciones_ignoradas"
        df_filtered = df_summary\
            .filter(f"base_datos='{layer}' AND tabla='{table_name}'")\
            .orderBy(f.desc("fecha_ejecucion")).limit(1)
        table_status = False
        fail_reason = ""
        if df_filtered.count()!=0:
            json_register = json.loads(df_filtered.toJSON().collect()[0])
            table_status = json_register.get("estado", None) # Si no encuentra registros, se lo considera como un FAIL (no se guardó)
            fail_reason = json_register.get("motivo_fallo", "Tabla no encontrada en registros de validación.")
            print("preprocessing_validator. summary json_register:")
            pprint(json_register)
            
            message = None if table_status=="SUCCESS" else (
                "Tabla no encontrada en registros de validación." 
                if table_status is None 
                else "Tabla no pasó las validaciones."
            )
        else:
            message = f"Table was not found in {layer} logs"
        validations[f"{layer.upper()}_SAVED"] = {
            "status": "SUCCESS" if table_status=="SUCCESS" else "FAIL",
            "message": message,
            "not_passed_validations": None if table_status=="SUCCESS" else (
                "No pasó todas las validaciones" in fail_reason
            )
        }
        if f"{layer.upper()}_SAVED" not in ignore_validations:
            execute_saving &= table_status=="SUCCESS"
        
        if is_historical:
            # Validación de que se haya cargado la tabla hasta donde el shift date está configurado
            outdated_tables = validate_outdatedness(
                source_tables=[table],
                dfs = [df_source],
                base_path = conf.config_globals["base_path"],
                layer = layer,
            ).get(table_name, {}) # Vacío si la tabla está actualizada
            print("Validate Outdatedness")
            pprint(outdated_tables)
            max_date = None if outdated_tables.__len__()==0 else outdated_tables.get("max_date")
            month_needed = None if outdated_tables.__len__()==0 else outdated_tables.get("month_needed")
            months_difference = None if outdated_tables.__len__()==0 else outdated_tables.get("months_difference")
            message = None if outdated_tables.__len__()==0 else (
                f"The table has {months_difference} months missing. " + 
                f"It has until {max_date} and it's needed until {month_needed.strftime('%Y-%m')}."
            )
            validations["OUTDATENESS"] = {
                "n_months": months_difference,
                "month_found": None if outdated_tables.__len__()==0 else max_date.strftime("%Y-%m-%d"),
                "month_needed": None if outdated_tables.__len__()==0 else month_needed.strftime("%Y-%m-%d"),
                "status": "SUCCESS" if outdated_tables.__len__()==0 else "FAIL",
                "message": message,
            }
            if "OUTDATENESS" not in ignore_validations:
                execute_saving &= (outdated_tables.__len__()==0)

        return execute_saving, validations
    else:
        print("No hay una previa version de la tabla. Recien se va a guardar.")
        return False, {}
    

def ingestion_validator(
        df_source:DataFrame, 
        dataset:str, 
        params:dict,
        ignore_validations:tp.List[str]=[],
        layer:tp.Literal["source", "ingestion", "preprocessing", "feature"]=None,
        force_last_version:bool=False,
        verbose:bool=False,
    ) -> tp.Tuple[bool, dict]:
    """
    
    """
    function_name = inspect.currentframe().f_code.co_name
    layer = "source"

    execute_saving = True
    validations = {}
    # Obtener data de la tabla
    table_name = conf.source_datasets[dataset]["table_name"]
    database = conf.source_datasets[dataset]["database"]
    table = f"{database}.{table_name}" if database is not None else table_name

    # Obtener el path tabla dependiendo del layer
    target_layer = "ingestion"
    path_key = target_layer if target_layer=="source" else f"{target_layer}_path"
    target_path = conf.dataset_paths[dataset][path_key]
    
    # OBTENER VALIDACIÓN 0: Verificar si la tabla ya fue guardada correctamente en su último corte.
    print(f"{function_name.upper()}. Analyzing MONTHLY_VAL for {dataset}")
    last_register_validation = get_last_validation(database, table_name)
    if not last_register_validation:
        print(f"{function_name.upper()}. The table {table_name} doesn't exist in the monthly_data_load validation. We're going to ignore this step and continue with the following validations.")
        validations["MONTHLY_VAL"] = {
            "status": None,
            "message": None,
        }
    else:
        status = last_register_validation.get("estado").lower()
        fail_reason = last_register_validation.get("motivo_fallo", "Razon no identificada.").replace("\n", "")
        validations["MONTHLY_VAL"] = {
            "status": "SUCCESS" if status == "success" else "FAIL",
            "message": None if status == "success" else fail_reason,
        }
        if "MONTHLY_VAL" not in ignore_validations:
            execute_saving &= status == "success"

    # EJECUTAR VALIDACIÓN 1: Verificar que existan todas las columnas necesarias
    print(f"{function_name.upper()}. Analyzing MISSING_COLUMNS for {dataset}")
    missing_columns = [ col_name for col_name in params["fields"].keys() if col_name not in df_source.columns ]
    message = None
    if missing_columns:
        message = f"There are {missing_columns.__len__()} missing columns in the table."
        if database is None:
            message += f" You can find the table in `{target_path}`"
        warnings.warn(message)
    validations["MISSING_COLUMNS"] = {
        "missing_columns": missing_columns,
        "status": "SUCCESS" if missing_columns.__len__()==0 else "FAIL",
        "message": message,
    }
    if "MISSING_COLUMNS" not in ignore_validations:
        execute_saving &= (missing_columns.__len__()==0) # ejecuta si no hay columnas faltantes

    historical_key_params = get_historical_key(ingestion_params=params)
    source_date_col = historical_key_params.get("original_name", None)
    target_date_col = historical_key_params.get("renamed_name", None)
    date_format = historical_key_params.get("format_source", None)
    del historical_key_params

    validations["TABLE_TYPE"] = "historical" if source_date_col is not None else "dimensional"
    print(f"{function_name.upper()}. The table is {validations['TABLE_TYPE']}")

    if DeltaTable.isDeltaTable(spark, target_path): # existe
        validations["ALREADY_SAVED"] = True

        df_target = get_dataframe_to_compare(
            path=target_path,
            force_last_version=force_last_version,
        )
        params = conf.ingestion_params["tables"][dataset]

        # EJECUTAR VALIDACIÓN 2: Verificar que conserven el tipo de las columnas
        # validations["DATA_TYPE_CHANGED"] = {
        #     columns: {
        #         "expected": type,
        #         "found": type,
        #     },
        #     status: ...,
        #     message: f"The table has changed the data type of {n} columns."
        # }

        # EJECUTAR VALIDACIÓN 3: Verificar cuadre
        if (source_date_col is not None) and (source_date_col in df_source.columns): # cuadre para tablas históricas
            # Asegurarnos de que las columnas estén en formato mensual
            print(f"{function_name.upper()}. Asegurando que las columnas estén en formato mensual")
            print(f"source_date_col: {source_date_col}, target_date_col: {target_date_col}")
            
            if target_date_col not in df_target.columns:
                raise AnalysisException(
                    f"Saved Dataframe doesn't contain `{target_date_col}` column. "
                    "Probably you have change the column defined in `date_col_filter` param y your ingestion-params.yml to a column not saved before.\n"
                    "Recomendation: Use your before column defined in `date_col_filter` and execute the pipeline to save the column you want to add. "
                    "After saving the column you can redifine the column to use in `date_col_filter` param.\n"
                    "If your DataFrame already exist, you can only redifine the `date_col_filter` param with columns already saved."
                )
            
            ## primero, casteamos a tipo fecha con formato yyyy-MM-dd
            if date_format:
                print(f"date_format: {date_format}")
                if date_format.lower() == "yyyy-mm-qq":
                    df_source = df_source.withColumn(
                        source_date_col, f.regexp_replace(f.lower(source_date_col), "q1", "01"),
                    ).withColumn(
                        source_date_col, f.regexp_replace(f.lower(source_date_col), "q2", "16"),
                    )
                    date_format = "yyyy-MM-dd"
                
                df_source = df_source.withColumn(
                    source_date_col,
                    f.to_date(f.col(source_date_col), date_format).cast("date"),
                )
            ## segundo, convertimos los valores de la columna a mensuales
            df_source = df_source.withColumn(
                source_date_col, 
                f.date_format(f.col(source_date_col), "yyyy-MM-01").cast("date")
            )
            df_target = df_target.withColumn(
                target_date_col, 
                f.date_format(f.col(target_date_col), "yyyy-MM-01").cast("date")
            )

            # filtrar df_source para asegurarnos de tenerlo hasta el maximo del df_target
            max_date_already_saved = df_target.select(f.max(target_date_col)).collect()[0][0]
            print(f"{function_name.upper()}. max_date_already_saved: {max_date_already_saved}, {type(max_date_already_saved)}")
            start_date_filter = conf.config_globals["start_date"]
            source_date_filter = [
                f"{source_date_col}<='{max_date_already_saved}'",
                f"{source_date_col}>='{start_date_filter}'",
            ]
            df_source_same_date = df_source.filter(" AND ".join(source_date_filter))

            # ejecutar conteo de registros
            source_counts = df_source_same_date\
                .groupBy(source_date_col).count()\
                .withColumnsRenamed({source_date_col: "date", "count": "source_count"})
            target_counts = df_target\
                .groupBy(target_date_col).count()\
                .withColumnsRenamed({target_date_col: "date", "count": "target_count"})
            
            # Calcular estadisticos para la validación:
            # 1. Calcular diferencias en los conteos. Si existe diferencia, no pasa la validación
            # 2. Calcular porcentaje. Útil para que el usuario dimensione la afectación
            # 3. Calcular si hay algun mes faltante (registros 0) en source o target. 
            #    Esto ayuda a entender si una de las razones de la diferencia puede ser por un mes faltante 
            #    en los datos o la inclusion de un mesque antes no se lo tenía.
            pct_query = """
                CASE WHEN target_count=0 OR target_count IS NULL 
                    THEN NULL 
                    ELSE ABS(source_count - target_count)/target_count 
                END
            """
            df_differences = source_counts.join(
                target_counts,
                on="date",
                how="outer"
            ).na.fill(0).withColumns({
                "date": f.col("date").cast("string"),
                "difference": f.expr("source_count - target_count"),
                "percentage": f.expr(pct_query),
                "source_has_cero": f.expr("source_count = 0"),
                "target_has_cero": f.expr("target_count = 0"),
            })

            counts_registers = [ json.loads(s) for s in df_differences.toJSON().collect() ]
            print("ingestion_validation counts_registers:")
            pprint(counts_registers)

            total_difference = df_differences.select(f.sum(f.abs("difference"))).collect()[0][0]
            cross_checking_status = total_difference==0
            message = None if cross_checking_status==True else (
                f"Has {total_difference} registers of difference in the whole table."
            )
            validations["CROSS_CHECKING"] = {
                "type": "historical",
                "difference": {
                    "total": total_difference,
                    "per_month": {
                        row["date"]: {
                            "difference": row["difference"],
                            "percentage": round(row["percentage"]*100, 6) if row.get("percentage") is not None else None,
                            "source_has_cero": row["source_has_cero"],
                            "target_has_cero": row["target_has_cero"],
                        }
                        for row in counts_registers
                    },
                },
                "status": "SUCCESS" if cross_checking_status else "FAIL",
                "message": message,
            }
            if "CROSS_CHECKING" not in ignore_validations:
                execute_saving &= cross_checking_status

            print(f"Verificando que no esté vacío, antes de validate_outdatedness: {df_source.count()}")
            # Validación de que se haya cargado la tabla hasta donde el shift date está configurado
            outdated_tables = validate_outdatedness(
                source_tables=[table],
                dfs = [df_source],
                base_path = conf.config_globals["base_path"],
                layer = layer,
            ).get(table_name, {}) # Vacío si la tabla está actualizada
            print("Validate Outdatedness")
            pprint(outdated_tables)
            max_date = None if outdated_tables.__len__()==0 else outdated_tables.get("max_date")
            month_needed = None if outdated_tables.__len__()==0 else outdated_tables.get("month_needed")
            months_difference = None if outdated_tables.__len__()==0 else outdated_tables.get("months_difference")
            message = None if outdated_tables.__len__()==0 else (
                f"The table has {months_difference} months missing. " + 
                f"It has until {max_date} and it's needed until {month_needed.strftime('%Y-%m')}."
            )
            validations["OUTDATENESS"] = {
                "n_months": months_difference,
                "month_found": None if outdated_tables.__len__()==0 else max_date.strftime("%Y-%m-%d"),
                "month_needed": None if outdated_tables.__len__()==0 else month_needed.strftime("%Y-%m-%d"),
                "status": "SUCCESS" if outdated_tables.__len__()==0 else "FAIL",
                "message": message,
            }
            if "OUTDATENESS" not in ignore_validations:
                execute_saving &= (outdated_tables.__len__()==0)
        elif (source_date_col is not None) and (source_date_col not in df_source.columns):
            # Es histórica pero la columna de fecha no está en el df_source
            message = f"The column '{source_date_col}' is not in the table. "
            id_to_date_params = params.get("id_to_date", None)
            if id_to_date_params is not None:
                message = f"`id_to_date` process had problem to generate the date column."
            else:
                message += "Make sure that the column existed in previous versions and it's not a problem of a change in your params."

            validations["CROSS_CHECKING"] = {
                "type": "historical",
                "difference": None,
                "status": "FAIL",
                "message": message,
            }
            if "CROSS_CHECKING" not in ignore_validations:
                execute_saving &= False

            validations["OUTDATENESS"] = {
                "n_months": None,
                "month_found": None,
                "month_needed": None,
                "status": "FAIL",
                "message": message,
            }
            if "OUTDATENESS" not in ignore_validations:
                execute_saving &= False

        else: # cuadre para tablas dimensionales
            # cuadre para tablas dimensionales no es necesario 
            # porque se considera en la validación del último corte (validación de axcel)
            # a menos que sea para una tabla a la que no se le hizo dicha validación
            # Por ejemplo: tablas de RFM
            # Aunque si es útil hacerlo para obtener los registros de la diferencia entre las cantidades
            threshold = 0.05 # diferencia debe ser menor o igual al 5%
            target_count = df_target.count()
            source_count = df_source.count()
            # Se valida respecto al que ya existía debido a que es en quien tenemos mayor confianza
            difference = source_count - target_count
            percentage = abs(difference)/target_count if target_count!=0 else None
            cross_checking_status = percentage <= threshold
            message = None if cross_checking_status==True else (
                f"Has {difference} registers of difference in the whole table."
            )
            validations["CROSS_CHECKING"] = {
                "type": "dimensional",
                "difference": {
                    "total": difference,
                    "percentage": round(percentage*100, 6),
                    "source_has_cero": source_count==0,
                    "target_has_cero": target_count==0,
                },
                "status": "SUCCESS" if cross_checking_status else "FAIL",
                "message": message,
            }
            if "CROSS_CHECKING" not in ignore_validations:
                execute_saving &= cross_checking_status

    else:
        print("No hay una previa version de la tabla. Recien se va a guardar.")
        validations["ALREADY_SAVED"] = False
        execute_saving &= True
    # Leer la tabla desde mi layer actual

    return execute_saving, validations

class ValidationSchemas:
    """Class to manage validation schemas"""
    @staticmethod
    def get_summary_schema() -> StructType:
        return StructType([
            StructField("fecha_ejecucion", TimestampType(), False),
            StructField("base_datos", StringType(), True), # Puede ser Null para tablas que vienen de GESTION
            StructField("tabla", StringType(), False),
            StructField("tipo_tabla", StringType(), False),
            StructField("estado", StringType(), True), 
            StructField("motivo_fallo", StringType(), True),
            StructField("validaciones_ignoradas", StringType(), True),
        ])
        # Una razones por las que `estado` puede ser Null:
        #   La validación de MONTHLY_VAL pudo no incluir alguna tabla

    @staticmethod
    def get_detail_schema() -> StructType:
        return StructType([
            StructField("fecha_ejecucion", TimestampType(), False),
            StructField("base_datos", StringType(), True), # Puede ser Null para tablas que vienen de GESTION
            StructField("tabla", StringType(), False),
            StructField("tipo_tabla", StringType(), False),
            StructField("estado", StringType(), True),
            StructField("tipo_validacion", StringType(), False),
            StructField("ignora_validacion", BooleanType(), False),
            StructField("detalle_validacion", StringType(), True), # contiene json
        ])

validation_keys = {
    "ingestion": ["MONTHLY_VAL", "MISSING_COLUMNS", "CROSS_CHECKING", "OUTDATENESS"], # DATA_TYPE_CHANGED
    "preprocessing": ["INGESTION_SAVED", "OUTDATENESS"],
    "feature": ["PREPROCESSING_SAVED", "OUTDATENESS"],
}

def register_validation(
    dataset:str,
    layer:tp.Literal["ingestion", "preprocessing", "feature"], 
    execution_timestamp,
    successfully_saved:bool, # True: saved, False: something were wrong while saving
    validation_values:dict, 
    ignore_validations:tp.List[str]=[],
    mode:tp.Literal["validate", "save"]="save",
    ) -> str:
    table_name = conf.source_datasets[dataset]["table_name"]
    database = conf.source_datasets[dataset]["database"]
    table_type = validation_values["TABLE_TYPE"]
    
    data_summary = []
    data_detail = []
    failure_reason = []
    resume_status = True
    already_saved = validation_values.get("ALREADY_SAVED", False)
    for val_key in validation_keys[layer]:
        # detail: "fecha_ejecucion","base_datos","tabla","tipo_tabla","estado","tipo_validacion","ignora_validacion","detalle_validacion"
        if table_type=="dimensional" and val_key in ["OUTDATENESS"]:
            # no se valida OUTDATENESS para dimensionales
            continue

        if not already_saved and val_key in ["CROSS_CHECKING", "OUTDATENESS"]:
            # no se valida CROSS_CHECKING y OUTDATENESS si no se ha guardado la tabla previamente
            continue

        default_content = {"status": None, "message": None}
        val_description = validation_values.get(val_key, default_content)
        print(f"register_validation {val_key}")
        pprint(val_description)
        val_status = val_description.pop("status")
        if val_status: # El estado puede ser Null
            resume_status &= ((val_status=="SUCCESS") or (val_key in ignore_validations))
        val_message = val_description.pop("message")
        if val_message is not None and val_status is not None:
            failure_reason.append(f"{val_key}: {val_message}")
        
        val_detail_register = (
            execution_timestamp, database, table_name, 
            table_type, val_status, val_key.lower(), 
            val_key in ignore_validations,
            json.dumps(val_description)
        )
        data_detail.append(val_detail_register)

    saving_message = None
    saving_key = "SAVING"
    print(f"register_validation. successfully_saved: {successfully_saved}, resume_status: {resume_status}")
    if not successfully_saved and resume_status==True:
        resume_status &= False
        saving_message = f"{saving_key}: Se cayó en el proceso de guardado."
        failure_reason.append(saving_message)
    elif not successfully_saved and resume_status==False:
        saving_message = f"{saving_key}: No pasó todas las validaciones."
        failure_reason.append(saving_message)
    print(f"register_validation. successfully_saved: {successfully_saved}, resume_status: {resume_status}")
    # TODO: elif successfully_saved: resume_status=True

    val_detail_register = (
        execution_timestamp, layer, table_name, 
        table_type, "SUCCESS" if successfully_saved else "FAIL", saving_key.lower(),
        False, # No se puede ignorar la validación de guardado
        saving_message
    )
    data_detail.append(val_detail_register)
    print("register_validation data_detail:")
    pprint(data_detail)

    # summary: "fecha_ejecucion", "base_datos", "tabla", "tipo_tabla", "estado", "motivo_fallo"
    failure_reason = None if failure_reason.__len__()==0 else "\n".join(failure_reason)
    val_summary_register = (
        execution_timestamp, layer, table_name, 
        table_type, "SUCCESS" if successfully_saved else "FAIL", failure_reason,
        json.dumps(ignore_validations)
    )
    data_summary.append(val_summary_register)
    print("register_validation data_summary:")
    pprint(data_summary)
    
    # Save the data
    if mode=="save":
        if data_summary.__len__()>0:
            print("register_validation. start summary saving")
            summary_schema = ValidationSchemas().get_summary_schema()
            df_summary = spark.createDataFrame(data_summary, summary_schema)
            summary_mode = "append" if DeltaTable.isDeltaTable(spark, summary_path) else "overwrite"
            df_summary.write.format('delta').mode(summary_mode).save(summary_path)
            print("register_validation. finished summary saving")

        if data_detail.__len__()>0:
            print("register_validation. start detail saving")
            detail_schema = ValidationSchemas().get_detail_schema()
            df_detail = spark.createDataFrame(data_detail, detail_schema)
            detail_mode = "append" if DeltaTable.isDeltaTable(spark, detail_path) else "overwrite"
            df_detail.write.format('delta').mode(detail_mode).save(detail_path)
            print("register_validation. finished detail saving")
    elif mode=="validate":
        print("register_validation. saving skipped")

    return failure_reason
