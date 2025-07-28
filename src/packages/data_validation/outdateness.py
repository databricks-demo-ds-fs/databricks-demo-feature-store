from constants import conf

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import typing as tp
from tqdm.auto import tqdm

from pprint import pprint
import logging

logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()


def validate_outdatedness(
    source_tables: tp.List[str],
    base_path:str,
    dfs: tp.List[DataFrame] = None,
    layer: tp.Literal["source", "ingestion", "preprocessing"] = "source",
):
    """
    Validate that the tables with historical information are up to date considering the date shifts.
    base_path: could change between enviroments
    source_tables: tables that my Feature Table use

    TODO: validar entre diferentes capas, al momento es solo source
    """
    tables = [
        table.split(".")[-1] if "." in table else table for table in source_tables
    ]
    # si no tiene '.' es porque no se la consume del catalogo de databricks, o sea es del folder GESTION

    df_shift_tables = spark.read.format("delta").load(f"{base_path}/shift_tables")
    filter_query = (
        f"table='{tables[0]}'" if len(tables) == 1 else f"table IN {tuple(tables)}"
    )
    df_shift_tables = df_shift_tables.filter(filter_query)

    outdated_tables = {}
    for i, table_data in tqdm(enumerate(df_shift_tables.toJSON().collect())):
        table_data = json.loads(table_data)
        print("validate_outdatedness table_data:")
        pprint(table_data)
        table = table_data["table"]
        source = table_data["source"]
        real_disponibilization_in_months = table_data[
            "real_disponibilization_in_months"
        ] # si no tiene un valor, se tiene que caer
        source_date_column = table_data.get("source_date_column", None)
        source_date_format = table_data.get("source_format", None)
        source_id_date_column = table_data.get("source_id_date_column", None)
        renamed_id_date_column = table_data.get("renamed_id_date_column", None)
        renamed_date_column = table_data.get("renamed_date_column", None)

        if layer != "source":
            dataset = [ d for d, v in conf.source_datasets.items() if table==v["table_name"] ][0]
            path = conf.dataset_paths[dataset][f"{layer}_path"]
        else:
            path = source
        print(f"validate_outdatedness path: {path}")
        
        if not dfs:
            if path.startswith("dbfs:/"):
                df_table = spark.read.format("delta").load(path)
            else:
                df_table = spark.read.table(path)
        else:
            print(f"validate_outdatedness: {len(dfs)}")
            df_table = dfs[i]

        validate_with_renamed_col = False
        if layer == "source":
            if source_date_column:
                # if df_table.schema[source_date_column].dataType.lower().startwith("string"):
                df_table = df_table.withColumn(
                    source_date_column,
                    f.date_format(
                        f.to_timestamp(f.col(source_date_column).cast("string"), "yyyy"),
                        "yyyy-12-01",
                    ).cast("date")
                    if source_date_format == "yyyy"
                    else f.to_date(f.col(source_date_column), source_date_format).cast(
                        "date"
                    ),
                )
                print("validate_outdatedness: pasó casting de fecha")

                max_date = df_table.select(
                    f.max(source_date_column).alias("max_date")
                ).collect()[0][0]
                print(f"validate_outdateness df_table is empty: {df_table.count()}")
                print(f"validate_outdatedness max_date: {max_date}")
                # print(f"validate_outdatedness {source_date_column} values:")
                # pprint(df_table.groupBy(source_date_column).count().select(source_date_column).collect())
                
                date_shifted = max_date + relativedelta(
                    months=real_disponibilization_in_months
                )
                actual_month = datetime.today().replace(day=1).date()
                month_needed = actual_month - relativedelta(
                    months=real_disponibilization_in_months
                )

                if date_shifted < actual_month:
                    months_difference = (
                        relativedelta(actual_month, date_shifted).months
                        + relativedelta(actual_month, date_shifted).years * 12
                    )
                    outdated_tables[table] = {
                        "source": source,
                        "date_column": source_date_column,
                        "max_date": max_date,
                        "month_needed": month_needed,
                        "actual_month": actual_month,
                        "months_difference": months_difference,
                    }
            elif (source_date_column is None) and (source_id_date_column is not None):
                # if the table has an id_date column, we need to convert it to a date column
                validate_with_renamed_col = True
            else:
                raise Exception(
                    f"The validate_outdateness function has no implementation for de following conditions:" +
                    f"\n\t- source_date_column: {source_date_column}" +
                    f"\n\t- source_date_format: {source_date_format}" +
                    f"\n\t- source_id_date_column: {source_id_date_column}" +
                    f"\n\t- renamed_date_column: {renamed_date_column}" +
                    f"\n\t- renamed_id_date_column: {renamed_id_date_column}" +
                    f"\n\t- source: {source}" +
                    f"\n\t- table: {table}"
                )
        
        if layer != "source" or validate_with_renamed_col:
            # Is thought to be implemented for preprocessing, not feature tables
            # validate_with_renamed_col solo cuando pasas como argumento el df
            max_date = df_table.select(
                f.max(renamed_date_column).alias("max_date")
            ).collect()[0][0]
            print(f"validate_outdateness df_table is empty: {df_table.count()}")
            print(f"validate_outdatedness max_date: {max_date}")
            # print(f"validate_outdatedness {renamed_date_column} values:")
            # pprint(df_table.groupBy(renamed_date_column).count().select(renamed_date_column).collect())
            
            date_shifted = max_date + relativedelta(
                months=real_disponibilization_in_months
            )
            actual_month = datetime.today().replace(day=1).date()
            month_needed = actual_month - relativedelta(
                months=real_disponibilization_in_months
            )
            if date_shifted < actual_month:
                months_difference = (
                    relativedelta(actual_month, date_shifted).months
                    + relativedelta(actual_month, date_shifted).years * 12
                )
                outdated_tables[table] = {
                    "source": source,
                    "date_column": source_date_column,
                    "max_date": max_date,
                    "month_needed": month_needed,
                    "actual_month": actual_month,
                    "months_difference": months_difference,
                }

    return outdated_tables

def raise_if_needed_for_outdateness(outdated_tables):    
    if outdated_tables:  # if it's not empty
        tables_description = [
            f"\t- {table}: it's max date is {table_data['max_date']}, has {table_data['months_difference']} months difference with the actual month {table_data['actual_month']}\n\t  source: {table_data['source']}"
            for table, table_data in outdated_tables.items()
        ]
        message = "The following tables are outdated:\n" + "\n".join(tables_description)
        raise Exception(message)
    else:
        print("All tables are up to date.")
