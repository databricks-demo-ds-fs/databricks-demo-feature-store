# Databricks notebook source
base_path_dev = "dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/AA_PROJECTS/DATA_ENGINEERING_DEV"
base_path_prod = "dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/AA_PROJECTS/DATA_ENGINEERING"

monitor_path_dev = "dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/AA_PROJECTS/DATA_ENGINEERING_DEV/MONITORING/data_validation"
monitor_path_prod = "dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/AA_PROJECTS/MONITORING/data_validation"

fs_db_dev = "feature_store_demo_dev"
fs_db_prod = "feature_store_demo_prod"

fs_base_path_dev = base_path_dev + "/FEATURE-STORE"
fs_base_path_prod = base_path_prod + "/FEATURE-STORE"

# COMMAND ----------

DATABRICKS_CREDENTIALS = {
    "prod": {
        "prod_bnca_pers": {
            "HOST": "https://adb-YOURWORKSPACE.A.azuredatabricks.net/",
        },
        "prod_med_pag": {
            "HOST": "https://adb-YOURWORKSPACE.B.azuredatabricks.net/",
        },
    },
    "dev": {
        "dev_bnca_pers": {
            "HOST": "https://adb-YOURWORKSPACE.A.azuredatabricks.net/",
        },
        "dev_med_pag": {
            "HOST": "https://adb-YOURWORKSPACE.B.azuredatabricks.net/",
        },
    },
}

scope_name = "advanced_analytics"
for env, workspaces in DATABRICKS_CREDENTIALS.items():
    for env_ws, creds in workspaces.items():
        secret_key = f"{env_ws}_credential"
        creds["TOKEN"] = dbutils.secrets.get(scope=scope_name, key=secret_key)

del scope_name

# COMMAND ----------

from pprint import pprint

pprint(DATABRICKS_CREDENTIALS)