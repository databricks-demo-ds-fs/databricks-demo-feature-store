# Databricks notebook source
import subprocess
import warnings
from datetime import date
import typing as tp

# COMMAND ----------

def get_actual_date():
    """Get the first day of the current month in YYYY-MM-DD format.

    Returns:
        str: Date string in YYYY-MM-DD format representing the first day of current month.
    """
    end_date = date.today().replace(day=1)
    return end_date.strftime('%Y-%m-%d')

def validate_feature_table_notebook(feature_store_files_path, feature_table_notebook):
    """Validate if a feature table notebook exists in the specified path.

    Args:
        feature_store_files_path (str): Path to the feature store files directory.
        feature_table_notebook (str): Name of the feature table notebook to validate.

    Raises:
        FileNotFoundError: If the specified notebook is not found in the feature store path.
    """
    # Get the feature tables' notebook path
    # feature_store_files_path = f"{folder_path}/{environment_folder}/FEATURE_STORE"
    command = f"ls {feature_store_files_path}"
    files_in_feature_folder = subprocess.run(
        command.split(" "),
        capture_output=True,
        text=True,
        check=False
    ).stdout.split('\n')[:-1]
    feature_notebook_files = [f for f in files_in_feature_folder if f.startswith("fs_")]

    # Validate if the notebook exists in the path
    noteebook_not_found = feature_table_notebook not in feature_notebook_files
    if noteebook_not_found:
        message = f"The following notebook {feature_table_notebook} " + \
                f"was not found not found in '{feature_store_files_path}' folder. " + \
                "There are the following files:" + \
                "\n\t- " + "\n\t- ".join([ file for file in feature_notebook_files ])
        raise FileNotFoundError(message)

# COMMAND ----------

def preprocess_notebook_parameters(
        notebook_parameters:tp.Dict[str, str]
    ) -> tp.Tuple[str, str, str, tp.Dict[str, str]]:
    """Validate and preprocess notebook parameters for feature table execution.

    This function validates required parameters and performs preprocessing operations:
    - Validates the 'ignore' parameter
    - Validates and processes the environment parameter
    - Processes the end_date parameter
    - Validates the feature table notebook parameter
    - Validates the notebook's existence in the feature store path

    Args:
        notebook_parameters (Dict[str, str]): Dictionary containing notebook parameters.

    Raises:
        ValueError: If required parameters are missing or invalid.
        Exception: If the notebook should be ignored.
        FileNotFoundError: If the feature table notebook is not found.

    Returns:
        tuple[str, str, str, Dict[str, str]]: A tuple containing:
            &nbsp;- environment (str): The processed environment value
            &nbsp;- feature_table_notebook (str): The name of the feature table notebook
            &nbsp;- feature_table_path (str): The path of the feature table notebook
            &nbsp;- notebook_parameters (Dict[str, str]): The remaining processed parameters
    """
    # Ignore the execution of this notebook?
    ignore = notebook_parameters.pop("ignore", "false").lower()
    if ignore not in ["true", "false"]:
        raise ValueError("Parameter `ignore` must be 'true' or 'false'")
    ignore = True if ignore=="true" else False
    if ignore:
        raise Exception("Ignore this feature")

    # Get the environment
    environment = notebook_parameters.pop("env", None)
    if environment is None:
        raise ValueError("`env` was not defined in the yml resource")
    environment_folder = "production" if environment=="prod" else "development"

    # Preprocess the end_date parameter
    if "end_date" in notebook_parameters:
        if notebook_parameters.get("end_date") == "actual":
            notebook_parameters["end_date"] = get_actual_date()

    # Validate existence of the fs_notebook parameter (notebook name to be executed)
    feature_table_notebook = notebook_parameters.pop("fs_notebook", None)
    if feature_table_notebook is None:
        raise ValueError("`fs_notebook` was not defined in the yml resource")
    feature_table_notebook = feature_table_notebook.lower().strip()

    # Get the notebook's folder path
    this_notebook_path = dbutils.notebook.entry_point.getDbutils()\
        .notebook().getContext()\
        .notebookPath().get()
    folder_path = "/".join(this_notebook_path.split("/")[:-1])
    if not folder_path.startswith("/Workspace"):
        folder_path = "/Workspace" + ('' if folder_path.startswith("/") else '/') + folder_path

    feature_store_files_path = f"{folder_path}/{environment_folder}"
    validate_feature_table_notebook(feature_store_files_path, feature_table_notebook)
    feature_table_path = f"{feature_store_files_path}/{feature_table_notebook}"

    return environment, feature_table_notebook, feature_table_path, notebook_parameters

# COMMAND ----------

environment = dict(dbutils.widgets.getAll())["env"]

# COMMAND ----------

# MAGIC %run /Shared/databricks-demo-feature-store/notebooks/utils $env=environment

# COMMAND ----------

def run_command(notebook_path, **kwargs):
    dbutils.notebook.run(
        notebook_path,
        **kwargs
    )

# COMMAND ----------

def feature_pipeline(
    notebook_parameters:tp.Dict[str, str]
    ):
    """Execute a feature table pipeline with monitoring and error handling.

    This function:
    1. Validates and preprocesses the notebook parameters
    2. Executes the feature table notebook
    3. Handles any errors during execution
    4. Reports success/failure status

    Args:
        notebook_parameters (Dict[str, str]): Dictionary containing pipeline parameters.
    """
    # ================================
    # VARIABLE SETTING
    # ================================
    # Validate and preprocess the notebook_parameters
    (
        env,
        fs_notebook_name,
        notebook_path,
        notebook_parameters
    ) = preprocess_notebook_parameters(notebook_parameters)

    # Initial configuration
    # Job configuration
    CLUSTER_ID = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

    # =========================================================================
    print("🚀 STARTING DATABRICKS MONITORING SYSTEM")
    print("=" * 60)
    print(f"👨🏻‍💻 Enviroment: {env}")
    print(f"🗂️ Notebook: {notebook_path}")
    print(f"🖥️ Cluster ID: {CLUSTER_ID}")
    print("=" * 60)

    # Execute with error handling
    success = False
    try:
        run_command(
            notebook_path=notebook_path,
            timeout_seconds=0,  # no timeout configured
            arguments=notebook_parameters
        )
        success = True
        print("🎉 Execution completed successfully!")
    except Exception as e:
        # Spreading the exception with context
        message = f"Error procesando Feature Table: {fs_notebook_name}"
        logger.error(message, exc_info=True) # Logging the complete traceback
        raise Exception(message) from e
    finally:
        print("🏁 Process completed")


# COMMAND ----------

if __name__ == "__main__":
    feature_pipeline(
        notebook_parameters = dict(dbutils.widgets.getAll())
    )
