# Databricks notebook source
import pyspark
import pyspark.sql.functions as f
from pyspark.sql.types import *
from databricks.feature_store import FeatureStoreClient
from enum import Enum
import requests
import json
import time
import typing as tp

from tqdm.auto import tqdm
from pprint import pprint

import gc


# COMMAND ----------

all_inputs = dict(dbutils.widgets.getAll())
is_testing = all_inputs.get("testing", "false")
is_testing = True if is_testing.lower() == "true" else False
environment = all_inputs.get("env", "dev")

# COMMAND ----------

# MAGIC %run /Shared/databricks-demo-feature-store/notebooks/utils $env=environment

# COMMAND ----------

DATABRICKS_CREDENTIALS

# COMMAND ----------

class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    SKIPPED = "SKIPPED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    QUEUED = "QUEUED"
    WAITING_FOR_RETRY = "WAITING_FOR_RETRY"
    BlOCKED = "BlOCKED"

class JobResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

# COMMAND ----------

# Update metadata
def _update_metadata(table:str, properties: dict):
    properties_str = ", ".join([f"'{k}' = '{v}'" for k, v in properties.items()])
    spark.sql(f"""
    ALTER TABLE {table}
    SET TBLPROPERTIES ({properties_str})
    """)

# Get metadata
def get_metadata(table:str):
    metadata = {}
    table_descriptions = json.loads(spark.sql(f"DESCRIBE DETAIL {table}").toJSON().collect()[0])
    metadata["partition_columns"] = table_descriptions.get("partitionColumns", None) #list
    metadata["path"] = table_descriptions["location"] # str
    properties = table_descriptions["properties"] # dict
    metadata["description"] = properties.get("description", None)
    metadata["primary_keys"] = json.loads(properties.get("primary_keys", json.dumps(list())))
    metadata["timestamp_keys"] = json.loads(properties.get("timestamp_keys", json.dumps(list())))
    metadata["tags"] = json.loads(properties.get("tags", json.dumps(dict())))
    metadata["source_tables"] = json.loads(properties.get("source_tables", json.dumps(list())))
    return metadata

# Add table comment
def _set_table_comment(table:str, comment: str):
    spark.sql(f"""
    COMMENT ON TABLE {table}
    IS '{comment}'
    """)

def fs_table_exists(table: str) -> bool:
    """
    Check if a table exists in the current database.

    Args:
        table (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    try:
        FeatureStoreClient().get_table(table)
        return True
    except:
        return False

def _registry_feature_table(table:str, fs_conf: dict, fs: FeatureStoreClient):
    source_tables = fs_conf.pop("source_tables") if "source_tables" in fs_conf else []
    # TODO: erase this when save function saves with delta function
    items_conf = list(fs_conf.items()) # parche
    for k, v in items_conf: # parche
        if not v: # empty or None
            print(f"drop '{k}' from config feature")
            fs_conf.pop(k)
    table_name = table.split(".")[1] # parche
    primary_keys = ["pk_customer", "tpk_release_dt"] if table_name.startswith("fs_cus") else ["pk_customer", "pk_account", "tpk_release_dt"] # parche
    timestamp_keys = ["tpk_release_dt"] # parche
    fs_conf["primary_keys"] = primary_keys # parche
    fs_conf["timestamp_keys"] = timestamp_keys # parche

    if fs_table_exists(table):
        current_data_sources = fs.get_table(table).custom_data_sources
        fs.register_table(
            delta_table=table,
            **fs_conf
        )
        if source_tables: # parche
            current_has_more = set(current_data_sources).difference(set(source_tables))
            new_has_more = set(source_tables).difference(set(current_data_sources))
            if current_has_more or new_has_more:
                if current_data_sources: # Solo borrar cuando hayan tablas que borrar
                    fs.delete_data_sources(
                        feature_table_name=table,
                        source_names=current_data_sources,
                    )
                fs.add_data_sources(
                    feature_table_name=table,
                    source_names=source_tables,
                    source_type="table"
                )
    else:
        fs.register_table(
            delta_table=table,
            **fs_conf
        )
        if source_tables: # parche
            fs.add_data_sources(
                feature_table_name=table,
                source_names=source_tables,
                source_type="table"
            )

def registry_feature_table(table:str):
    """Register a feature table in the Feature Store.
    
    Args:
        table (str): The name of the table to register.
    """
    print(f"[FEATURE REGISTRATION][TABLE METADATA] Reading metadata for table {table}")
    fs_conf = get_metadata(table)
    fs_conf.pop("partition_columns")
    path = fs_conf.pop("path")
    print("[FEATURE REGISTRATION][TABLE METADATA] values:")
    pprint(fs_conf)

    fs = FeatureStoreClient()

    try:
        print(f"[FEATURE REGISTRATION][PROCESS] Registering table {table}")
        _registry_feature_table(
            table=table,
            fs_conf=fs_conf,
            fs=fs,
        )
        print(f"[FEATURE REGISTRATION][SUCCESS] Table {table} registered successfully")
        
    except ValueError as e:
        if "already exists with a different schema" in str(e):
            print(f"\n[FEATURE REGISTRATION][SCHEMA CONFLICT] Schema Conflict Detected")
            print(f"[FEATURE REGISTRATION][SCHEMA CONFLICT] =======================")
            print(f"[FEATURE REGISTRATION][SCHEMA CONFLICT][INFO] Table {table} exists with different schema")
            print(f"[FEATURE REGISTRATION][SCHEMA CONFLICT][ACTION] Dropping existing table")
            fs.drop_table(table)
            
            print(f"\n[FEATURE REGISTRATION][TABLE RECREATION] Recreating Table")
            print(f"[FEATURE REGISTRATION][TABLE RECREATION] ==================")
            print(f"[FEATURE REGISTRATION][TABLE RECREATION][PROCESS] Creating new table with updated schema")
            spark.sql(f"""
                CREATE TABLE {table}
                USING DELTA
                LOCATION '{path}'
            """)
            
            print(f"\n[FEATURE REGISTRATION][REREGISTRATION] Attempting Table Re-registration")
            print(f"[FEATURE REGISTRATION][REREGISTRATION] ==============================")
            _registry_feature_table(
                table=table,
                fs_conf=fs_conf,
                fs=fs,
            )
            print(f"[REREGISTRATION][SUCCESS] Table {table} re-registered successfully")
        else:
            print(f"\n[FEATURE REGISTRATION][ERROR] Registration Failed for {table}")
            raise e

def registry_simulator(target):
    num_feature_tables = 2
    for i in tqdm(range(num_feature_tables), desc=f"Registry Progress for `{target}` tables"):
        print(f"\tRegistering table {i+1} of {num_feature_tables} for target `{target}`")
        time.sleep(60)
    print(f"\tRegistry for target `{target}` finished")

def request_databricks_api(
        url:str,
        headers:dict,
        params:dict=None
    ):
    """Request the Databricks API.

    Args:
        url (str): The URL of the Databricks API endpoint.
        headers (dict): The headers to include in the API request.
        params (dict, optional): The parameters to include in the API request. Defaults to None.

    Returns:
        requests.Response: The response from the Databricks API.

    Raises:
        Exception: If the API request fails with a status code other than 200.
    """
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}. {response.text}")
    return response

# COMMAND ----------

def get_job_params(
    DATABRICKS_HOST:str,
    DATABRICKS_TOKEN:str,
    job_name:str="feature-store",
    stage_tag:str="feature",
    environment:tp.Literal["dev", "staging", "prod"]="dev",
    ):
    """Retrieve the parameters of a Databricks job.

    Args:
        job_id (str): The ID of the job to retrieve parameters for.
        DATABRICKS_HOST (str): The Databricks host URL.
        DATABRICKS_TOKEN (str): The Databricks authentication token.
        job_name (str, optional): The name's job witch you want to search. Defaults to "feature-store".
        stage_tag (str, optional): The tag's job witch you want to search. Defaults to "feature".

    Returns:
        dict: A dictionary containing the job parameters.
    """
    HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/list"
    response = request_databricks_api(
        url = url,
        headers = HEADERS,
    )
    
    job_results = {'jobs': response.json().get('jobs', [])}
    print(f"\n{job_results}\n\n")
    for job_params in response.json().get("jobs", []):
        settings = job_params["settings"]
        print(f"job name: {settings['name']}")
        if settings["name"] == job_name and \
            settings.get("tags", {}).get("stage", "") == stage_tag and \
            settings.get("tags", {}).get("environment", "") == environment:
            response_job_detail = request_databricks_api( # to get more details of the job
                url = f"{DATABRICKS_HOST}/api/2.1/jobs/get",
                headers = HEADERS,
                params = { "job_id": job_params["job_id"] }
            )
            # if "settings" not in response_job_detail.json():
            #     print("\n\n no settings found")
            #     pprint(response_job_detail.json())
            settings = response_job_detail.json()["settings"]
            return {
                "job_id": job_params["job_id"],
                "name": settings["name"],
                "created_time": job_params["created_time"],
                "task_names": [
                    task["task_key"]
                    for task in settings.get("tasks", [])
                    if task.get("task_key", None) is not None
                ],
            }
    return {}


def check_task_status_per_job(
    job_id: int,
    tasks_to_check: list,
    DATABRICKS_HOST: str,
    DATABRICKS_TOKEN: str,
):
    """Check the status of tasks in a Databricks job.

    Args:
        job_id (int): The ID of the job to check.
        DATABRICKS_HOST (str): The Databricks host URL.
        DATABRICKS_TOKEN (str): The Databricks authentication token.

    Returns:
        dict: A dictionary with task names as keys and their state, result state, and state message as values.

    Raises:
        Exception: If the request to the Databricks API fails or no runs are found.
    """
    HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/list"
    params = {
        "job_id": job_id,
    }
    response = request_databricks_api(url=url, headers=HEADERS, params=params)
    runs = response.json().get("runs", [])
    if not runs:
        raise Exception("No runs found for the specified job ID.")

    latest_run_id = runs[0]["run_id"] # por defecto están ordenados de más reciente a más antiguo
    task_status_url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get"
    task_params = {
        "run_id": latest_run_id,
    }
    task_response = request_databricks_api(url=task_status_url, headers=HEADERS, params=task_params)

    tasks = task_response.json().get("tasks", [])
    tasks_status = {}
    task_time = {}
    for task in tasks:
        if task['task_key'] in tasks_to_check:
            if task['task_key'] in tasks_status and task_time.get(task['task_key'], 0) > task['start_time']:
                # Como una task puede repararse, en un mismo run puede haber varias veces informacion de una misma task
                # Si hay varias ejecuciones, nos aseguramos de coger el registro del task que más recientemente se ha ejecutado
                continue

            task_time[task['task_key']] = task['start_time']
            tasks_status[task['task_key']] = {
                'state': task['state']['life_cycle_state'],  
                'result_state': task['state'].get('result_state', 'N/A'),
                'state_message': task['state'].get('state_message', 'N/A')
            }

    return tasks_status


def check_job_status(
    job_id:int,
    DATABRICKS_HOST:str,
    DATABRICKS_TOKEN:str,
    ):
    """Check the status of a Databricks job.

    Args:
        job_id (str): The ID of the job to check.
        DATABRICKS_HOST (str): The Databricks host URL.
        DATABRICKS_TOKEN (str): The Databricks authentication token.

    Returns:
        tuple: A tuple containing the state, result state, and state message of the job.

    Raises:
        Exception: If the request to the Databricks API fails or no runs are found.
    """
    HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/list"
    params = { 
        "job_id": job_id,
        # "active_only": "true",
    }
    response = request_databricks_api(url=url, headers=HEADERS, params=params)

    runs = response.json().get("runs", [])
    if not runs:
        raise Exception("No runs found")

    # Tomar el ultimo run
    last_run = runs[0] # por defecto están ordenados de más reciente a más antiguo
    state = last_run.get("state", {}).get("life_cycle_state")
    result_state = last_run.get("state", {}).get("result_state") # has a value just when `state` is `TERMINATED`
    state_message = last_run.get("state", {}).get("state_message", "No error message provided")
    
    return state, result_state, state_message


def process_job(
    target:str,
    job_params,
    credentials,
    errors_dict:dict,
    ):
    """Process a single job and update its status.
    
    Args:
        target (str): The target workspace.
        job_params (dict): The parameters of the job.
        credentials (dict): The credentials for the Databricks workspace.
        errors_dict (dict): A dictionary to store errors encountered during job processing.

    Returns:
        bool: True if the job is finished, False otherwise.
    """
    # TODO: ponerlo en un yml o atomatizar su deteccion
    features_per_file = {
        # file_name: [feature_tables, ..., ...]
        "fs_comportamiento_transaccional": ["fs_cus_payment", "fs_cus_transfer", "fs_cus_withdrawal", "fs_cus_check", "fs_cus_deposit"],
        "fs_cus_channel_usage": ["fs_cus_channel_usage"],
        "fs_cus_jupiter": ["fs_cus_jupiter"],
        "fs_cus_sac_procedure": ["fs_cus_sac_procedure"],
        "fs_cus_financial_statement": ["fs_cus_financial_statement"],
        "fs_cus_exports": ["fs_cus_exports"],
        "fs_cus_imports": ["fs_cus_imports"],
        "fs_cus_company_roll_sat": ["fs_cus_company_roll_sat"],
        "fs_cus_company_sat": ["fs_cus_company_sat"],
        "fs_cus_company_provider_sat": ["fs_cus_company_provider_sat"],
        "fs_cus_beneficiary_sat": ["fs_cus_beneficiary_sat"],
        "fs_acc_credit_card_consumption": ["fs_acc_credit_card_consumption"],
        "fs_cus_credit_card_consumption": ["fs_cus_credit_card_consumption"],
        "fs_cus_acc_portfolio": ["fs_cus_portfolio", "fs_acc_portfolio"],
        "fs_cus_debit_card": ["fs_cus_debit_card"],
        "fs_cus_credit_portfolio": ["fs_cus_credit_portfolio"],
        "fs_cus_demographic": ["fs_cus_demographic"],
        "fs_cus_ivc": ["fs_cus_ivc"],
        "fs_cus_holding_products": ["fs_cus_holding_products"],
        "fs_acc_checking_account": ["fs_acc_checking_account"],
        "fs_cus_checking_account": ["fs_cus_checking_account"],
        "fs_acc_savings_account": ["fs_acc_savings_account"],
        "fs_cus_savings_account": ["fs_cus_savings_account"],
        "fs_cus_certificate_deposit": ["fs_cus_certificate_deposit"],
        "fs_cus_credit_risk": ["fs_cus_credit_risk"],
        "fs_cus_credit_risk_holder": ["fs_cus_credit_risk_holder"],
        "fs_cus_ext_credit_risk": ["fs_cus_ext_credit_risk"],
        "fs_cus_resul_campaign_credimax": ["fs_cus_resul_campaign_credimax"],
        "fs_cus_deposit_liabilities": ["fs_cus_deposit_liabilities"],
    }

    print("\n[STATUS CHECK] Retrieving Job Status")
    print("[STATUS CHECK] ====================")
    print(f"[STATUS CHECK][QUERY] Checking status for job {job_params['JOB_ID']}")
    tasks_status = check_task_status_per_job(
        job_id=job_params["JOB_ID"],
        tasks_to_check=list(job_params["TASKS"].keys()),
        DATABRICKS_HOST=credentials["HOST"],
        DATABRICKS_TOKEN=credentials["TOKEN"],
    )

    still_with_tasks = False
    tasks_terminated = []

    print("\n[TASK EXECUTION] Processing Tasks")
    print("[TASK EXECUTION] ================")
    for task_key, task_status in tasks_status.items():
        state = task_status["state"] # Estado del ciclo de vida del job
        result_state = task_status["result_state"] # Estado del resultado del job (cuando ya ha finalizado)
        state_message = task_status["state_message"] # Descriptivo del result_state
        
        print(f"[TASK EXECUTION][STATUS] Task: {task_key}")
        print(f"[TASK EXECUTION][STATUS] State: {state}")
        print(f"[TASK EXECUTION][STATUS] Result: {result_state}")
        
        if job_params["TASKS"][task_key]["STATUS"] == JobStatus.TERMINATED:
            print(f"[TASK EXECUTION][SKIP] Task {task_key} already terminated")
            strill_with_tasks |= False
            continue

        if state == JobStatus.TERMINATED.value:
            if result_state == JobResult.SUCCESS.value:
                print(f"\n[FEATURE REGISTRATION] Starting Feature Registration for {task_key}")
                print(f"[FEATURE REGISTRATION] =====================================")
                
                if is_testing:
                    print("[FEATURE REGISTRATION][SIMULATION] Running in test mode")
                    registry_simulator(target)
                else:
                    feature_tables = features_per_file.get(task_key, [])
                    if not feature_tables:
                        print(f"[FEATURE REGISTRATION][ERROR] No feature tables found for task {task_key}")
                        raise ValueError(f"No feature tables found for task '{task_key}'")
                    
                    for table_name in feature_tables:
                        print(f"[FEATURE REGISTRATION][PROCESS] Registering table {table_name}")
                        registry_feature_table(f"{fs_database}.{table_name}")
                        print(f"[FEATURE REGISTRATION][SUCCESS] Table {table_name} registered")
                
                tasks_terminated.append(task_key)
                still_with_tasks |= False
                
            elif result_state in (JobResult.FAILED.value, JobResult.CANCELED.value):
                print(f"[TASK EXECUTION][ERROR] Task {task_key} {result_state}")
                print(f"[TASK EXECUTION][ERROR] Reason: {state_message}")
                
                if target not in errors_dict:
                    errors_dict[target] = {}
                errors_dict[target][task_key] = {
                    "state": state,
                    "result_state": result_state,
                    "state_message": state_message
                }
                tasks_terminated.append(task_key)
                still_with_tasks |= False
        else:
            print(f"[TASK EXECUTION][PENDING] Task {task_key} still running")
            still_with_tasks |= True
    
    print(f"\n[EXECUTION SUMMARY] Process Status")
    print(f"[EXECUTION SUMMARY] ===============")
    print(f"[EXECUTION SUMMARY] Tasks completed: {len(tasks_terminated)}")
    print(f"[EXECUTION SUMMARY] Tasks pending: {len(tasks_status) - len(tasks_terminated)}")
    print(f"[EXECUTION SUMMARY] Tasks with errors: {len(errors_dict.get(target, {}))}")
    
    return still_with_tasks, tasks_terminated

# COMMAND ----------

def main(credentials:dict, jobs_params:dict):
    """Main function to check the status of jobs and execute the registry simulator.

    Args:
        credentials (dict): The credentials for the Databricks workspaces.
        jobs_params (dict): The parameters of the jobs.

    Returns:
        dict: A dictionary containing errors encountered during job processing.
    """
    print("\n[INITIALIZATION] Starting Feature Store Registration Process")
    print("[INITIALIZATION] ========================================")
    errors_dict = {}
    jobs_per_finish = len(credentials.keys())
    total_tasks = sum(len(job["TASKS"]) for job in jobs_params.values())
    print(f"[INITIALIZATION][CONFIG] Total jobs to process: {jobs_per_finish}")
    print(f"[INITIALIZATION][CONFIG] Total tasks to process: {total_tasks}")
    
    if total_tasks == 0:
        print("[INITIALIZATION][ERROR] No tasks found to process")
        raise ValueError("No tasks to process")
    
    iteration = 0
# with tqdm(desc="Overall Progress") as job_progress:
    while True:
        iteration += 1
        print(f"\n[JOB TRACKING] Iteration {iteration}")
        print(f"[JOB TRACKING] ==================")
        print(f"[JOB TRACKING][SUMMARY] Jobs remaining: {jobs_per_finish}")
        
        for target, job_params in jobs_params.items():
            if job_params["STATUS"] == JobStatus.TERMINATED:
                print(f"[JOB TRACKING][SKIP] Target {target} already terminated")
                continue
            
            if not job_params["TASKS"]:
                print(f"[JOB TRACKING][SKIP] Target {target} has no tasks")
                jobs_params[target]["STATUS"] = JobStatus.TERMINATED
                jobs_per_finish -= 1
                continue
                
            if job_params["STATUS"] in (JobStatus.PENDING, JobStatus.RUNNING):
                print(f"\n[TASK PROCESSING] Starting Task Processing for Target: {target}")
                print(f"[TASK PROCESSING] =========================================")
                job_still_with_tasks, tasks_terminated = process_job(
                    target = target,
                    job_params = job_params,
                    credentials = credentials[target],
                    errors_dict = errors_dict,
                )
                
                if job_still_with_tasks:
                    print(f"[TASK PROCESSING][STATUS] Target {target} has pending tasks")
                else:
                    print(f"[TASK PROCESSING][STATUS] Target {target} completed all tasks")
                
                for task in tasks_terminated:
                    jobs_params[target]["TASKS"][task]["STATUS"] = JobStatus.TERMINATED
                    print(f"[TASK PROCESSING][COMPLETE] Task {task} for target {target} finished")

                if not job_still_with_tasks:
                    jobs_params[target]["STATUS"] = JobStatus.TERMINATED
                    jobs_per_finish -= 1
                    print(f"[TASK PROCESSING][COMPLETE] All tasks for target {target} finished")
        
        print(f"\n[JOB TRACKING][PROGRESS] Iteration {iteration} completed")
        print(f"[JOB TRACKING][PROGRESS] {jobs_per_finish} jobs still in 'pending' or 'running' state")
        
        if jobs_per_finish == 0:
            print("\n[COMPLETION] Feature Store Registration Process Finished")
            print("[COMPLETION] =========================================")
            break
        else:
            print("\n[JOB TRACKING][WAIT] Pausing for 60 seconds before next iteration...")
            time.sleep(60)
    
    if errors_dict:
        print("\n[COMPLETION][ERROR] Some jobs encountered errors:")
        print("[COMPLETION][ERROR] ==============================")
        for t, e in errors_dict.items():
            print(f"\t- Target {t}: {len(e)} error(s)")
    else:
        print("\n[COMPLETION][SUCCESS] All jobs completed successfully")
        print("[COMPLETION][SUCCESS] ==============================")
    
    return errors_dict

# COMMAND ----------

if __name__ == "__main__":
    notebook_inputs = dict(dbutils.widgets.getAll())
    environment = notebook_inputs.get("env", None)
    this_job_target = notebook_inputs.get("target", None)
    if environment is None:
        raise ValueError("`env` key not found in notebook params")
    if this_job_target is None:
        raise ValueError("`target` key not found in notebook params")

    job_name_to_search = "demo-feature-store"
    
    databricks_credentials = DATABRICKS_CREDENTIALS[environment]
    pop_response = databricks_credentials.pop(this_job_target, None)
    if pop_response is None:
        raise ValueError(f"Target `{this_job_target}` not found in DATABRICKS_CREDENTIALS")

    # user = mail.split("@")[0]
    mail = notebook_inputs.get("run_as_user", None)
    print(f"\n\n\n{mail}\n\n")
    if not mail:
        raise ValueError("`run_as_user` key not found in notebook params")
    user = mail.split("@")[0]
    print(f"environment: {environment}")
    print(f"mail: {mail}")
    print(f"user: {user}")
    
    jobs_params = {}
    for key in databricks_credentials.keys():
        job_params = get_job_params(
            job_name=job_name_to_search if environment=="prod" else f"[dev {user}] {job_name_to_search}",
            stage_tag="feature",
            environment=environment,
            DATABRICKS_HOST=databricks_credentials[key]["HOST"],
            DATABRICKS_TOKEN=databricks_credentials[key]["TOKEN"],
        )
        print(f"{key}: ", end="")
        pprint(job_params)

        if not job_params:
            print(f"[WARNING] No job found for target {key}. Skipping...")
            continue
        
        jobs_params[key] = {}
        jobs_params[key]["JOB_ID"] = job_params["job_id"]
        jobs_params[key]["STATUS"] = JobStatus.PENDING
        jobs_params[key]["TASKS"] = {
            task_key: {"STATUS": JobStatus.PENDING}
            for task_key in job_params["task_names"]
            if task_key.startswith("fs_") \
                and (
                    not ("register" in task_key or "registry" in task_key)
                    and (True if is_testing else ("test" not in task_key))
                )
        }

    if len(jobs_params) == 0:
        print("[WARNING] No jobs found to process. Exiting...")
        errors = {}
    else:
        errors = main(databricks_credentials, jobs_params)

    if errors:
        message =  "\t- " + "\n\t- ".join([
            f"job for target {target} finished with the following tasks status:" + \
            "".join([
                f"\n\t\ttask {task}:" + \
                f"\n\t\t\tstate: {s['state']}" + \
                f"\n\t\t\tresult_state: {s['result_state']}" + \
                f"\n\t\t\tstate_message: {s['state_message']}" # suele dar la razon de la falla o de la cancelacion del job
                for task, s in errors_per_task.items()
            ])
            for target, errors_per_task in errors.items()
        ])
        raise Exception("Registry finished with the following error:\n" + message)
