# Configuration Guide

This guide outlines the specific files and sections you need to modify to configure this Feature Store project in your own Databricks workspaces.

## Prerequisites

- Two Databricks workspaces (one for bncapers, one for medpag workflows)
- Admin access to both workspaces for secret management
- Source tables available in `demo_db` database or equivalent

---

## 1. Workspace URLs Configuration

### File: `databricks.yml`

Update the workspace URLs for all targets:

```yaml
targets:
  dev_bnca_pers:
    workspace:
      host: https://adb-YOURWORKSPACE-A.azuredatabricks.net/  # ← UPDATE THIS
  
  dev_med_pag:
    workspace:
      host: https://adb-YOURWORKSPACE-B.azuredatabricks.net/  # ← UPDATE THIS
  
  prod_bnca_pers:
    workspace:
      host: https://adb-YOURWORKSPACE-A.azuredatabricks.net/  # ← UPDATE THIS
  
  prod_med_pag:
    workspace:
      host: https://adb-YOURWORKSPACE-B.azuredatabricks.net/  # ← UPDATE THIS
```

### File: `notebooks/globals.py`

Update the workspace URLs in the credentials section:

```python
DATABRICKS_CREDENTIALS = {
    "prod": {
        "prod_bnca_pers": {
            "HOST": "https://adb-YOURWORKSPACE-A.azuredatabricks.net/",  # ← UPDATE
        },
        "prod_med_pag": {
            "HOST": "https://adb-YOURWORKSPACE-B.azuredatabricks.net/",  # ← UPDATE
        },
    },
    "dev": {
        "dev_bnca_pers": {
            "HOST": "https://adb-YOURWORKSPACE-A.azuredatabricks.net/",  # ← UPDATE
        },
        "dev_med_pag": {
            "HOST": "https://adb-YOURWORKSPACE-B.azuredatabricks.net/",  # ← UPDATE
        },
    },
}
```

---

## 2. Authenticate
To establish the connection with Databricks, you must choose between the following two [authentication](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/authentication) options:
1. OAuth (preferred)
2. Personal Access Token

---

## 3. Cross-Workspace Token Configuration

### Why This Is Critical

Features created in Workspace A are not automatically accessible from Workspace B's Feature Store. The cross-workspace registration requires tokens from both workspaces to be available in each workspace.

### Step 1: Create Personal Access Tokens

1. **In Workspace A**: Create a Personal Access Token
2. **In Workspace B**: Create a Personal Access Token

### Step 2: Configure Secrets (Required in BOTH Workspaces)

Create the following secrets in **both workspaces**:

**Secret Scope**: `advanced_analytics`

**Required Secrets**:
```
dev_bnca_pers_credential: <Workspace A Token>
dev_med_pag_credential: <Workspace B Token>
prod_bnca_pers_credential: <Workspace A Token>
prod_med_pag_credential: <Workspace B Token>
```

Use the following code to register your tokens as secrets in each workspace:
```python
# Replace with the correct tokens
DATABRICKS_CREDENTIALS = {
    "prod": {
        "prod_bnca_pers": {
            "HOST": "https://adb-2855125147922183.3.azuredatabricks.net/",
            "TOKEN": "dapixxxxxxxxxxxxxxxxxxxxxxxxxxxxx-1"
        },
        "prod_med_pag": {
            "HOST": "https://adb-8667311077316568.8.azuredatabricks.net/",
            "TOKEN": "dapixxxxxxxxxxxxxxxxxxxxxxxxxxxxx-2"
        },
    },
    "dev": {
        "dev_bnca_pers": {
            "HOST": "https://adb-2855125147922183.3.azuredatabricks.net/",
            "TOKEN": "dapixxxxxxxxxxxxxxxxxxxxxxxxxxxxx-1"
        },
        "dev_med_pag": {
            "HOST": "https://adb-8667311077316568.8.azuredatabricks.net/",
            "TOKEN": "dapixxxxxxxxxxxxxxxxxxxxxxxxxxxxx-2"
        },
    },
}

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

config = Config(
    # prod_bnca_pers, prod_med_pag, prod_riesgo 
    host=DATABRICKS_CREDENTIALS["prod"]["prod_med_pag"]["HOST"],
    token=DATABRICKS_CREDENTIALS["prod"]["prod_med_pag"]["TOKEN"],
)
w = WorkspaceClient(config=config)

scope_name = "advanced_analytics"
w.secrets.create_scope(scope=scope_name)

for env, env_dict in DATABRICKS_CREDENTIALS.items():
    for env_workspace, creds in env_dict.items():
        secret_key = f"{env_workspace}_credential"
        w.secrets.put_secret(scope=scope_name, key=secret_key, string_value=creds["TOKEN"])
```

### Step 3: Verify Secret Configuration

Test in both workspaces:

```python
# Run this in both Workspace A and Workspace B
dbutils.secrets.get(scope="advanced_analytics", key="dev_bnca_pers_credential")
dbutils.secrets.get(scope="advanced_analytics", key="dev_med_pag_credential")
dbutils.secrets.get(scope="advanced_analytics", key="prod_bnca_pers_credential")
dbutils.secrets.get(scope="advanced_analytics", key="prod_med_pag_credential")
```

---

## 4. Storage and Database Configuration

### File: `notebooks/globals.py`

Update storage paths and database names:

```python
# Storage paths - UPDATE THESE
base_path_dev = "dbfs:/mnt/YOUR-MOUNT/WORKSPACE/YOUR-USER/PROJECTS/DATA_ENGINEERING_DEV"
base_path_prod = "dbfs:/mnt/YOUR-MOUNT/WORKSPACE/YOUR-USER/PROJECTS/DATA_ENGINEERING"

# Database names - UPDATE THESE
fs_db_dev = "your_feature_store_demo_dev"
fs_db_prod = "your_feature_store_demo_prod"
```

---

## 5. Email Notifications Configuration

### Files: All Resource Files in `config/`

Update email addresses in these files:
- `config/dev/feature-store-bncapers-resource.yml`
- `config/dev/feature-store-medpag-resource.yml`
- `config/prod/feature-store-bncapers-resource.yml`
- `config/prod/feature-store-medpag-resource.yml`

**Find and replace**:
```yaml
email_notifications:
  on_failure:
    - admin-mail@domain.com  # ← CHANGE TO YOUR EMAIL
```

### File: `databricks.yml`

Update user information:

```yaml
variables:
  run_as_user:
    default: "your-email@domain.com"  # ← UPDATE THIS

targets:
  dev_bnca_pers:
    permissions:
      - user_name: your-email@domain.com  # ← UPDATE THIS
    run_as:
      user_name: your-email@domain.com    # ← UPDATE THIS
    variables:
      run_as_user: your-email@domain.com  # ← UPDATE THIS
```
If you Authenticated with Personal Access Token method, you must use the email's user who generated the token.

Repeat for all targets (dev_med_pag, prod_bnca_pers, prod_med_pag).

---

## 6. Metastore Configuration (If Using External Metastore)

### Files: All Resource Files in `config/`

If you're using an external metastore, update these configurations:

```yaml
spark_conf:
  spark.hadoop.javax.jdo.option.ConnectionURL: jdbc:sqlserver://YOUR-METASTORE.database.windows.net:1433;database=hivemetastore  # ← UPDATE
  spark.hadoop.javax.jdo.option.ConnectionUserName: "{{secrets/YOUR-SCOPE/sqlusuariomstore}}"  # ← UPDATE SCOPE/KEY
  spark.hadoop.javax.jdo.option.ConnectionPassword: "{{secrets/YOUR-SCOPE/sqlpasswdmstore}}"   # ← UPDATE SCOPE/KEY
```

**If using Unity Catalog**, remove the metastore configurations entirely.

---

## 7. Source Data Configuration

### File: `notebooks/utils.py`

Update source table references if your tables are in a different database:

```python
source_datasets = [
    # description_name, table_name, database
    ("dim_customers", "clientes", "YOUR_DB"),        # ← UPDATE DATABASE NAME
    ("fact_products", "productos", "YOUR_DB"),       # ← UPDATE DATABASE NAME
    ("fact_credits_payment", "pagos", "YOUR_DB"),    # ← UPDATE DATABASE NAME
    ("fact_credit_risk", "buro_credito", "YOUR_DB"), # ← UPDATE DATABASE NAME
    ("fact_liabilities_transactions", "transacciones", "YOUR_DB"), # ← UPDATE DATABASE NAME
]
```

---

## 8. Cluster Configuration (Optional)

### Files: All Resource Files in `config/`

You may want to adjust cluster specifications based on your requirements or the node types you have available:

```yaml
_feature_clusters:
  job_clusters: 
    - job_cluster_key: fs_demo_Dadsv5_32_deploy
      new_cluster:
        node_type_id: Standard_D16s_v5  # ← ADJUST BASED ON NEEDS
    - job_cluster_key: fs_credit_behavior_Dsv5_32_deploy
      new_cluster:
        node_type_id: Standard_D16s_v5  # ← ADJUST BASED ON NEEDS
```

---

## 9. Deployment Steps

### Step 1: Validate Configuration

```bash
databricks bundle validate
```

### Step 2: Deploy to Development
You can deploy using either the Databricks VS Code extension or directly from the terminal:

- **Using the Databricks VS Code Extension**:  
  Open the Command Palette (`Ctrl+Shift+P`), search for "Databricks: Deploy Bundle", and select your target environment. Or use the UI of the databricks extension.

- **Using the Terminal**:  
  Run the following commands:
  ```bash
  databricks bundle deploy --target dev_bnca_pers
  databricks bundle deploy --target dev_med_pag
  ```

### Step 3: Deploy to Production

- **Using the Databricks VS Code Extension**:  
  Open the Command Palette (`Ctrl+Shift+P`), search for "Databricks: Deploy Bundle", and select your target environment. Or use the UI of the databricks extension.

- **Using the Terminal**:  
  Run the following commands:
  ```bash
  databricks bundle deploy --target prod_bnca_pers
  databricks bundle deploy --target prod_med_pag
  ```

---

## 10. Verification Checklist

After configuration, verify:

- [ ] Both workspaces can access all 4 required secrets
- [ ] Storage paths are accessible and writable
- [ ] Source tables exist and are readable
- [ ] Email notifications are working
- [ ] Bundle validation passes without errors
- [ ] Jobs deploy successfully in both workspaces

---

## Common Issues & Solutions

### Issue: Secret Not Found
**Solution**: Verify the secret scope name is exactly `advanced_analytics` and key names match exactly.

### Issue: Cross-Workspace Registration Fails
**Solution**: Ensure both workspace tokens are valid and have appropriate permissions. Test secret access in both workspaces.

### Issue: Storage Path Not Accessible
**Solution**: Verify mount points exist and paths are writable. Check path format (dbfs:/ vs /mnt/).

### Issue: Source Tables Not Found
**Solution**: Verify database and table names in `source_datasets`. Ensure tables exist and are readable.

---

## Next Steps

Once configured successfully:

1. Run a test deployment in development environment
2. Execute individual feature notebooks to verify functionality
3. Check that features are accessible from both workspaces
4. Set up monitoring and alerting as needed

For more details about the project structure and features, see [README.md](README.md).
