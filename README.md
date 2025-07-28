# Databricks Feature Store Multi-Workspace Project

A comprehensive Feature Store implementation using Databricks Asset Bundles that demonstrates enterprise-grade feature engineering across multiple workspaces with automated cross-workspace registration and orchestration.

## Project Overview

This project implements a **multi-workspace Feature Store architecture** designed to:
- Process and manage customer-centric feature tables across specialized workspaces
- Demonstrate enterprise MLOps patterns with automated feature pipeline orchestration
- Handle cross-workspace feature registration for unified feature access
- Maintain separate development and production environments with identical configurations

## Architecture

### Multi-Workspace Distribution

In this project it is necessary to make the features available for more than one databricks workspace. So, taking advantage of the fact that it will still be necessary to deploy a registration process of the features in more than one workspace, we have decided to distribute the computational load of the Feature Tables execution.
The project uses two specialized Databricks Workspaces to distribute computational workload:

- **Workspace A (bncapers)**: Processes foundational customer features
  - Demographic features
  - Credit risk and bureau data features
  
- **Workspace B (medpag)**: Processes behavioral and transactional features
  - Product holding patterns
  - Payment behavior analysis
  - Transaction volume and patterns

### Cross-Workspace Feature Registration

A key innovation is the **cross-workspace feature registration system** that ensures features created in one workspace are accessible from all workspaces. The registration process works through a sophisticated orchestration system:

**How It Works:**

1. **Job Monitoring**: The [`main()`](notebooks/feature_store_register.py#L509) function continuously monitors feature generation (each 60 seconds) jobs across multiple workspaces using the Databricks Jobs API.

2. **Task Status Tracking**: With the tasks' status given by  [`check_task_status_per_job()`](notebooks/feature_store_register.py#L285) analyze if it has been successfully completed, failed or is still executing.

3. **Automatic Registration Trigger**: When [`process_job()`](notebooks/feature_store_register.py#L380) detects a completed task, it automatically triggers feature table registration using a predefined mapping in `features_per_file` dictionary.

4. **Cross-Workspace Registration**: [`registry_feature_table()`](notebooks/feature_store_register.py#L144) registers each feature table in the current workspace's Feature Store, making it accessible for ML workflows regardless of where it was originally created.

5. **Token-Based Authentication**: The system uses cross-workspace tokens stored in secrets to authenticate and register features from remote workspaces.

**Benefits:**
- **Unified feature discovery** across the organization
- **Consistent feature access** for ML workflows from any workspace
- **Centralized feature governance** and lineage tracking
- **Automatic orchestration** without manual intervention

## Project Structure

```
databricks-demo-feature-store/
├── config/                           # Environment-specific configurations
│   ├── dev/                         # Development environment configs
│   │   ├── feature-store-bncapers-resource.yml
│   │   └── feature-store-medpag-resource.yml
│   ├── prod/                        # Production environment configs
│   │   ├── feature-store-bncapers-resource.yml
│   │   └── feature-store-medpag-resource.yml
│   └── staging/                     # Staging environment (note implemented)
├── notebooks/                       # Core feature processing logic
│   ├── feature_store_register.py   # Cross-workspace registration orchestrator
│   ├── globals.py                   # Global configurations and credentials
│   ├── notebooks_orchestator.py    # Notebook execution orchestrator
│   ├── utils.py                     # FeatureStoreManager and utilities
│   ├── development/                 # Development notebooks
│   │   ├── fs_cus_credit_risk.py
│   │   ├── fs_cus_demographic.py
│   │   ├── fs_cus_holding_products.py
│   │   ├── fs_cus_payment_behavior.py
│   │   └── fs_cus_transactions.py
│   └── production/                  # Production notebooks (identical logic)
│       ├── fs_cus_credit_risk.py
│       ├── fs_cus_demographic.py
│       ├── fs_cus_holding_products.py
│       ├── fs_cus_payment_behavior.py
│       └── fs_cus_transactions.py
├── databricks.yml                   # Asset Bundle configuration
├── requirements.txt                 # Python dependencies
└── setup_configs.md               # Configuration guide
```

## Feature Tables

### Customer Demographic Features (`fs_cus_demographic`)

**Source**: `demo_db.clientes`

**Features Generated**:
- Basic demographic information (gender, marital status, education level)
- Geographic data (city, residence zone)
- Income metrics (monthly average income)
- Customer lifecycle metrics (account age in months)
- Customer segmentation data

### Credit Risk Features (`fs_cus_credit_risk`)

**Source**: `demo_db.buro_credito`

**Features Generated**:
- Bureau scores and risk ratings
- Credit inquiries and debt metrics
- Time-series credit behavior patterns
- Rolling statistics (3 and 6-month windows)
- Credit risk ratios and indicators

### Product Holding Features (`fs_cus_holding_products`)

**Source**: `demo_db.productos`

**Features Generated**:
- Product portfolio composition
- Product-specific amount aggregations
- Product holding indicators (binary flags)
- Rolling statistics for product amounts
- Product diversification metrics


### Payment Behavior Features (`fs_cus_payment_behavior`)

**Source**: `demo_db.pagos`

**Features Generated**:
- Payment history and patterns
- Credit utilization metrics
- Payment timing analysis
- Credit duration and closure patterns
- Payment performance indicators

### Transaction Features (`fs_cus_transactions`)

**Source**: `demo_db.transacciones`

**Features Generated**:
- Transaction volume and frequency
- Average transaction amounts
- Transaction recency metrics
- Rolling transaction statistics
- Customer activity patterns

## Technical Features

### Advanced Feature Engineering

Each feature table implements sophisticated feature engineering patterns:

**Time-Series Features**:
- Lag features (1, 3, 6 months) for trend analysis
- Rolling statistics (averages, standard deviations) for stability metrics
- Rate of change calculations for momentum indicators

**Business Logic Features**:
- Financial ratios and proportions
- Risk indicators and behavioral flags
- Product diversification and concentration metrics

**Schema Evolution Support**:
- Automatic schema validation and evolution
- Type safety with decimal-to-float and long-to-integer conversions
- Missing column detection and handling

### Orchestration and Monitoring

**Robust Execution Framework**:
- **Parameterized notebook execution with error handling**: Implemented in [`notebooks/notebooks_orchestator.py`](notebooks/notebooks_orchestator.py#L135) through the `feature_pipeline()` function, which handles parameter validation via [`preprocess_notebook_parameters()`](notebooks/notebooks_orchestator.py#L50) and executes notebooks using [`run_command()`](notebooks/notebooks_orchestator.py#L126)
- **Comprehensive logging with colored output for development**: Managed by the [`CustomLogger`](notebooks/utils.py#L182) class in `notebooks/utils.py`, featuring the [`ColoredFormatter`](notebooks/utils.py#L127) class and [`LogLevel`](notebooks/utils.py#L94) enum for environment-specific colored console output
- **Timeout management and user interaction for critical decisions**: Provided by the [`TimedInput`](notebooks/utils.py#L586) class in `notebooks/utils.py` with methods like [`get_yes_no_input()`](notebooks/utils.py#L732) for handling schema validation conflicts and user confirmations. It is used specifically for when missing columns are detected in the saving process.

**Data Quality and Validation**:
- **Schema validation before feature table updates**: Implemented through the [`FeatureStoreManager._validate_schema()`](notebooks/utils.py#L1510) method in `notebooks/utils.py`, which detects new columns, missing columns, and data type changes
- **Data preprocessing with accent removal and text normalization**: Handled by [`preprocessing_ingesting_tables()`](notebooks/utils.py#L464) function in `notebooks/utils.py`, utilizing `replace_no_alphanum_dataframe()` and `trim_columns_dataframe()` functions
- **Configurable overwrite and schema evolution options**: Managed by [`FeatureStoreManager.save()`](notebooks/utils.py#L1919) method with parameters like `force_overwrite`, `overwriteSchema`, and `omit_data_validation_errors`

**Cross-Workspace Coordination**:
- **Automated feature registration across multiple workspaces**: Orchestrated by `notebooks/feature_store_register.py` through the [`main()`](notebooks/feature_store_register.py#L509) function, which coordinates multiple workspace operations using [`process_job()`](notebooks/feature_store_register.py#L380) for each target workspace
- **Token-based authentication for secure cross-workspace access**: Managed via the credential system in [`notebooks/globals.py`](notebooks/globals.py#L14) with the `DATABRICKS_CREDENTIALS` dictionary and secret retrieval through `dbutils.secrets.get()`
- **Dependency management and task orchestration**: Implemented through [`get_job_params()`](notebooks/feature_store_register.py#L229), [`check_task_status_per_job()`](notebooks/feature_store_register.py#L285), and [`check_job_status()`](notebooks/feature_store_register.py#L341) functions in `notebooks/feature_store_register.py` for monitoring task execution and managing job dependencies

## Data Sources

The project processes data from five main source tables in the `demo_db` database:

| Source Table | Description | Related Feature Table |
|--------------|-------------|----------------------|
| `demo_db.clientes` | Customer master data with demographics | `fs_cus_demographic` |
| `demo_db.buro_credito` | Credit bureau data and risk scores | `fs_cus_credit_risk` |
| `demo_db.productos` | Product holdings and balances | `fs_cus_holding_products` |
| `demo_db.pagos` | Payment history and behavior | `fs_cus_payment_behavior` |
| `demo_db.transacciones` | Transaction details and patterns | `fs_cus_transactions` |

## Feature Store Standards

### Naming Conventions

**Table Names**: `fs_cus_{domain}` format
- `fs_cus_demographic`
- `fs_cus_credit_risk` 
- `fs_cus_holding_products`
- `fs_cus_payment_behavior`
- `fs_cus_transactions`

**Primary Keys**: Standardized across all tables
- `pk_customer`: Customer identifier
- `tpk_release_dt`: Time-series timestamp key

**Feature Prefixes**: Domain-specific identification
- `cdmg_`: Customer demographic
- `chr_`: Credit risk
- `chp_`: Customer holding products
- `cpym_`: Customer payment behavior
- `ctrx_`: Customer transactions

### Time-Series Design

All feature tables implement **time-series patterns** enabling:
- Point-in-time feature lookups for model training
- Historical feature evolution tracking
- Temporal feature joins with consistent timestamps
- Monthly feature refresh cycles with incremental updates

## Use Cases

This Feature Store implementation demonstrates:

1. **Customer Risk Assessment**: Combining credit bureau data with payment behavior for comprehensive risk models
2. **Product Recommendation**: Using holding patterns and transaction data for cross-sell opportunities
3. **Customer Lifecycle Management**: Tracking customer evolution through demographic and behavioral changes
4. **Fraud Detection**: Leveraging transaction patterns and behavioral anomalies
5. **Customer Segmentation**: Multi-dimensional customer profiling using all feature domains

## Getting Started

To set up this project in your environment, see [setup_configs.md](setup_configs.md) for detailed configuration instructions.

## Contributing

This project serves as a reference implementation for enterprise Feature Store patterns. Contributions should maintain the established patterns for:
- Multi-workspace coordination
- Feature engineering consistency
- Documentation and naming standards
- Error handling and monitoring approaches
