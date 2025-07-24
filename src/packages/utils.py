from constants import conf

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException
from delta.tables import DeltaTable
import typing as tp
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import gc
import time

import warnings
from pprint import pprint
import logging

logger = logging.getLogger(__name__)


def get_dbutils(spark):
    try:
        from pyspark.dbutils import DBUtils # pylint: disable=import-outside-toplevel
        dbutils = DBUtils(spark)
    except ImportError:
        import IPython # pylint: disable=import-outside-toplevel
        dbutils = IPython.get_ipython().user_ns["dbutils"]
    return dbutils

def get_datasets(
        datasets: list,
        tables_to_omit: list = None,
    ) -> list:
    """Retrieves a list of datasets after omitting specified tables.

    Args:
        datasets (list): A list of dataset names to include. If empty or contains "*",
            all datasets from the configuration will be included.
        tables_to_omit (list): A list of table names to omit. If contains "*",
            all tables will be omitted.

    Returns:
        list: A list of dataset names after omitting the specified tables.
    """
    if tables_to_omit is None:
        tables_to_omit = []

    datasets = datasets if len(datasets) > 0 else conf.datasets
    if "*" in datasets:
        datasets = conf.datasets
    datasets = datasets if len(datasets) > 0 else conf.datasets

    if "*" in tables_to_omit:
        tables_to_omit = datasets

    datasets = list(set(datasets).difference(set(tables_to_omit)))
    return conf.sort_datasets(datasets)
