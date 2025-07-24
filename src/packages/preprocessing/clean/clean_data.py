import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _dedupe_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """It transposes the dataframe, drops the duplicates, and then transposes it back.

    Args:
      df (pd.DataFrame): The dataframe you want to deduplicate

    Returns:
      A dataframe with the duplicated columns removed.
    """
    columns = list(df.columns)
    initial_shape = df.shape
    initial_dtypes = (
        pd.DataFrame(df.dtypes)
        .reset_index()
        .drop_duplicates("index")
        .set_index("index")
        .to_dict()[0]
    )
    new_df = df.T.groupby(level=0).first().T
    new_df = new_df.astype(dtype=initial_dtypes)
    new_cols = list(df.columns)
    diff = list(set(columns).difference(set(new_cols)))
    final_shape = new_df.shape
    if len(diff) > 0:
        logger.debug(
            f"Dropping the one of following list of features because they are duplicated: {diff}"
        )
        logger.debug(f"Shapes changes: {initial_shape} to {final_shape}")
    return new_df
