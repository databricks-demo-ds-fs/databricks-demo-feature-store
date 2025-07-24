"""These contains functions for cleaning data in pandas."""

import logging
import typing as tp

import pandas as pd
import pyspark
import pyspark.sql.functions as f
from pyspark.sql.types import StringType
from unidecode import unidecode

logger = logging.getLogger(__name__)


def _detect_string_columns(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Detects string columns in a PySpark DataFrame.

    Args:
        df (DataFrame): The PySpark DataFrame to analyze.

    Returns:
        list: A list of column names that contain string data.
    """
    string_columns = []
    for field in df.schema.fields:
        # TODO: omit some types of columns like mails
        if isinstance(field.dataType, StringType):
            string_columns.append(field.name)
    return string_columns


def _replace_elements(
    somestring: str,
    elem_list: tp.Tuple = (
        ["á", "a"],
        ["Á", "A"],
        ["é", "e"],
        ["É", "E"],
        ["í", "i"],
        ["Í", "I"],
        ["ó", "o"],
        ["Ó", "O"],
        ["ú", "u"],
        ["Ú", "U"],
        ["ý", "y"],
        ["Ý", "Y"],
        ["à", "a"],
        ["À", "A"],
        ["è", "e"],
        ["È", "E"],
        ["ì", "i"],
        ["Ì", "I"],
        ["ò", "o"],
        ["Ò", "O"],
        ["ù", "u"],
        ["Ù", "U"],
        ["ä", "a"],
        ["Ä", "A"],
        ["ë", "e"],
        ["Ë", "E"],
        ["ï", "i"],
        ["Ï", "I"],
        ["ö", "o"],
        ["Ö", "O"],
        ["ü", "u"],
        ["Ü", "U"],
        ["ÿ", "y"],
        ["Ÿ", "Y"],
        ["â", "a"],
        ["Â", "A"],
        ["ê", "e"],
        ["Ê", "E"],
        ["î", "i"],
        ["Î", "I"],
        ["ô", "o"],
        ["Ô", "O"],
        ["û", "u"],
        ["Û", "U"],
        ["ã", "a"],
        ["Ã", "A"],
        ["õ", "o"],
        ["Õ", "O"],
        ["ñ", "n"],
        ["Ñ", "N"],
        ["@", "a"],
    ),
) -> str:
    """Replace elements in a string."""
    for elems in elem_list:
        if elems[0] in somestring:
            somestring = str(somestring).replace(elems[0], elems[1])
    return somestring


def _unidecode_strings(
    somestring: str,
    characters_to_replace=(
        "(",
        ")",
        "*",
        " ",
        ":",
        ".",
        "-",
        ";",
        "<",
        "?",
        "/",
        ",",
        "'",
        "____",
        "___",
        "__",
        "'",
        "&",
    ),
) -> str:
    """Unidecode string.

    It takes a string, converts it to unicode, then converts it to ascii,
    then replaces all the characters in the list
    with underscores.

    Args:
      somestring (str): The string you want to unidecode.
      characters_to_replace: a list of characters to replace with an underscore

    Returns:
      A string formatted.
    """
    if somestring is None:
        return somestring

    # somestring = somestring.lower()
    u = unidecode(somestring, "utf-8")
    formated_string = unidecode(u)
    for character in characters_to_replace:
        formated_string = formated_string.replace(character, "_")
    if "_" in formated_string:
        last_underscore_index = formated_string.rindex("_")
        if last_underscore_index == len(formated_string) - 1:
            formated_string = formated_string[:-1]
    formated_string = _replace_elements(formated_string)
    return formated_string


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns names.

    It takes a dataframe and returns a dataframe with the same columns, but with the column names
    standardized.

    Args:
      df (pd.DataFrame): The dataframe you want to unidecode

    Returns:
      A dataframe with the columns unidecoded.
    """
    columns = list(df.columns)
    columns = [_unidecode_strings(col) for col in columns]
    columns = [_replace_elements(col) for col in columns]
    columns = [col.lower() for col in columns]
    df.columns = columns
    return df


def trim_columns_dataframe(
    df: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    Trim whitespace from string columns in a PySpark DataFrame.

    It does in automatically, detecting string columns.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with trimmed string columns.
    """
    string_columns = _detect_string_columns(df)

    dict_ops = {
        col: f.trim(f.col(col)) for col in string_columns
    }
    df = df.withColumns(dict_ops)

    return df


def replace_no_alphanum_dataframe(
    df: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    Replaces all non-alphanumeric values ​​in a column by '_'.
    And lowercase all string values.

    It does in automatically, detecting string columns.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with alphanumeric strings separated by '_'.
    """
    string_columns = _detect_string_columns(df)
    # Characters to replace
    chars_special = [
        # white space
        " ", "\t", "\n", "\r", "\x0b", "\x0c", '\xa0', '\xad', '\x1a', '\x90',
        # parentheses and grouping
        "(", ")", "[", "]", "{", "}",
        # puntuacion
        ",", ":", ".",";", "¿", "?", "¡", "!",
        # accents
        "´", "`",
        # quotes
        "'", "‘",
        # slashes
        "/", "\\",
        # guiones
        "-", # "_",
        # math
        "<", ">", "=", "+", "°", "³",
        # coins
        "$", "¥",
        # others
        "ç", "ž", "æ", "°", "^", "§" , "&", "#", "%",
        "•", "š", "·", "‰", "©", "ª",
    ]
    chars_to_replace = "[\\"+'\\'.join(chars_special)+"]"

    # Replace all specified characters with an underscore
    df = df.withColumns({
        col: f.regexp_replace(f.col(col), chars_to_replace, "_")
        for col in string_columns
    })

    # Handle multiple underscores by replacing them with a single underscore
    df = df.withColumns({
        col: f.regexp_replace(f.col(col), "_+", "_")
        for col in string_columns
    })

    # Handle initial and final underscores (drop)
    df = df.withColumns({
        col: f.regexp_replace(f.col(col), r"^_+|_+$", "")
        for col in string_columns
    })

    return df


def standarize_string_values(
    df: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    Replaces all non-alphanumeric values ​​in a column by '_'.
    And lowercase all string values.

    It does in automatically, detecting string columns.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with alphanumeric strings separated by '_'.
    """

    string_columns = _detect_string_columns(df)
    # Lowercase the column values
    df = df.withColumns({
        col: f.lower(f.col(col)) for col in string_columns
    })

    df = replace_no_alphanum_dataframe(df)

    return df
