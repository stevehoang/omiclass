import pandas as pd

def all_finite_numeric(df):
    """
     Check if all elements in a pandas DataFrame are finite numeric values.

     This function iterates over each element in the DataFrame to determine if
     all values are either integers or floats and are not NaN (not a number).
     It is useful for validating data before performing operations that require
     numeric and non-missing values.

     Parameters
     ----------
     df : pandas.DataFrame
         The DataFrame to be checked for finite numeric values.

     Returns
     -------
     bool
         Returns True if all elements in the DataFrame are finite numeric values
         (int or float), and False otherwise.

     Examples
     --------
     >>> import pandas as pd
     >>> import numpy as np
     >>> data = pd.DataFrame({
     ...     'A': [1, 2, 3],
     ...     'B': [4.5, 5.5, np.nan]
     ... })
     >>> all_finite_numeric(data)
     False
     """
    res = df.map(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).all().all()

    return res


def bit_flip(input_value):
    """
    Flips input from 0 to 1 or from 1 to 0.

    This function takes an integer input, either 0 or 1, and returns the opposite value.
    If the input is 0, the function returns 1, and if the input is 1, it returns 0.
    An exception is raised if the input is not 0 or 1.

    Parameters:
    input_value (int): An integer that should be either 0 or 1.

    Returns:
    int: The flipped bit value (1 if the input is 0, and 0 if the input is 1).

    Raises:
    ValueError: If the input_value is not 0 or 1.

    Example:
    >>> bit_flip(0)
    1
    >>> bit_flip(1)
    0
    """
    if input_value not in [0, 1]:
        raise ValueError("Input must be 0 or 1")

    return 1 if input_value == 0 else 0