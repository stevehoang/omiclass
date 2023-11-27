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
     >>> data = pd.DataFrame({
     ...     'A': [1, 2, 3],
     ...     'B': [4.5, 5.5, np.nan]
     ... })
     >>> all_finite_numeric(data)
     False
     """

    res = df.map(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).all().all()

    return res
