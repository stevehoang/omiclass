import pandas as pd
import numpy as np
from .utils import all_finite_numeric, bit_flip


def quantile_normalize(df, axis=0):
    """
    Perform quantile normalization on a pandas DataFrame along the specified axis.

    Quantile normalization is a technique used to transform the data so that
    the distribution of the values for each row or column in a DataFrame is the same.
    This is particularly useful in gene expression data analysis, ensuring
    that the distribution of expression levels is consistent across samples or features.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where typically each column is a sample and each row is a feature.
        The DataFrame should contain numeric values only.
    axis : int, default 0
        The axis along which to normalize the DataFrame.
        - If 0, normalize each column independently.
        - If 1, normalize each row independently.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the same shape as `df`, where the values have been
        quantile normalized. The returned DataFrame retains the original
        index and column names.

    Raises
    ------
    ValueError
        If the `axis` argument is not 0 or 1, or if the DataFrame contains non-numeric data.

    Examples
    --------
    import pandas as pd
    data = pd.DataFrame({
        'Sample1': [1, 2, 3],
        'Sample2': [2, 4, 3]
    })
    normalized_data = quantile_normalize(data)
    print(normalized_data)

    Notes
    -----
    - The function does not handle missing values or non-numeric data types.
      Ensure that the input DataFrame contains no missing values or non-numeric
      data types, as these may lead to errors or unexpected behavior.
    - The normalization process does not distinguish between zeros and missing
      values; it treats all entries equally.

    See Also
    --------
    Wikipedia: https://en.wikipedia.org/wiki/Quantile_normalization
    """

    # check axes
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    # ensure that all values in df are numeric
    if not all_finite_numeric(df):
        raise ValueError("All values in df must be either int or float. NaN values are not allowed.")

    # generate distribution from sample means
    df_sorted = pd.DataFrame(np.sort(df.values, axis=axis), index=df.index, columns=df.columns)
    df_mean = df_sorted.mean(axis=bit_flip(axis))

    # quantile normalize
    df_normalized = quantile_normalize_with_target(df, df_mean, axis=axis)

    return df_normalized


def quantile_normalize_with_target(df, target, axis=0):
    """
        Perform quantile normalization on a pandas DataFrame using a target distribution.

        This function normalizes the data in a DataFrame such that the distribution of
        the values along the specified axis matches that of the target distribution. It
        is particularly useful in scenarios where you want to align the distribution of
        your data to a predefined target, such as in certain types of data standardization
        or normalization tasks.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame where typically each column is a sample and each row is a feature.
            The DataFrame should contain numeric values only.
        target : pandas.Series
            A Pandas Series representing the target distribution to which the DataFrame's
            values will be normalized.
        axis : int, default 0
            The axis along which to normalize the DataFrame.
            - If 0, normalize each column independently to match the target distribution.
            - If 1, normalize each row independently to match the target distribution.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the same shape as `df`, where the values have been
            quantile normalized to match the target distribution. The returned DataFrame
            retains the original index and column names.

        Raises
        ------
        ValueError
            If the `axis` argument is not 0 or 1.
            If the dimensions of `df` are not compatible with the length of `target`.

        Examples
        --------
        import pandas as pd
        data = pd.DataFrame({
            'Sample1': [1, 2, 3],
            'Sample2': [2, 3, 4]
        })
        target = pd.Series([10, 20, 30])
        normalized_data = quantile_normalize_with_target(data, target)
        print(normalized_data)

        Notes
        -----
        - Ensure that the input DataFrame and target Series contain numeric values only.
          Non-numeric types may lead to errors or unexpected behavior.
        - The function does not handle missing values; ensure that the input DataFrame
          and target Series contain no missing values.

        See Also
        --------
        quantile_normalize: Function for regular quantile normalization without a target distribution.
        """

    # check axes
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    # check that dimensions are compatible
    if df.shape[axis] != len(target):
        raise ValueError("Dimensions of df are not compatible with target. Did you select the correct axis?")

    # ensure that all values in df are numeric
    if not all_finite_numeric(df):
        raise ValueError("All values in df must be either int or float. NaN values are not allowed.")

    # sort the target distribution
    target_sorted = target.sort_values(ascending=True)
    target_sorted.index = np.arange(1, len(target_sorted) + 1)

    # calculate the ranks of the original values
    df_rank = df.rank(axis=axis, ascending=True)

    # replace the original values with values from the distribution of means
    df_normalized = df_rank.map(lambda x: target_sorted[x])

    return df_normalized