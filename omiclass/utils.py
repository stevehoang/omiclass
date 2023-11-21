import pandas as pd
import numpy as np


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

    # set axes
    if axis == 0:
        a1 = 0
        a2 = 1
    elif axis == 1:
        a1 = 1
        a2 = 0
    else:
        raise ValueError("axis must be 0 or 1")

    # ensure that all values in df are numeric
    if not df.map(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).all().all():
        raise ValueError("All values in df must be either int or float. NaN values are not allowed.")

    # generate distribution from sample means
    df_sorted = pd.DataFrame(np.sort(df.values, axis=a1), index=df.index, columns=df.columns)
    df_mean = df_sorted.mean(axis=a2)
    df_mean = df_mean.sort_values(ascending=True)
    df_mean.index = np.arange(1, len(df_mean) + 1)

    # calculate the ranks of the original values
    df_rank = df.rank(axis=a1, ascending=True)

    # replace the original values with values from the distribution of means
    df_normalized = df_rank.map(lambda x: df_mean[x])

    return df_normalized
