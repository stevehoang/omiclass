import pandas as pd
import numpy as np


def quantile_normalize(df, axis = 0):
    """
    Perform quantile normalization on a pandas DataFrame.

    Quantile normalization is a technique used to transform the data so that
    the distribution of the values for each sample in a DataFrame is the same.
    This is particularly useful in gene expression data analysis, ensuring
    that the distribution of expression levels is consistent across samples.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where each column is a sample and each row is a feature.
        The DataFrame should contain numeric values only.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the same shape as `df`, where the values have been
        quantile normalized. The returned DataFrame retains the original
        index and column names.

    Example
    -------
    import pandas as pd
    data = pd.DataFrame({
    ...  'Sample1': [1, 2, 3],
    ...  'Sample2': [2, 3, 4]
    ... })
    normalized_data = quantile_normalize(data)
    print(normalized_data)

    Notes
    -----
    - Ensure that the input DataFrame contains no missing values or non-numeric
      data types, as these may lead to errors or unexpected behavior.
    - This function is designed for batch processing of samples. Individual
      sample normalization may require a different approach.
    - The normalization process does not distinguish between zeros and missing
      values; it treats all entries equally.

    See Also
    --------
    Pandas Documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    NumPy Documentation: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
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

    # sort data frame
    df_sorted = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_rank = df.rank(method='min').stack().astype(int)
    df_normalized = df_mean[df_rank].unstack()
    return df_normalized