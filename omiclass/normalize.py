import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from .utils import all_finite_numeric, bit_flip


def quantile_normalize(df: DataFrame, axis=0) -> DataFrame:
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


def quantile_normalize_with_target(df: DataFrame, target: Series, axis=0) -> DataFrame:
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
        raise ValueError("Dimensions of df are not compatible with target. df dimensions: {df.shape}, "
                         "target dimensions: {target.shape}. Did you select the correct axis?")

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


def fsqn_dataset(df_target: DataFrame, df_to_norm: DataFrame, axis=1) -> DataFrame:
    """
        Performs Feature-Specific Quantile Normalization (FSQN) on the given DataFrame (`df_to_norm`) against a target
        DataFrame (`df_target`).

        This function normalizes each feature in `df_to_norm` to match feature distributions of `df_target`. The
        normalization can be done either across rows (each column independtly, axis=0) or across columns (each row
        independently, axis=1). It checks that the dimensions and indices/columns (depending on the axis) of the two
        DataFrames match before proceeding with the normalization.

        Parameters:
        df_target (DataFrame): The target DataFrame whose distribution is to be used for normalization.
        df_to_norm (DataFrame): The DataFrame to be normalized.
        axis (int, optional): Axis along which to normalize the data. Default is 1 (columns).
            - axis=0: Normalize each column independently (the distribition is across rows for a given column).
            - axis=1: Normalize each row independently (the distribution is across columns for a given row) .

        Returns:
        DataFrame: A DataFrame with the same dimensions as `df_to_norm`, normalized according to `df_target`.

        Raises:
        ValueError: If `axis` is not 0 or 1.
        ValueError: If the dimensions of `df_target` and `df_to_norm` are not compatible.
        ValueError: If the row indices (when axis=1) or column names (when axis=0) do not match between `df_target` and
        `df_to_norm`.

        Example:
        >>> df_target = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df_to_norm = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
        >>> fsqn_result = fsqn_datasets(df_target, df_to_norm, axis=1)

        See Also
        --------
        Franks JM, et al. Feature specific quantile normalization endables cross-platform classification of molecular
        subtypes using gene expression data. Bioinformatics, 2018.
        """
    # check axes
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    # check that dimensions are compatible
    if df_target.shape[axis] != df_to_norm.shape[axis]:
        raise ValueError("Dimensions of df are not compatible with target. Target dimensions: {df_target.shape}, "
                         "df_to_norm dimensions: {df_to_norm.shape}. Did you select the correct feature_axis?")

    # do FSQN
    if axis == 1:
        if set(df_to_norm.index) != set(df_target.index):
            raise ValueError("The row indices of the two data frames do not match.")
        else:
            inds = df_to_norm.index
        fsqn_results = [quantile_normalize_with_target(df_to_norm.iloc[[i]], df_target.iloc[i], axis=axis)
                        for i in inds]
        fsqn_results = pd.concat(fsqn_results, axis=0)
    else:
        if set(df_to_norm.columns) != set(df_target.columns):
            raise ValueError("The column names of the two data frames do not match.")
        else:
            inds = df_to_norm.columns
        fsqn_results = [quantile_normalize_with_target(df_to_norm[[i]], df_target[i], axis=axis) for i in inds]
        fsqn_results = pd.concat(fsqn_results, axis=1)

    return fsqn_results
