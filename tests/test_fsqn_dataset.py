import pytest
import pandas as pd
from omiclass.normalize import fsqn_dataset


def test_normal_operation_axis1():
    # Test the function under normal conditions
    df_target = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_to_norm = pd.DataFrame({'A': [8, 7], 'B': [6, 5]})
    result = fsqn_dataset(df_target, df_to_norm, axis=1)
    expected = pd.DataFrame({'A': [3, 4], 'B': [1, 2]})
    pd.testing.assert_frame_equal(result, expected)


def test_normal_operation_axis0():
    # Test the function under normal conditions
    df_target = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_to_norm = pd.DataFrame({'A': [8, 7], 'B': [6, 5]})
    result = fsqn_dataset(df_target, df_to_norm, axis=0)
    expected = pd.DataFrame({'A': [2, 1], 'B': [4, 3]})
    pd.testing.assert_frame_equal(result, expected)


def test_mismatched_dimensions():
    # Test the function with mismatched dimensions
    df_target = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_to_norm = pd.DataFrame({'A': [5, 6], 'B': [7, 8], 'C': [9, 10]})
    with pytest.raises(ValueError):
        fsqn_dataset(df_target, df_to_norm, axis=1)


def test_invalid_axis():
    # Test the function with an invalid axis
    df_target = pd.DataFrame({'A': [1, 2]})
    df_to_norm = pd.DataFrame({'A': [5, 6]})
    with pytest.raises(ValueError):
        fsqn_dataset(df_target, df_to_norm, axis=2)


def test_mismatched_indices():
    # Test the function with mismatched row indices
    df_target = pd.DataFrame({'A': [1, 2]}, index=[1, 2])
    df_to_norm = pd.DataFrame({'A': [5, 6]}, index=[3, 4])
    with pytest.raises(ValueError):
        fsqn_dataset(df_target, df_to_norm, axis=1)
