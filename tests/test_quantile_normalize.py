import pandas as pd
import numpy as np
import pytest
from omiclass.normalize import quantile_normalize


def test_basic_functionality():
    data = pd.DataFrame({
        'Sample1': [2, 1, 3],
        'Sample2': [4, 3, 2]
    })
    expected = pd.DataFrame({
        'Sample1': [2.5, 1.5, 3.5],
        'Sample2': [3.5, 2.5, 1.5]
    })
    result = quantile_normalize(data, axis=0)
    pd.testing.assert_frame_equal(result, expected)


def test_axis_parameter():
    # Create a test case for axis = 1
    data = pd.DataFrame({
        'Sample1': [1, 2, 6],
        'Sample2': [2, 3, 7]
    })
    expected = pd.DataFrame({
        'Sample1': [3.0, 3.0, 3.0],
        'Sample2': [4.0, 4.0, 4.0]
    })
    result = quantile_normalize(data, axis=1)
    pd.testing.assert_frame_equal(result, expected)


def test_non_numeric_data():
    data = pd.DataFrame({
        'Sample1': [1, 'a', 3],
        'Sample2': [2, 3, 4]
    })
    with pytest.raises(ValueError):
        quantile_normalize(data)


def test_missing_values():
    data = pd.DataFrame({
        'Sample1': [1, np.nan, 3],
        'Sample2': [2, 3, 4]
    })
    with pytest.raises(ValueError):
        quantile_normalize(data)


def test_empty_dataframe():
    data = pd.DataFrame()
    result = quantile_normalize(data)
    assert result.empty
