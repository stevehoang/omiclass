import pandas as pd
import pytest
from omiclass.normalize import quantile_normalize_with_target


def test_basic_functionality():
    data = pd.DataFrame({
        'Sample1': [1, 2, 3],
        'Sample2': [4, 3, 2]
    })
    target = pd.Series([10, 20, 30])
    expected = pd.DataFrame({
        'Sample1': [10, 20, 30],
        'Sample2': [30, 20, 10]
    })
    result = quantile_normalize_with_target(data, target)
    pd.testing.assert_frame_equal(result, expected)


# def test_axis_parameter():
#     # Test with axis = 1
#     pass


def test_dimension_compatibility():
    data = pd.DataFrame({
        'Sample1': [1, 2, 3, 4],
        'Sample2': [2, 3, 4, 5]
    })
    target = pd.Series([10, 20, 30])
    with pytest.raises(ValueError):
        quantile_normalize_with_target(data, target)


def test_invalid_axis():
    data = pd.DataFrame({
        'Sample1': [1, 2, 3],
        'Sample2': [2, 3, 4]
    })
    target = pd.Series([10, 20, 30])
    with pytest.raises(ValueError):
        quantile_normalize_with_target(data, target, axis=2)



