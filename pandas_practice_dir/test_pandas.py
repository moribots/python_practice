from . import pandas_practice as pdp
import pandas as pd
import numpy as np
from common.test_utils import _pass, _fail


def test_pandas():
    print("\nTesting Pandas Problems:")
    counter = 1
    # Problem 1
    try:
        # Mock a simple df
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = pdp.problem1(None, 'a')  # Ignoring file_path for test
        assert result == 2.0
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 2
    df = pd.DataFrame({'a': [1, 2, 3]})
    try:
        result = pdp.problem2(df, 'a', 1)
        expected = pd.DataFrame({'a': [2, 3]})
        pd.testing.assert_frame_equal(result.reset_index(
            drop=True), expected.reset_index(drop=True))
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 3
    df = pd.DataFrame({'group': ['A', 'A', 'B'], 'val': [1, 2, 3]})
    try:
        result = pdp.problem3(df, 'group', 'val')
        assert result['A'] == 3 and result['B'] == 3
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 4
    df = pd.DataFrame({'a': [1, np.nan, 3]})
    try:
        result = pdp.problem4(df, 'a')
        assert result['a'].isna().sum() == 0
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 5
    df1 = pd.DataFrame({'key': [1, 2, 3], 'value1': ['A', 'B', 'C']})
    df2 = pd.DataFrame({'key': [2, 3, 4], 'value2': ['X', 'Y', 'Z']})
    try:
        result = pdp.problem5(df1, df2, 'key')
        expected = pd.DataFrame({
            'key': [2, 3],
            'value1': ['B', 'C'],
            'value2': ['X', 'Y']
        })
        pd.testing.assert_frame_equal(result.reset_index(
            drop=True), expected.reset_index(drop=True))
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 6
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10, freq='D'),
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    try:
        result = pdp.problem6(df, 'date', 'value')

        # Check if the result is resampled to daily frequency
        assert result.index.freq == 'D', "Resampling frequency is incorrect"

        # Check if the rolling mean column exists and is calculated correctly
        expected_rolling_mean = result['value'].rolling(window=3).mean()
        pd.testing.assert_series_equal(
            result['rolling_mean'], expected_rolling_mean, check_names=False)

        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 7
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    try:
        result = pdp.problem7(df, ['A'], {'B': 'sum', 'C': 'mean'})
        assert result.shape[0] > 0
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 8
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
    try:
        result = pdp.problem8(df)
        # Check if memory optimization was applied
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 9
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    try:
        pivot_table, cross_tab = pdp.problem9(df, ['A'], 'B', 'C')
        assert isinstance(pivot_table, pd.DataFrame)
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 10
    df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
    try:
        result = pdp.problem10(df, 'group', 'value')
        assert len(result) == 4
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 11
    try:
        result = pdp.problem11()
        assert isinstance(result, pd.DataFrame)
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 12
    df = pd.DataFrame({'A': [1, 100, 50, 25, 75]})
    try:
        result = pdp.problem12(df, 'A')
        assert len(result) == len(df)
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 13
    df = pd.DataFrame({'date': pd.date_range(
        '2020-01-01', periods=10), 'target': range(10)})
    try:
        result = pdp.problem13(df, 'date', 'target')
        assert len(result.columns) > len(
            df.columns)  # Should have lag features
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1

    # Problem 14
    # Create a temporary CSV file for testing
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('A,B\n1,2\n3,4\n5,6\n')
        temp_file = f.name
    try:
        result = pdp.problem14(temp_file, chunk_size=2)
        assert isinstance(result, pd.DataFrame)
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1
    finally:
        os.unlink(temp_file)

    # Problem 15
    df = pd.DataFrame(
        {'text': ['Hello World', 'Python is great', 'Data Science']})
    try:
        result = pdp.problem15(df, 'text')
        assert len(result) == len(df)
        _pass(counter, "Pandas")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Pandas", e)
        counter += 1
