"""
When running tests first change the numerics in processing_utils -> get_numerics
"""

import unittest
import pandas as pd
import joblib
import os
from pandas.testing import assert_frame_equal
from src.utils.preprocessing_utils import main_transformer


class TestPreprocessingUtilsCase(unittest.TestCase):
    def setUp(self) -> None:
        od = {'cat1': ['A', 'B', 'C', 'D', 'E'],
              'num': [1, 3, 5, 6, 8],
              'data1': [0.1, 0.1, 0.1, 0.1, 0.1],
              'cat2': ['A', 'B', 'C', 'D', 'E'],
              'data2': [0.1, 0.1, 0.1, 0.1, 0.1]}
        self.odf = pd.DataFrame(od)
        self.odf_tr = self.odf.copy()

        td = {'cat1': ['B', 'C', 'A', 'E'],
              'num': [1, 2, 2, 5],
              'data1': [0.1, 0.1, 0.1, 0.1],
              'cat2': ['B', 'C', 'A', 'E'],
              'data2': [0.1, 0.1, 0.1, 0.1]}
        self.tdf = pd.DataFrame(td)
        self.tdf_tr = self.tdf.copy()

        td2 = {'cat1': ['B', 'C', 'A', 'G', 'D'],
               'data1': [0.1, 0.1, 0.1, 0.1, 0.1],
               'num': [1, 2, 2, 5, 4],
               'cat2': ['B', 'C', 'A', 'G', 'D'],
               'data2': [0.1, 0.1, 0.1, 0.1, 0.1]}
        self.tdf2 = pd.DataFrame(td2)
        self.tdf_tr2 = self.tdf2.copy()

        self.transformer = main_transformer(self.odf_tr)
        # Save the transformer
        transformer_name = "test_transformer.sav"
        joblib.dump(self.transformer, os.path.join('local_model_storage', transformer_name))

        # Expected dataframes while testing transformer
        self.expected_tdf_tr = pd.DataFrame(
            {'0': [0.00000, 0.14285714285714285, 0.14285714285714285, 0.5714285714285714],
             '1': [0, 0, 1, 0],
             '2': [1, 0, 0, 0],
             '3': [0, 1, 0, 0],
             '4': [0, 0, 0, 0],
             '5': [0, 0, 0, 1],
             '6': [0, 0, 1, 0],
             '7': [1, 0, 0, 0],
             '8': [0, 1, 0, 0],
             '9': [0, 0, 0, 0],
             '10': [0, 0, 0, 1],
             '11': [0.1, 0.1, 0.1, 0.1],
             '12': [0.1, 0.1, 0.1, 0.1]})
        self.expected_tdf_tr = self.expected_tdf_tr.astype('float64')

        self.expected_tdf2_tr = pd.DataFrame(
            {'0': [0.00000, 0.14285714285714285, 0.14285714285714285, 0.5714285714285714, 0.42857142857142855],
             '1': [0, 0, 1, 0, 0],
             '2': [1, 0, 0, 0, 0],
             '3': [0, 1, 0, 0, 0],
             '4': [0, 0, 0, 0, 1],
             '5': [0, 0, 0, 0, 0],
             '6': [0, 0, 1, 0, 0],
             '7': [1, 0, 0, 0, 0],
             '8': [0, 1, 0, 0, 0],
             '9': [0, 0, 0, 0, 1],
             '10': [0, 0, 0, 0, 0],
             '11': [0.1, 0.1, 0.1, 0.1, 0.1],
             '12': [0.1, 0.1, 0.1, 0.1, 0.1]})
        self.expected_tdf2_tr = self.expected_tdf2_tr.astype('float64')

    def test_transformer(self) -> None:
        """Test the transformer
        Failure cases:
             1) Fit the transformer again on the test data
             2) Not recognizing unknown categories
             3) When applying OneHotEncoding not adding 0s corresponding to a category that exists in odf (original
             dataframe) on which the transformer was fitted but doesn't exist in tdf (test dataframe). As a result a
             zero column corresponding to that category will not exist at the transformed dataframe
             4) Not adding in the sparse matrix - array the features in the right order
             (e.g.  Categorical-Numeric-Remainder instead of Numeric-Categorical-Remainder)
             5) Change the order of the numeric or categorical features or remainder features
             (e.g. Num2-Num1 instead of Num1-Num2)
             6) Missing values that will result to missing columns or rows when converting to dataframe
             7) Error when transforming from sparse matrix or array to dataframe
        Expected:
            1) Transform the test data by the already fitted transformer
            2) Recognize unknown categories and ignore
            3) Add 0s corresponding to a category that exists in odf  on which the transformer was fitted but doesn't
            exist in tdf.
            4) Features in the array have the order Numeric-Categorical-Remainder
            5) Keep the order of the numeric, categorical and remainder features
            6) Add all the transformed values in the array
            7) Convert sparse matrix or array to dataframe
          """
        transformed_tdf = self.transformer.transform(self.tdf_tr)
        transformed_tdf2 = self.transformer.transform(self.tdf_tr2)
        transformed_tdf.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        transformed_tdf2.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        print(transformed_tdf)
        print(transformed_tdf2)

        assert_frame_equal(transformed_tdf, self.expected_tdf_tr)
        assert_frame_equal(transformed_tdf2, self.expected_tdf2_tr)


if __name__ == '__main__':
    unittest.main()
