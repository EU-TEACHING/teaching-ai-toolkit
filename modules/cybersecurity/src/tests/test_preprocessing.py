import pandas as pd
from src.utils.preprocessing_utils import main_transformer

od = {'cat1': ['A', 'B', 'C', 'D', 'E'], 'num': [1, 3, 5, 6, 8], 'data1': [0.1, 0.1, 0.1, 0.1, 0.1],
      'cat2': ['A', 'B', 'C', 'D', 'E'], 'data2': [0.1, 0.1, 0.1, 0.1, 0.1]}
odf = pd.DataFrame(od)
odf_tr = odf.copy()

td = {'cat1': ['B', 'C', 'A', 'E'], 'num': [1, 2, 2, 5], 'data1': [0.1, 0.1, 0.1, 0.1],
      'cat2': ['B', 'C', 'A', 'E'], 'data2': [0.1, 0.1, 0.1, 0.1]}
tdf = pd.DataFrame(td)
tdf_tr = tdf.copy()

td2 = {'cat1': ['B', 'C', 'A', 'G', 'D'], 'data1': [0.1, 0.1, 0.1, 0.1, 0.1], 'num': [1, 2, 2, 5, 4],
       'cat2': ['B', 'C', 'A', 'G', 'D'], 'data2': [0.1, 0.1, 0.1, 0.1, 0.1]}
tdf2 = pd.DataFrame(td2)
tdf_tr2 = tdf2.copy()

transformer, total_features_to_transform = main_transformer(odf_tr)
remainder = [x for x in tdf if x not in total_features_to_transform]
tdf_tr = transformer.transform(tdf_tr)
tdf_tr_total = pd.concat([tdf_tr, tdf[remainder]], axis=1)

remainder2 = [x for x in tdf2 if x not in total_features_to_transform]
tdf_tr2 = transformer.transform(tdf_tr2)
tdf_tr2_total = pd.concat([tdf_tr2, tdf2[remainder]], axis=1)
k = 1