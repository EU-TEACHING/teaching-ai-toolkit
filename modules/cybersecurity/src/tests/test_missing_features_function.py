import pandas as pd
from src.utils.preprocessing_utils import missing_features, main_transformer

od = {'num': [1, 3, 5, 6, 8], 'cat1': ['A', 'B', 'C', 'D', 'E'], 'cat2': ['A', 'B', 'C', 'D', 'E']}
odf = pd.DataFrame(od)
odf_tr = odf.copy()

td1 = {'num': [1, 2, 2, 5], 'cat1': ['B', 'C', 'A', 'E'], 'cat2': ['B', 'C', 'A', 'E']}
tdf1 = pd.DataFrame(td1)
tdf_tr1 = tdf1.copy()

td2 = {'num': [1, 2, 2, 5, 4], 'cat1': ['B', 'C', 'A', 'G', 'D'], 'cat2': ['B', 'C', 'A', 'G', 'D']}
tdf2 = pd.DataFrame(td2)
tdf_tr2 = tdf2.copy()

features = ['num', 'cat1', 'cat2']
transformer, total_features_to_transform = main_transformer(odf_tr)
remainder = [x for x in tdf1 if x not in total_features_to_transform]
tdf_tr1 = transformer.transform(tdf_tr1)
tdf_tr1_total = pd.concat([tdf_tr1, tdf1[remainder]], axis=1)

remainder2 = [x for x in tdf2 if x not in total_features_to_transform]
tdf_tr2 = transformer.transform(tdf_tr2)
tdf_tr2_total = pd.concat([tdf_tr2, tdf2[remainder]], axis=1)

tdf_tr1 = missing_features(tdf_tr1, transformer)
tdf_tr2 = missing_features(tdf_tr2, transformer)
k = 1
