from pandas.io.parsers import read_csv
from rpy2.robjects import r, pandas2ri
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from boruta import BorutaPy
from pprint import pprint
# from matplotlib import interactive


import matplotlib.pyplot as plt
import rpy2.robjects as robjects
# import pandas.rpy.common as com
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import validation
# interactive(True)

import algorithms



# result for k nearest pclass2

# [2017-05-18 22:34:06] Features: 1/1 -- score: 0.95720270577Sequential Forward Selection (k=3):
# (0,)
# CV Score:
# 0.95720270577
#   avg_score     ci_bound                                          cv_scores  \
# 1  0.957203  0.000235988  [0.956929193487, 0.957102157803, 0.95719278311...
# 2  0.957203  0.000235988  [0.956929193487, 0.957102157803, 0.95719278311...
# 3  0.957203  0.000235988  [0.956929193487, 0.957102157803, 0.95719278311...
# 4  0.957203  0.000235988  [0.956929193487, 0.957102157803, 0.95719278311...
# 5  0.957203  0.000235988  [0.956929193487, 0.957102157803, 0.95719278311...
# 6  0.957187  0.000244217  [0.95690337682, 0.95708494358, 0.957184175189,...
# 7  0.955863  0.000256905  [0.955578121235, 0.95578526979, 0.955772475295...
# 8  0.948008  0.000192621  [0.947936387732, 0.947840906157, 0.94801673380...
# 9  0.947336  0.000627397  [0.946370176586, 0.947531050154, 0.94749165031...
#
#                    feature_idx      std_dev      std_err
# 1                         (0,)  0.000183607  9.18033e-05
# 2                       (0, 1)  0.000183607  9.18033e-05
# 3                    (0, 1, 2)  0.000183607  9.18033e-05
# 4                 (0, 1, 2, 6)  0.000183607  9.18033e-05
# 5              (0, 1, 2, 6, 7)  0.000183607  9.18033e-05
# 6           (0, 1, 2, 6, 7, 8)  0.000190009  9.50046e-05
# 7        (0, 1, 2, 3, 6, 7, 8)  0.000199881  9.99405e-05
# 8     (0, 1, 2, 3, 5, 6, 7, 8)  0.000149866  7.49329e-05
# 9  (0, 1, 2, 3, 4, 5, 6, 7, 8)  0.000488136  0.000244068




data = pd.read_csv('../../../Data/datacsv_Pclass3.csv') # change to pclass 2

data = algorithms.pre_process(data)


data = data.sample(1000)
# 2l22111gorithms.feature_importance(data)
# algorithms.featureselection(data)
# algorithms.pipeline_anova(data)
# algorithms.randomForst_to_ndcg(data)
#
# algorithms.gradientBoosting(data)

algorithms.knearestClassifier(data)

# data = pd.read_csv('../../../Data/datacsv_Pclass2.csv')
# algorithms.randomForst_to_ndcg(data)
#
# data = pd.read_csv('../../../Data/datacsv_Pclass3.csv')
# algorithms.randomForst_to_ndcg(data)
#
# data = pd.read_csv('../../../Data/datacsv_Pclass4.csv')
# algorithms.randomForst_to_ndcg(data)

# data = pd.read_csv('../../../Data/datacsv_Pclass5.csv')
# algorithms.randomForst_to_ndcg(data)
# algorithms.gradientBoosting(data)
# algorithms.knearestClassifier(data)
# data = pd.read_csv('../../../Data/datacsv_Pclass6.csv')
# algorithms.randomForst_to_ndcg(data)

exit(0)








# from rpy2.robjects import r, pandas2ri
#
# pandas2ri.activate()


# import pandas.rpy.common as com
#
#
#
# # datacsv_Pclass2 = r.data("../../../Data/datacsv_Pclass2.RData")
# import os
# os.getcwd()
#
# datacsv_Pclass1  = com.load_data('../../../Data/data_Pclass1.RData')
#
# datacsv_Pclass1.head()


# load .RData and converts to pd.DataFrame


# print(data.head())


# robject = robjects.r.load('../../../Data/data_Pclass1.RData')

# robject = robjects.r.data(robject)

# robject = r['Data_Pclass1']

# print(robject.head())


# pandas2ri.activate()
# data_train=pandas2ri.ri2py(robject)

# data = com.load_data(robject)
# data = pd.DataFrame(data=data_train)
# print(data.describe())
# exit(0)

# print(pd.isnull(data_Pclass1['price_usd_normalized']).sum() / len(data_Pclass1['price_usd_normalized']) )
# print(data_Pclass1['price_usd_normalized'])
# exit(0)
# data = data_Pclass1.select_dtypes(['number'])



# sns.set_context('poster')
# sns.set_color_codes()
# plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

# plt.scatter(y_test, y_result, **plot_kwds)

# plt.show()






















# %matplotlib inline
# sns.set_context('poster')
# sns.set_color_codes()
# plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}


# def plot_clusters(data, algorithm, args, kwds):
#     start_time = time.time()
#     labels = algorithm(*args, **kwds).fit_predict(data)
#     end_time = time.time()
#     palette = sns.color_palette('deep', np.unique(labels).max() + 1)
#     colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
#     plt.scatter(data['score'], data['prop_location_score2'], c=colors, **plot_kwds)
#     frame = plt.gca()
#     frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_yaxis().set_visible(False)
#     plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
#     plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
#     plt.show()


# plot_clusters(data, cluster.KMeans, (), {'n_clusters':60})



# plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})

# plot_clusters(data, cluster.MeanShift, (0.175,), {'cluster_all':False})

# plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':6})

# plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})

# plot_clusters(data, cluster.DBSCAN, (), {'eps':0.025})



# exit(0)
# plt.scatter(data['score'], data['prop_location_score2'], c='b', **plot_kwds)
# frame = plt.gca()
# frame.axes.get_xaxis().set_visible(False)
# frame.axes.get_yaxis().set_visible(False)
# plt.show()
