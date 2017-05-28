from pandas import scatter_matrix
from pandas.io.parsers import read_csv
from rpy2.robjects import r, pandas2ri
from sklearn import preprocessing, model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
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

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import validation
# interactive(True)

import algorithms
import nDCG





# data = pd.read_csv('../../../Data/data.csv',nrows = 1000)
data = pd.read_csv('../../../Data/data.csv')
# data = pd.DataFrame(data)
# data = data.sample(10000)

# pprint(data.head())

# testdata = pd.read_csv('../../../Data/TESTdatacsv.csv')
# testdata = testdata.sample(1)

# pprint(srch_id_groups.head())
# nDCG.ndcg(data)
#
# exit(0)

data = algorithms.pre_process(data)
# testdata = algorithms.pre_process(testdata)


# scatter_matrix(data)
# plt.show()
#
# exit(0)
# algorithms.forest_of_trees(data)

# algorithms.feature_importance(data)
# algorithms.featureselection(data)
# algorithms.pipeline_anova(data)


# algorithms.randomForstClassifier(data)
# algorithms.knearestClassifier(data)
# algorithms.decisiontreeClassifier(data)
algorithms.gradientBoosting(data)


# algorithms.compare_classifiers(data)




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
print('done')