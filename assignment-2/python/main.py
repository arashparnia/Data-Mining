
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from boruta import BorutaPy
from pprint import pprint
# from matplotlib import interactive


import matplotlib.pyplot as plt
import rpy2.robjects as robjects
import pandas.rpy.common as com
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time




# interactive(True)




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
data_train = robjects.r.load('../../../Data/data_Pclass2.RData')
data_Pclass1 = com.load_data(data_train[0])
data = data_Pclass1.select_dtypes(['number'])
data =  data[[
             'score',
             'srch_id',
             'date_time',
             'site_id',
             'visitor_location_country_id',
             'prop_country_id',
             'prop_id',
             'prop_starrating',
             'prop_brand_bool',
             'prop_location_score2',
             'prop_log_historical_price',
             'price_usd',
             'promotion_flag',
             'srch_destination_id',
             'srch_length_of_stay',
             'srch_booking_window',
             'srch_adults_count',
             'srch_children_count',
             'srch_room_count',
             'srch_saturday_night_bool',
             'random_bool',
             'Pclass'
             ]]



X_train, X_test, y_train, y_test = train_test_split(data, data['score'], test_size=0.5, random_state=42)





# data_test = robjects.r.load('../../../Data/dataTEST_Pclass1.RData')
# data_Pclass1_test = com.load_data(data_test[0])
# data_test =  data_Pclass1_test.select_dtypes(['number'])
# data_test =  data_test[[
#              'srch_id',
#              'date_time',
#              'site_id',
#              'visitor_location_country_id',
#              'prop_country_id',
#              'prop_id',
#              'prop_starrating',
#              'prop_brand_bool',
#              'prop_location_score2',
#              'prop_log_historical_price',
#              'price_usd',
#              'promotion_flag',
#              'srch_destination_id',
#              'srch_length_of_stay',
#              'srch_booking_window',
#              'srch_adults_count',
#              'srch_children_count',
#              'srch_room_count',
#              'srch_saturday_night_bool',
#              'random_bool',
#              'Pclass'
#              ]]



# srch_id                        False
# date_time                      False
# site_id                        False
# visitor_location_country_id    False
# visitor_hist_adr_usd            True
# prop_country_id                False
# prop_id                        False
# prop_starrating                False
# prop_brand_bool                False
# prop_location_score2           False
# prop_log_historical_price      False
# position                       False
# price_usd                      False
# promotion_flag                 False
# srch_destination_id            False
# srch_length_of_stay            False
# srch_booking_window            False
# srch_adults_count              False
# srch_children_count            False
# srch_room_count                False
# srch_saturday_night_bool       False
# srch_query_affinity_score       True
# orig_destination_distance       True
# random_bool                    False
# click_bool                     False
# gross_bookings_usd              True
# booking_bool                   False
# Pclass                         False
# score                          False











# print(pd.isnull(data).sum() > 0)
#
# exit(0)



# pprint(data)
# data = data.iloc[:,0:23]
# pprint(data)
# data = pd.DataFrame(data)
#
#
# data = data.head()






# rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=7)
rf = RandomForestClassifier(n_jobs=-1, class_weight='auto',)


rf.fit(X_train, y_train)

# print( np.mean(cross_val_score(rf, X_train, y_train, cv=10)))



y_result = rf.predict(X_test)


plt.scatter(y_test,y_result)
plt.show()














exit(0)
# %matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data['score'], data['prop_location_score2'], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.show()


# plot_clusters(data, cluster.KMeans, (), {'n_clusters':60})



# plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})

# plot_clusters(data, cluster.MeanShift, (0.175,), {'cluster_all':False})

# plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':6})

# plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})

# plot_clusters(data, cluster.DBSCAN, (), {'eps':0.025})



# exit(0)
plt.scatter(data['score'], data['prop_location_score2'], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.show()



