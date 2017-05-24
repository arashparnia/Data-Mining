from pprint import pprint

from sklearn import ensemble, preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import k_means
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import validation



def pre_process(data):


    preprocessing.normalize(data['price_usd'], axis=1, norm='l2', copy=False)
    data = data.apply(lambda x: pd.factorize(x)[0])
    return data

def feature_importance(data):
    labels = [
        'srch_id',
        'site_id',
        'prop_id',
        'prop_starrating',
        'prop_review_score',
        'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        'position',
        'price_usd',
        'promotion_flag',
        # 'srch_saturday_night_bool'
        'random_bool',
        'click_bool',
        'booking_bool',
        # 'price_usd_normalized',
        # 'consumer'
        # 'Pclass'
        # 'score'
    ]

    data = data.apply(lambda x: pd.factorize(x)[0])

    y = (data['score'])

    x = data[labels]

    X = StandardScaler().fit_transform(x)



    n_classes = np.unique(y).size


    svm = SVC(kernel='linear')
    cv = StratifiedKFold(2)

    score, permutation_scores, pvalue = permutation_test_score(
        svm, X, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

    print("Classification score %s (pvalue : %s)" % (score, pvalue))

    plt.hist(permutation_scores, 20, label='Permutation scores')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
             label='Classification Score'
                   ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.show()


def pipeline_anova(data):
    labels = [
        'srch_id',
        'site_id',
        'prop_id',
        'prop_starrating',
        'prop_review_score',
        'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        'position',
        'price_usd',
        'promotion_flag',
        # 'srch_saturday_night_bool'
        'random_bool',
        'click_bool',
        'booking_bool',
        # 'price_usd_normalized',
        # 'consumer'
        # 'Pclass'
        # 'score'
    ]

    data = data.apply(lambda x: pd.factorize(x)[0])

    y = (data['score'])

    x = data[labels]

    X = StandardScaler().fit_transform(x)



    # ANOVA SVM-C
    # 1) anova filter, take 3 best ranked features
    anova_filter = SelectKBest(f_regression, k=3)
    # 2) svm
    clf = svm.SVC(kernel='linear')

    anova_svm = make_pipeline(anova_filter, clf)
    print(anova_svm)
    # anova_svm.fit(X, y)
    # anova_svm.predict(X)


def featureselection(data):
    labels = [
        'srch_id',
        'site_id',
        'prop_id',
        'prop_starrating',
        'prop_review_score',
        'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        'position',
        'price_usd',
        'promotion_flag',
        # 'srch_saturday_night_bool'
        'random_bool',
        'click_bool',
        'booking_bool',
        # 'price_usd_normalized',
        # 'consumer'
        # 'Pclass'
        # 'score'
    ]

    data = data.apply(lambda x: pd.factorize(x)[0])

    y = (data['score'])

    x = data[labels]

    X = StandardScaler().fit_transform(x)



    # Create the RFE object and rank each pixel
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1,verbose=1)
    rfe.fit(X, y)
    ranking = rfe.ranking_.reshape(data.images[0].shape)

    # Plot pixel ranking
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Ranking of pixels with RFE")
    plt.show()

#-----------------------------CLASIFICATIONS

def randomForst_to_ndcg(data):

    labels = [
        # 'srch_id',
        # 'site_id',
        # 'prop_id',
        'prop_starrating',
        'prop_review_score',
        'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        # 'position',
        'price_usd',
        # 'promotion_flag',
        # 'srch_saturday_night_bool'
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        'consumer'
        # 'Pclass'
        # 'score'
    ]

    y = (data['score'])

    x = data[labels]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


    print("random forest")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto' , max_depth=200,verbose=1 )

    rf.fit(X_train, y_train)

    mse = mean_squared_error(y_test, rf.predict(X_test))
    print("MSE: %.6f" % mse)

    print("cross validation")
    scores = cross_val_score(rf, data, data['score'], cv=10)
    print(scores)
    print( np.mean(cross_val_score(rf, X_train, y_train, cv=10)))


    y_result = rf.predict(X_test)

    ndcgScore = validation.ndcg_score(y_test, y_result)
    print("ndcg: %.6f" % ndcgScore)

    feature_importance = rf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = pd.np.argsort(feature_importance)
    pos = pd.np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, labels)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    # plt.tight_layout()
    plt.show()

    yyr = pd.DataFrame(y_result)
    yyt = pd.DataFrame(y_test)

    print(yyr)
    print(yyt)


def gradientBoosting(data):
    labels = [
        # 'srch_id',
        # 'site_id',
        # 'prop_id',
        # 'prop_starrating',
        'prop_review_score',
        'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        # 'position',
        'price_usd',
        'promotion_flag',
        'srch_saturday_night_bool'
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        # 'consumer'
        # 'Pclass'
        # 'score'
    ]

    y = (data['score'])

    x = data[labels]
    x = StandardScaler().fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    print("GradientBoostingRegressor")


    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls',}
    # params = {'max_depth': 2000, 'learning_rate': 0.01, 'loss': 'huber'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.6f" % mse)

    y_result = clf.predict(X_test)

    ndcgScore = validation.ndcg_score(y_test, y_result)
    print("ndcg: %.6f" % ndcgScore)


    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = pd.np.argsort(feature_importance)
    pos = pd.np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, labels)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()





def knearestClassifier(data):
    labels = [
        # 'srch_id',
        # 'site_id',
        'prop_id',
        'prop_starrating',
        'prop_review_score',
        'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        # 'position',
        'price_usd',
        'promotion_flag',
        'srch_saturday_night_bool',
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        # 'consumer',
        # 'Pclass'
        # 'score'
        'dcg_score'
    ]


    y = (data['dcg_score'])

    x = data[labels]
    # x = StandardScaler().fit_transform(x)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    # pprint(X_train)
    print("KNeighborsClassifier")

    knn = KNeighborsClassifier(n_jobs=-1,algorithm='auto',n_neighbors=5,weights='distance',leaf_size=30)

    # sfs1 = SFS(knn,
    #            k_features=1,
    #            forward=False,
    #            floating=False,
    #            verbose=2,
    #            scoring='accuracy',
    #            cv=5,
    #            skip_if_stuck=True,
    #            n_jobs=-1,
    #            )
    #
    # sfs1.fit(X_train, y_train)

    knn.fit(X_train, y_train)


    # print('\nSequential Forward Selection (k=3):')
    # print(sfs1.k_feature_idx_)
    # print('CV Score:')
    # print(sfs1.k_score_)
    #
    # print(pd.DataFrame.from_dict(sfs1.get_metric_dict()).T)

    # ig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
    #
    # plt.ylim([0.8, 1])
    # plt.title('Sequential Forward Selection (w. StdDev)')
    # plt.grid()
    # plt.show()



    # print('Selected features:', sfs1.k_feature_idx_)
    #
    # fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')

    # plt.title('Sequential Forward Selection (w. StdErr)')
    # plt.grid()
    # plt.show()



    # exit(0)

    # mse = mean_squared_error(y_test, knn.predict(X_test))
    # print("MSE: %.6f" % mse)

    y_result = knn.predict(X_test)

    X_test = pd.DataFrame(X_test)
    X_test['prediction'] = y_result

    pprint(X_test)





    import nDCG
    # ndcgScore = nDCG.nDCG(float(y_result),float(y_test),[1])
    # ndcgScore = validation.ndcg_score(y_test, y_result)
    # print("ndcg: %.6f" % ndcgScore)


    # yr = y_result.tolist()
    # yt = y_test.tolist()
    # for i in range(10000):
    #     if yr[i] > 0 and yt[i] >0:
    #         print('prediction=', yr[i] , ' ground truth = ' , yt[i])



    # feature_importance = knn.feature_importances_
    # # make importances relative to max importance
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #
    # sorted_idx = pd.np.argsort(feature_importance)
    # pos = pd.np.arange(sorted_idx.shape[0]) + .5
    # plt.subplot(1, 2, 2)
    # plt.barh(pos, feature_importance[sorted_idx], align='center')
    # plt.yticks(pos, labels)
    # plt.xlabel('Relative Importance')
    # plt.title('Variable Importance')
    # plt.show()

