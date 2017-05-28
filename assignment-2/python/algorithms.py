import os
from pprint import pprint

from sklearn import ensemble, preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import k_means
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn import preprocessing, model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import export_graphviz
from subprocess import check_call
from treeinterpreter import treeinterpreter as ti
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
from sklearn.tree import DecisionTreeClassifier

import validation






def pre_process(data):
    preprocessing.normalize(data['price_usd'], axis=1, norm='l2', copy=False)
    data = data.apply(lambda x: pd.factorize(x)[0])
    return data

def compare_classifiers(data):
    labels = [
        # 'srch_id',
        # 'site_id',
        # 'prop_id',
        # 'prop_starrating',
        # 'prop_review_score',
        # 'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        # 'position',
        'price_usd',
        # 'promotion_flag',
        # 'srch_saturday_night_bool',
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        # 'consumer',
        # 'Pclass',
        # 'score'
        # 'dcg_score'
    ]

    # data['dcg_score'] = data['dcg_score'].apply(lambda x: pd.factorize(x)[0])
    y = (data['score'])

    x = data[labels]
    # x = StandardScaler().fit_transform(x)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    # array = data.values
    # X = array[:,0:4]
    # Y = array[:,4]
    # validation_size = 0.20
    # seed = 7
    # X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    seed = 7
    scoring = 'accuracy'
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

def forest_of_trees(data):
    # import numpy as np
    # import matplotlib.pyplot as plt

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
        'promotion_flag',
        # 'srch_saturday_night_bool',
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        'consumer',
        'Pclass',
        # 'score'
        # 'dcg_score'
    ]

    # data['dcg_score'] = data['dcg_score'].apply(lambda x: pd.factorize(x)[0])
    y = (data['booking_bool'])

    X = data[labels]
    # x = StandardScaler().fit_transform(x)


    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    # Build a classification task using 3 informative features


    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

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
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        # 'consumer'
        # 'Pclass'
        # 'score'
    ]

    # data = data.apply(lambda x: pd.factorize(x)[0])

    y = (data['booking_bool'])

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

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------CLASIFICATIONS-------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

def randomForstClassifier(data):

    labels = [
        'srch_id',
        # 'site_id',
        'prop_id',
        # 'prop_starrating',
        # 'prop_review_score',
        # 'prop_brand_bool',
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
        # 'consumer'
        # 'Pclass'
        # 'score'
    ]

    # testdata = (testdata[labels])

    y = (data['score'])

    x = data[labels]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    # pprint(X_test)

    print("random forest")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto' , n_estimators=1000,max_depth=400,verbose=0 )

    rf.fit(X_train, y_train)

    predictions  = rf.predict(X_test)

    print('accuracy_score ',accuracy_score(y_test, predictions))
    print('confusion_matrix ', confusion_matrix(y_test, predictions))
    print('classification_report' , classification_report(y_test, predictions))

    prediction, bias, contributions = ti.predict(rf, X_test)
    print ("Prediction", prediction)
    print ("Bias (trainset prior)", bias)
    print ("Feature contributions:")
    for c, feature in zip(contributions[0],
                          labels):
        print (feature, c)

    # from sklearn import tree
    # i_tree = 0
    # for tree_in_forest in rf.estimators_:
    #     with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
    #         my_file = tree.export_graphviz(tree_in_forest, out_file=my_file)
    #     i_tree = i_tree + 1

    # check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

    # results = pd.DataFrame({'srch_id': X_test['srch_id'], 'prop_id': X_test['prop_id'], 'score': predictions})
    #
    # f = lambda x: x.sort('score', ascending=False)
    # ranked = results.groupby('srch_id', sort=False).apply(f)
    # ranked.reset_index(0, drop=True)
    #
    # # results.groupby('srch_id').sort('score')
    #
    # tocsv(ranked, 'predictions_RandomForestClassifier.csv')
    #

    # mse = mean_squared_error(y_test, predictions)
    # print("MSE: %.6f" % mse)

    # print("cross validation")
    # scores = cross_val_score(rf, data, data['score'], cv=10)
    # print(scores)
    # print( np.mean(cross_val_score(rf, X_train, y_train, cv=10)))



    # feature_importance = rf.feature_importances_
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #
    # sorted_idx = pd.np.argsort(feature_importance)
    # pos = pd.np.arange(sorted_idx.shape[0]) + .5
    # plt.subplot(1, 2, 2)
    # plt.barh(pos, feature_importance[sorted_idx], align='center')
    # plt.yticks(pos, labels)
    # plt.xlabel('Relative Importance')
    # plt.title('Variable Importance')
    # plt.tight_layout()
    # plt.show()



def knearestClassifier(data):
    labels = [
        'srch_id',
        # 'site_id',
        'prop_id',
        # 'prop_starrating',
        # 'prop_review_score',
        # 'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        # 'position',
        'price_usd',
        # 'promotion_flag',
        # 'srch_saturday_night_bool',
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        # 'consumer',
        # 'Pclass',
        # 'score'
        # 'dcg_score'
    ]

    # testdata = (testdata[labels])
    # data['dcg_score'] = data['dcg_score'].apply(lambda x: pd.factorize(x)[0])
    y = (data['score'])

    x = data[labels]
    # x = StandardScaler().fit_transform(x)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    # pprint(X_train)
    print("KNeighborsClassifier")

    knn = KNeighborsClassifier(n_jobs=-1,algorithm='auto',n_neighbors=5,weights='distance',leaf_size=400)

    # sfs1 = SFS  (knn,
    #            k_features=3,
    #            forward=False,
    #            floating=False,
    #            verbose=2,
    #            scoring='recall',
    #            cv=10,
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
    # #
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
    #
    # plt.title('Sequential Forward Selection (w. StdErr)')
    # plt.grid()
    # plt.show()



    # exit(0)

    # mse = mean_squared_error(y_test, knn.predict(X_test))
    # print("MSE: %.6f" % mse)

    predictions = knn.predict(X_test)

    results = pd.DataFrame({'srch_id': X_test['srch_id'], 'prop_id': X_test['prop_id'], 'score': predictions})

    f = lambda x: x.sort('score', ascending=False)
    ranked = results.groupby('srch_id', sort=False).apply(f)
    ranked.reset_index(0, drop=True)


    tocsv(ranked,'predictions_KNeighborsClassifier.csv')
    #

    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))



    # import nDCG
    #
    # nDCG.ndcg(X_test)

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

def decisiontreeClassifier(data):
    labels = [
        'srch_id',
        # 'site_id',
        'prop_id',
        # 'prop_starrating',
        # 'prop_review_score',
        # 'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        # 'position',
        'price_usd',
        # 'promotion_flag',
        # 'srch_saturday_night_bool',
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        # 'consumer',
        # 'Pclass',
        # 'score'
        # 'dcg_score'
    ]


    y = (data['score'])

    x = data[labels]


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)


    print("DecisionTreeClassifier")

    dtc = DecisionTreeClassifier(criterion="entropy",max_features="auto")

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

    dtc.fit(X_train, y_train)


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
    #
    # plt.title('Sequential Forward Selection (w. StdErr)')
    # plt.grid()
    # plt.show()



    # exit(0)

    # mse = mean_squared_error(y_test, knn.predict(X_test))
    # print("MSE: %.6f" % mse)
    predictions = dtc.predict(X_test)
    results = pd.DataFrame({'srch_id': X_test['srch_id'], 'prop_id': X_test['prop_id'], 'score': predictions})

    f = lambda x: x.sort('score', ascending=False)
    ranked = results.groupby('srch_id', sort=False).apply(f)
    ranked.reset_index(0, drop=True)

    tocsv(ranked, 'predictions_DecisionTreeClassifier.csv')


    # predictions = knn.predict(X_validation)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


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
def gradientBoosting(data):
    labels = [
        'srch_id',
        # 'site_id',
        'prop_id',
        # 'prop_starrating',
        # 'prop_review_score',
        # 'prop_brand_bool',
        'prop_location_score1',
        'prop_location_score2',
        # 'position',
        'price_usd',
        # 'promotion_flag',
        # 'srch_saturday_night_bool',
        # 'random_bool',
        # 'click_bool',
        # 'booking_bool',
        # 'price_usd_normalized',
        # 'consumer',
        # 'Pclass',
        # 'score'
        # 'dcg_score'
    ]

    y = (data['score'])

    x = data[labels]
    # x = StandardScaler().fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    print("GradientBoostingRegressor")


    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls','verbose' : 2}
    # params = {'max_depth': 2000, 'learning_rate': 0.01, 'loss': 'huber'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    # mse = mean_squared_error(y_test, clf.predict(X_test))
    # print("MSE: %.6f" % mse)

    predictions = clf.predict(X_test)

    results = pd.DataFrame({'srch_id': X_test['srch_id'], 'prop_id': X_test['prop_id'], 'score': predictions})

    f = lambda x: x.sort('score', ascending=False)
    ranked = results.groupby('srch_id', sort=False).apply(f)
    ranked.reset_index(0, drop=True)

    tocsv(ranked, 'predictions_GradientBoostingRegressor.csv')

    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))


    # feature_importance = clf.feature_importances_
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



def tocsv(data,name):
    df = pd.DataFrame(data)
    df.to_csv('../../../Data/' + name,mode = 'w', index=False)