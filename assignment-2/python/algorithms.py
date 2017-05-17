from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import validation


def randomForst_to_ndcg(data):

    data = data.apply(lambda x: pd.factorize(x)[0])

    y = (data['score'])

    x = data[[
        'srch_id',
        'site_id',
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
        'consumer',
        # 'Pclass'
        # 'score'
    ]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)




    print("random forest")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto' , max_depth=200, )


    rf.fit(X_train, y_train)

    mse = mean_squared_error(y_test, rf.predict(X_test))
    print("MSE: %.4f" % mse)

    # print("cross validation")
    # scores = cross_val_score(rf, data, data['score'], cv=10)
    #
    # print(scores)


    # print( np.mean(cross_val_score(rf, X_train, y_train, cv=10)))



    y_result = rf.predict(X_test)

    ndcgScore = validation.ndcg_score(y_test, y_result)
    print("ndcg: %.4f" % ndcgScore)

    feature_importance = rf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = pd.np.argsort(feature_importance)
    pos = pd.np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, list(data))
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def gradientBoosting(data):
    data = data.apply(lambda x: pd.factorize(x)[0])

    y = (data['score'])

    x = data[[
        'srch_id',
        'site_id',
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
        'consumer',
        'Pclass'
        # 'score'
    ]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)

    print("GradientBoostingRegressor")


    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    # params = {'max_depth': 2000, 'learning_rate': 0.01, 'loss': 'huber'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)

    y_result = clf.predict(X_test)

    ndcgScore = validation.ndcg_score(y_test, y_result)
    print("ndcg: %.4f" % ndcgScore)


    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = pd.np.argsort(feature_importance)
    pos = pd.np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, list(data))
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()