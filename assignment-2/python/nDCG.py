from math import log2
import pandas as pd
import numpy as np
from pprint import pprint

def dcg(data):
    data = pd.DataFrame(data)
    data['dcg_score'] =0
    data['dcg_index'] = list(range(data.shape[0]))
    data['dcg_index'] = data['dcg_index'] +1
    # data.reset_index()
    data.ix[data['click_bool'] == 1, 'dcg_score'] = 1
    data.ix[data['booking_bool'] == 1, 'dcg_score'] = pow(2,5)

    # pprint(data['dcg_index'] )
    data['dcg_index'].apply(log2)
    data['dcg_score'] = data['dcg_score'] / data['dcg_index']
    # data['dcg_score'] = data['dcg_score'] / log2((float(data.index + 1)))  # log2(data.index)
    # # pprint( data)
    # max_dcg = max(data['dcg_score'])
    # return ('max dcg =', max_dcg)

def ndcg(data,g = 'srch_id'):
    srch_id_groups = data.groupby(g)
    # dfs =[]
    sum_dcg_score_per_group = list()
    max_dcg_score_per_group =list()
    for name, group in srch_id_groups:
        dcg(group)
        max_dcg = max(group['dcg_score'])
        sum_dcg = sum(group['dcg_score'])
        # sum_dcg_score_per_group.append(sum_dcg)
        # max_dcg_score_per_group.append(max_dcg)
        # ndcg_scores =
        # dfs.append(group)
    # data = pd.concat(dfs)
    average_dcg_all_groups = np.average(sum_dcg_score_per_group)
    ndcg_scores = max_dcg_score_per_group/average_dcg_all_groups
    pprint(ndcg_scores)
    ndcg_score = np.average(ndcg_scores)
    pprint(ndcg_score)








def predictionToRelevancies(predictions,actuals):

    I  = sorted(range(len(predictions)), key=lambda k: predictions[k])

    relevancies = [predictions, I]
    return relevancies

# [~,I] = sortrows(predictions);
# unsorted = 1:length(predictions);
# indices(I) = unsorted;
#
# relevancies = [predictions,actuals(indices,3)];
#
# end




    # def ranking_precision_score(y_true, y_score, k=10):
#     """Precision at rank k
#     Parameters
#     ----------
#     y_true : array-like, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array-like, shape = [n_samples]
#         Predicted scores.
#     k : int
#         Rank.
#     Returns
#     -------
#     precision @k : float
#     """
#     unique_y = np.unique(y_true)
#
#     if len(unique_y) > 2:
#         raise ValueError("Only supported for two relevance levels.")
#
#     pos_label = unique_y[1]
#     n_pos = np.sum(y_true == pos_label)
#
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order[:k])
#     n_relevant = np.sum(y_true == pos_label)
#
#     # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
#     return float(n_relevant) / min(n_pos, k)
#
#
# def average_precision_score(y_true, y_score, k=10):
#     """Average precision at rank k
#     Parameters
#     ----------
#     y_true : array-like, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array-like, shape = [n_samples]
#         Predicted scores.
#     k : int
#         Rank.
#     Returns
#     -------
#     average precision @k : float
#     """
#     unique_y = np.unique(y_true)
#
#     if len(unique_y) > 2:
#         raise ValueError("Only supported for two relevance levels.")
#
#     pos_label = unique_y[1]
#     n_pos = np.sum(y_true == pos_label)
#
#     order = np.argsort(y_score)[::-1][:min(n_pos, k)]
#     y_true = np.asarray(y_true)[order]
#
#     score = 0
#     for i in xrange(len(y_true)):
#         if y_true[i] == pos_label:
#             # Compute precision up to document i
#             # i.e, percentage of relevant documents up to document i.
#             prec = 0
#             for j in xrange(0, i + 1):
#                 if y_true[j] == pos_label:
#                     prec += 1.0
#             prec /= (i + 1.0)
#             score += prec
#
#     if n_pos == 0:
#         return 0
#
#     return score / n_pos


# def dcg_score(y_true, y_score, k=10):
#     """Discounted cumulative gain (DCG) at rank k
#     Parameters
#     ----------
#     y_true : array-like, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array-like, shape = [n_samples]
#         Predicted scores.
#     k : int
#         Rank.
#     gains : str
#         Whether gains should be "exponential" (default) or "linear".
#     Returns
#     -------
#     DCG @k : float
#     """
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order[:k])
#
#
#
#     gains = y_true
#
#
#     # highest rank is 1 so +2 instead of +1
#     discounts = np.log2(np.arange(len(y_true)) + 2)
#     return np.sum(gains / discounts)
#
#
# def ndcg_score(y_true, y_score, k=10, gains="exponential"):
#     """Normalized discounted cumulative gain (NDCG) at rank k
#     Parameters
#     ----------
#     y_true : array-like, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array-like, shape = [n_samples]
#         Predicted scores.
#     k : int
#         Rank.
#     gains : str
#         Whether gains should be "exponential" (default) or "linear".
#     Returns
#     -------
#     NDCG @k : float
#     """
#     best = dcg_score(y_true, y_true, k, gains)
#     actual = dcg_score(y_true, y_score, k, gains)
#     return actual / best
