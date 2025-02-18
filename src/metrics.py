import numpy as np
# import torch

def hit_rate(recommended_list, bought_list):
    """был ли хотя бы 1 релевантный товар среди рекомендованных"""

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate

def hit_rate_at_k(recommended_list, bought_list, k=5):
    """был ли хотя бы 1 релевантный товар среди топ-k рекомендованных"""

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(bought_list, recommended_list)

    hit_rate_at_k = (flags.sum() > 0) * 1

    return hit_rate_at_k


def precision(recommended_list, bought_list):
    import numpy as np

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    import numpy as np
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:5]

    flags = np.isin(bought_list, recommended_list)

    precision_at_k = flags.sum() / len(recommended_list)

    return precision_at_k


def money_pricision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]

    flags = np.isin(recommended_list, bought_list)

    money_pricision_at_k = (flags * prices_recommended).sum() / prices_recommended.sum()

    return money_pricision_at_k


def recall(recommended_list, bought_list):
    """доля рекомендованных товаров среди релевантных """

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    import numpy as np
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(bought_list, recommended_list)

    recall_at_k = flags.sum() / len(bought_list)

    return recall_at_k


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)

    flags = np.isin(recommended_list, bought_list)

    money_recall_at_k = (flags * prices_recommended).sum() / prices_bought.sum()

    return money_recall_at_k


def ap_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    if not flags.sum(): return 0

    sum_ = np.array([
        precision_at_k(recommended_list, bought_list, k=i + 1)
        for i in range(k - 1) if flags[i]
    ]).sum()

    result = sum_ / flags.sum()

    return result


def map_at_k(recommended_list, bought_list, k=5):
    map_at_k = np.array(
        [ap_at_k(recommended_list[i], bought_list[i], k=k)
         for i in range(u)]
    ).sum() / len(recommended_list)

    return map_at_k

# def compute_gain(y_value: float, gain_scheme: str) -> float:
#     if gain_scheme == 'exp2':
#         gain = 2 * y_value - 1
#     elif gain_scheme == 'const':
#         gain = y_value
#     else:
#         raise ValueError(F'{gain_scheme} метод не поддерживается, только exp2 и const')
#     return float(gain)
# 
# def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
#     _, argsort = torch.sort(ys_pred, descending=True, dim=0)
#     ys_true_sorted = ys_true[argsort]
#     ret = 0
#     for idx, cor_y in enumerate(ys_true_sorted, 1):
#         gain = compute_gain(cur_y, gain_scheme)
#         ret += (2 ** gain - 1) / np.log2(1 + idx)
#     return ret
# 
# def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'const') -> float:
#     pred_dcg = dcg(ys_true, ys_pred, gain_scheme)
#     ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
# 
#     ndcg = pred_dcg / ideal_dcg
#     return ndcg
# 
# def reciprocal_rank(recommended_list, bought_list):
#     ranks=0
#     for item_rec in recommended_list:
#         for i, item_bought in enumerate(bought_list):
#             if item_rec == item_bought:
#                 ranks += 1 / (i+1)
#     return ranks / len(recommended_list)
