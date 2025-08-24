from typing import Iterable
import numpy as np


def _ap_at_k(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if actual is None:
        return 0.0

    return score / min(len(actual), k)


def _rk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = sum([1 for r in actual if r in predicted]) / len(actual)

    return score


def map_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    return np.mean(
        [_ap_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None]
    )


def recall_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    return np.mean([_rk(a, p, k) for a, p in zip(actual, predicted)])


def hr_at_k(actual: Iterable, predicted: Iterable, k: int = 10) -> float:
    count = 0
    for i, actual_i in enumerate(actual):
        for p in predicted[i][:k]:
            if p in actual_i:
                count += 1
                break
    return count / len(actual)
def _ndcg_at_k(actual, predicted, k=10):
    """
    Compute NDCG@k for a single query/user.
    
    Args:
        actual: List of true relevant items.
        predicted: List of predicted items.
        k: Number of items to consider.
    
    Returns:
        Float representing NDCG@k score.
    """
    if predicted is None or isinstance(predicted, float) or not predicted:
        return 0.0
    if actual is None or not actual:
        return 0.0
    if isinstance(predicted, str):
        predicted = predicted.split()
    elif not isinstance(predicted, (list, tuple)):
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    
    # Compute DCG@k
    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)  # log2(i+2) since i starts at 0
    
    # Compute IDCG@k (ideal ranking)
    ideal_relevance = [1.0] * min(len(actual), k)  # Binary relevance
    idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_relevance)))
    
    return dcg / idcg if idcg > 0 else 0.0

def _mrr_at_k(actual, predicted, k=10):
    """
    Compute Reciprocal Rank for a single query/user, considering top-k items.
    
    Args:
        actual: List of true relevant items.
        predicted: List of predicted items.
        k: Number of items to consider.
    
    Returns:
        Float representing reciprocal rank score.
    """
    if predicted is None or isinstance(predicted, float) or not predicted:
        return 0.0
    if actual is None or not actual:
        return 0.0
    if isinstance(predicted, str):
        predicted = predicted.split()
    elif not isinstance(predicted, (list, tuple)):
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    
    for i, p in enumerate(predicted):
        if p in actual:
            return 1.0 / (i + 1.0)
    return 0.0
def ndcg_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    """
    Compute Mean NDCG@k across all queries/users.
    
    Args:
        actual: Iterable of lists containing true relevant items for each query/user.
        predicted: Iterable of lists containing predicted items for each query/user.
        k: Number of items to consider.
    
    Returns:
        Float representing mean NDCG@k score.
    """
    scores = [
        _ndcg_at_k(a, p, k)
        for a, p in zip(actual, predicted)
        if a is not None and p is not None and not isinstance(p, float)
    ]
    return np.mean(scores) if scores else 0.0

def mrr_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    """
    Compute Mean Reciprocal Rank at k across all queries/users.
    
    Args:
        actual: Iterable of lists containing true relevant items for each query/user.
        predicted: Iterable of lists containing predicted items for each query/user.
        k: Number of items to consider.
    
    Returns:
        Float representing mean MRR@k score.
    """
    scores = [
        _mrr_at_k(a, p, k)
        for a, p in zip(actual, predicted)
        if a is not None and p is not None and not isinstance(p, float)
    ]
    return np.mean(scores) if scores else 0.0