import numpy as np
from cleanlab.rank import get_label_quality_scores

def probability_mass_above_given_label_score(labels, pred_probs, adjust_pred_probs=False, alpha=.99):
    """Returns label quality scores based on predictions from an ensemble of models. Uses the total probability mass 
    assigned by model to these more-likely-than-given-label classes as a more monotonic score instead that 
    provides more information. Only usable for ``method='self_confidence'`` (see :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` for method desciption) and ``adjust_pred_probs=False``.
    
    Score is between 0 and 1:
    - 1 --- clean label (given label is likely correct).
    - 0 --- dirty label (given label is likely incorrect).
    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
    pred_probs_list : List[np.array]
      Each element in this list should be an array of pred_probs in the same format
      expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
      Each element of `pred_probs_list` corresponds to the predictions from one model for all examples.
    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
    alpha :
      Sort-of-hacky way to circumvent too many ties in pred_probs by makeing the score partially account for self-confidence * alpha term.
    Returns
    -------
    probability_mass_above_given_label_score : np.array
    """
        
    # self confidence
    self_confidence = get_label_quality_scores(labels, pred_probs, method="self_confidence", adjust_pred_probs=adjust_pred_probs)
    # probability mass above given label
    mass_above_given_label = np.sum(np.clip(pred_probs - self_confidence[:,np.newaxis], a_min=0, a_max=None), axis=1)
    return (1 - mass_above_given_label) * alpha + (self_confidence) * (1 - alpha)