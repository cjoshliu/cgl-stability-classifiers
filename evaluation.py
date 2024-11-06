# evaluation.py


import numpy as np


def evaluate(predict, target):
    '''Returns dictionary of evaluation metrics

    Parameters
    ----------
    predict : np.ndarray
        Predicted labels
    
    target : np.ndarray
        Target labels

    Returns
    -------
    eval_metrics : dict
        Evaluation metrics, self-explanatory keys are
        'accuracy', 'PPV', 'NPV', 'sensitivity', 'specificity', 'F-score'
    '''
    confusion = confusion_matrix(predict, target)

    eval_metrics = {}
    eval_metrics['accuracy'] = np.sum(np.diag(confusion))/np.sum(confusion)
    eval_metrics['PPV'] = confusion[1, 1]/np.sum(confusion[1])
    eval_metrics['NPV'] = confusion[0, 0]/np.sum(confusion[0])
    eval_metrics['sensitivity'] = confusion[1, 1]/np.sum(confusion, axis=0)[1]
    eval_metrics['specificity'] = confusion[0, 0]/np.sum(confusion, axis=0)[0]
    eval_metrics['F-score'] = 2.0/(1.0/eval_metrics['PPV']+1.0/eval_metrics['sensitivity'])

    return eval_metrics


def confusion_matrix(predict, target):
    '''Returns confusion matrix of counts
    [[true negatives, false negatives], [false positives, true positives]]

    Parameters
    ---------
    predict : np.ndarray
        Predicted labels
    
    target : np.ndarray
        Target labels

    Returns
    -------
    confusion : np.ndarray
        Confusion matrix
    '''
    tn = np.sum((predict == target)&(target.astype(bool) == False))
    fn = np.sum((predict != target)&(target.astype(bool) == True))
    tp = np.sum((predict == target)&(target.astype(bool) == True))
    fp = np.sum((predict != target)&(target.astype(bool) == False))

    confusion = np.array([[tn, fn], [fp, tp]])

    return confusion
