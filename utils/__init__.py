from . import visualize
from . import arguments
from . import augs

import os
import time

#################################################################################
### General Utils
#################################################################################
def timer_func(func, identifier=""):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {identifier}:{func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#################################################################################
### Classification
#################################################################################

def compute_classification_stats(tp, fp, tn, fn, matrix_savepath=None):
    precision = safe_divide(tp , (tp + fp))
    recall = safe_divide(tp , (tp + fn))
    f1 = safe_divide(tp , (tp + 0.5*(fp+fn)))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    balanced_accuracy = 0.5 * (tn/(tn + fp) + tp/(tp+fn))
    confusion_matrix = [[tn, fp], [fn, tp]]
    
    if matrix_savepath is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.matshow(confusion_matrix)

        for (i, j), z in np.ndenumerate(confusion_matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

        plt.ylabel("True Label")
        plt.xlabel("Predicted label")
        plt.title(f'Prec:{precision:.3f}|Rec:{recall:.3f}|F1:{f1:.3f}|Acc:{accuracy:.3f}')
        plt.savefig(matrix_savepath)
        plt.clf(); plt.close()
    return {'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy, 'balanced_accuracy':balanced_accuracy}


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def dict_merge(d1, d2, assert_unique_keys=True):
    '''
    Simple merge dict operation for non-nested dicts
    '''
    d = d1.copy()
    for k,v in d2.items():
        if (assert_unique_keys) and (k in d):
            raise ValueError(f"Found duplicate key {k}")
        d[k] = v
    return d

def dict_sum(d1, d2, assert_same=True):
    '''
    Mutates d1
    '''
    if not assert_same:
        raise NotImplementedError
    assert sorted(d1.keys())==sorted(d2.keys())
    for k in d2:
        d1[k] = d1[k] + d2[k]
    

def safe_divide(num, den):
    try:
        return num/den
    except ZeroDivisionError as e:
        return None
