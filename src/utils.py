import numpy as np
from numba import njit
@njit()
def where_sum(arr,Sum):
    up = len(arr)
    low = 0
    test = up//2
    while up-low>1:
        if np.sum(arr[:test])<Sum:
            low = test
        else:
            up = test
        test = (up + low)//2
    return test
@njit()
def where(arr,value):
    up = len(arr)
    low = 0
    test = up//2
    while up-low>1:
        if arr[test]<value:
            low = test
        else:
            up = test
        test = (up + low)//2
    return test
@njit()
def find_nearest(val,Arr):
    if val<=Arr[0]:
        return 0
    ind0 = np.argwhere(val>=Arr)[-1,0]
    if ind0 == Arr.shape[0]-1:
        return ind0
    elif val-Arr[ind0]>Arr[ind0+1]-val:
        return ind0 + 1
    else:
        return ind0
    
def numbers_trans(f,N,seed = None):
    """
    Generates numbers according to a pdf with the transformation method. f is the inverse of the integral of the normalised pdf.
    Arguments:
    f: A function returning the inverse of the CDF of the desired pdf
    N: The number of random numbers to generate
    seed: The seed from which to generate the numbers. If None, then no seed is used.
    """
    rng = np.random.default_rng(seed = seed)
    x = rng.uniform(size = N)
    n = f(x)
    return n

def fit_line_to_points(x,y):
    a = (np.sum(x) * np.sum(y) - len(x) * np.sum(x*y)) / (-len(x) * np.sum(x*x) + np.sum(x) ** 2)
    b = (np.sum(y) - a * np.sum(x)) / len(x)
    return a,b
