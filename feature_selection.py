import numpy as np
from scipy import special
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelBinarizer

def _clean_nans(scores):
    """
    Fixes Issue #1240: NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores

def safe_mask(X, mask):
    """Return a mask which is safe to use on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask: array
        Mask to be used on X.

    Returns
    -------
        mask
    """
    mask = np.asarray(mask)
    if np.issubdtype(mask.dtype, np.int):
        return mask

    if hasattr(X, "toarray"):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask

def _chisquare(f_obs, f_exp):
        f_obs = np.asarray(f_obs, dtype=np.float64)
        k = len(f_obs)
        chisq = f_obs
        chisq -= f_exp
        chisq **= 2
        with np.errstate(invalid="ignore"):
            chisq /= f_exp
        chisq = chisq.sum(axis=0)
        return chisq, special.chdtrc(k - 1, chisq)

class FeatureSelection:
    """Select features according to the k highest scores.
    Scores are calculated using chi squared statistics.
    Batch training is possible
    Usage:
        Call function batchData to train on data.
        After whole data is received call fit and transform
    """
    def __init__(self, n_classes, n_features,k=10):
        self.count_ = 0
        self.k = k
        self.observed = np.zeros((n_classes, n_features), dtype=np.float64)
        self.feature_count = np.zeros((1, n_features), dtype=np.float64)
        self.class_prob = np.zeros((1, n_classes), dtype=np.float64)
        self.label = LabelBinarizer().fit( range(n_classes) )
        self.scores_ = np.zeros(n_features, dtype=np.float64)
        self.pvalues_ = np.zeros(n_features, dtype=np.float64)
        self.mask = np.zeros(n_features, dtype=bool)

    def batchData(self, X, y):
        X = X + 15  #to make each value positive
        if np.any( X < 0):
            raise ValueError("Input X must be non-negative.")
        Y = self.label.transform(y)
        if Y.shape[1] == 1:
            Y = np.append(1 - Y, Y, axis=1)
        self.observed = self.observed + np.dot(Y.T, X) # n_classes * n_features
        self.feature_count = self.feature_count + X.sum(axis=0).reshape(1, -1)
        self.class_prob = self.class_prob + Y.sum(axis=0).reshape(1, -1)
        self.count_ = self.count_ + X.shape[0]

    def fit(self):
        self.class_prob = self.class_prob/self.count_
        expected = np.dot(self.class_prob.T, self.feature_count)
        self.scores_, self.pvalues_ = _chisquare(self.observed, expected)
        scores = _clean_nans(self.scores_)
        self.mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1

    def transform(self, X):
        return X[:, safe_mask(X, self.mask)]
        
        
#Tester
if __name__ == "__main__":
    b=2
    X = np.array([ [1,1,1,1], [2,2,2,2], [1,1,3,3], [2,2,3,4] ])
    y = np.array( [0,1,2,3] )
    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)
    df = Y.shape[1]-1
    X1, y1 = X[:b], y[:b]
    X2, y2 = X[b:], y[b:]
    print X
    print "----using whole data------"
    sel = SelectKBest(chi2, k=2)
    sel = sel.fit(X,y)
    print sel.pvalues_
    print sel.scores_
    X_new = sel.transform(X)
    print X_new
    print "------with batch data------"
    f = FeatureSelection(X.shape[1],Y.shape[1],k=2)
    f.batchData(X1,y1)
    f.batchData(X2,y2)
    f.fit()
    print f.pvalues_
    print f.scores_
    X_b = f.transform(X)
    print X_b
