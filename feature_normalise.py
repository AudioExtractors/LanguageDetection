import numpy as np

# Normalises training data for better training
class FeatureNormalise:
    """Normalises data using mean std normalisation
    Usage:
        Call function batchData to train on data.
        After whole data is received call fit and transform
    """
    def __init__(self, n_features):
        self.count_ = 0
        self.x = np.zeros(n_features, dtype=np.float64)
        self.x2 = np.zeros(n_features, dtype=np.float64)
        self.mu = 0
        self.sigma = 1

    def batchData(self, X):
        self.count_ = self.count_ + X.shape[0]
        self.x = self.x + np.sum(X, axis=0)
        self.x2 = self.x2 + np.sum(X*X, axis=0)

    def fit(self):
        self.mu = self.x/self.count_
        self.sigma = np.sqrt((self.x2 - 2*self.mu*self.x + self.count_*self.mu*self.mu)/(self.count_-1) )

    def transform(self, X):
        X = (X-self.mu)/self.sigma
        X[np.isnan(X)] = 0
        return X


# Tester
if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4]])
    print a
    sel = FeatureNormalise(2)
    sel.batchData(a)
    sel.fit()
    print sel.transform(a)
