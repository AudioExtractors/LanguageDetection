import numpy as np

# Normalises training data for better training


def normalise(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma

# Converts test data to a format on which NN was trained


def normConv(X, mu, sigma):
    return (X-mu)/sigma

# Tester
if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4]])
    print a
    b, mu, sigma = normalise(a)
    print b, mu, sigma
    print normConv(a, mu, sigma)
