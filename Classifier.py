from sklearn.neural_network import MLPClassifier
class Classifier:
    def __init__(self,hidden_layers=(5,2),epoch=500000):
        """

        :param hidden_layers:
        :param epoch:
        :return:
        """
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=hidden_layers, random_state=1)
    def train(self,X,Y):
        """

        :param X: list of feature vector [ [0.2,0.4,0.8] , [0.22,0.65,0.12] , [0.99,0.45,0.35] ]
        :param Y: and their corresponding class labels [1,2,3]
        :return: nothing
        """
        self.clf.fit(X,Y)
    def predict(self,feature):
        return self.clf.predict(feature)



