from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import activity_l2
from keras.utils import np_utils
import collections
import AppConfig
import numpy


class Classify:
    def __init__(self):
        """

        :param hidden_layers:
        :param epoch:
        :return:
        """
        hidden_layers = AppConfig.getHiddenLayer()
        epoch = AppConfig.getEpoch()
        self.model = Sequential()
        if isinstance(hidden_layers, (collections.Sequence, numpy.ndarray)):
            self.model.add(
                Dense(hidden_layers[0], activity_regularizer=activity_l2(),
                      input_dim=AppConfig.getNumFeatures() * AppConfig.getContextWindowSize(), activation='sigmoid'))
            # self.model.add(Dropout(0.3))
            for num in range(1, len(hidden_layers)):
                self.model.add(Dense(hidden_layers[num], activation='sigmoid'))
                # self.model.add(Dropout(0.3))
        else:
            # self.model.add(
            #     Dense(hidden_layers, activity_regularizer=activity_l2(),
            #           input_dim=13, activation='sigmoid'))
            self.model.add(
                Dense(hidden_layers, activity_regularizer=activity_l2(),
                      input_dim=AppConfig.getNumFeatures() * AppConfig.getContextWindowSize(), activation='sigmoid'))
            # self.model.add(Dropout(0.3))
        # self.model.add(Dense(2, activation='sigmoid'))
        self.model.add(Dense(AppConfig.getNumLanguages(), activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, Y):
        """

        :param X: list of feature vector [ [0.2,0.4,0.8] , [0.22,0.65,0.12] , [0.99,0.45,0.35] ]
        :param Y: and their corresponding class labels [1,2,3]
        :return: nothing
        """
        # print np_utils.to_categorical(Y)
        # To disable printing add verbose=0
        self.model.fit(X, np_utils.to_categorical(Y))

    def predict(self, feature):
        """
        :param feature:
        :return:list of subcanditates with probabilities
        """
        prediction_vector = self.model.predict(feature)
        probability = dict()
        total = len(prediction_vector)
        # For Sigmoid Neurons
        for predictions in prediction_vector:
            sum = 0
            for values in predictions:
                sum += values
            for lang in range(len(predictions)):
                predictions[lang] = predictions[lang] / sum
        # end
        for predictions in prediction_vector:
            for lang in range(len(predictions)):
                if lang in probability.keys():
                    probability[lang] += predictions[lang]
                else:
                    probability[lang] = predictions[lang]
        subCandidates = []
        for key in probability:
            tp = ((probability[key] / total), key)
            subCandidates.append(tp)
        subCandidates.sort()
        subCandidates.reverse()
        return subCandidates


# Predict Usage for a single feature Vector:
# print obj.predict(numpy.array([x_t[0]]))

# o = Classify()
# x_train = numpy.load("x_train1.npy")
# y_train = numpy.load("y_train1.npy")
# x_t = numpy.load("x_t1.npy")
# o.train(x_train, y_train)
# o.predict(x_t)

