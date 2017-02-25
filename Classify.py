from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import activity_l2
import AppConfig
import Audio
import os
import numpy


class Classify:
    def __init__(self, hidden_layers=(5, 2)):
        """

        :param hidden_layers:
        :param epoch:
        :return:
        """

        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], activity_regularizer=activity_l2(), input_dim=AppConfig.numFeatures,
                        activation='relu'))
        self.model.add(Dropout(0.3))
        for num in range(1, len(hidden_layers)):
            self.model.add(Dense(hidden_layers[num], activation='relu'))
            self.model.add(Dropout(0.3))
        self.model.add(Dense(AppConfig.getNumLanguages(), activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, Y):
        """

        :param X: list of feature vector [ [0.2,0.4,0.8] , [0.22,0.65,0.12] , [0.99,0.45,0.35] ]
        :param Y: and their corresponding class labels [1,2,3]
        :return: nothing
        """
        self.model.fit(X, Y)

    def predict(self, feature):
        """
        :param feature:
        :return:list of subcanditates with probabilities
        """
        prediction_vector = self.model.predict(feature)
        label = dict()
        total = len(prediction_vector)
        for frame_prediction in prediction_vector:
            if frame_prediction in label:
                label[frame_prediction] += 1
            else:
                label[frame_prediction] = 1
        subcandidates = []
        for key in label:
            tp = (float(label[key])/total, key)
            subcandidates.append(tp)
        subcandidates.sort()
        subcandidates.reverse()
        return subcandidates
