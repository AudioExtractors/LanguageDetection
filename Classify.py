import numpy
numpy.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import activity_l2, activity_l1l2
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import collections
import AppConfig


class Classify:
    def __init__(self):
        hidden_layers = AppConfig.getHiddenLayer()
        self.model = Sequential()
        if isinstance(hidden_layers, (collections.Sequence, numpy.ndarray)):
            self.model.add(Dense(hidden_layers[0], input_dim=AppConfig.selFeatures, activity_regularizer=activity_l1l2(),
                      activation='relu'))
            # self.model.add(Dropout(0.2))
            for num in range(1, len(hidden_layers)):
                self.model.add(Dense(hidden_layers[num], activity_regularizer=activity_l1l2(), activation='relu'))
                # self.model.add(Dropout(0.1))
        else:
            self.model.add(
                Dense(hidden_layers, input_dim=AppConfig.selFeatures, activity_regularizer=activity_l1l2(),
                      activation='relu'))
        self.model.add(Dense(AppConfig.getNumLanguages(), activity_regularizer=activity_l1l2(), activation='softmax'))
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, Y):
        # print np_utils.to_categorical(Y)
        # To disable printing add verbose=0
        # nb_epoch for number of epochs (Number of passes over complete Data)
        # batch_size for batch size
        # shuffle is true by default (shuffle batches)
        output = LabelBinarizer().fit(range(AppConfig.getNumLanguages())).transform(Y)
        if output.shape[1] == 1:
            output = numpy.append(1 - output, output, axis=1)
        X, output = shuffle(X, output, random_state=10)
        self.model.fit(X, output, batch_size=AppConfig.getBatchSize(), nb_epoch=AppConfig.getNumberEpochs())

    def predict(self, feature):
        prediction_vector = self.model.predict_proba(feature, verbose=0)
        probability = dict()
        total = len(prediction_vector)
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

    def load(self, name):
        del self.model
        self.model = load_model(name + ".h5")

    def save(self, name):
        self.model.save(name + ".h5")

# Predict Usage for a single feature Vector:
# print obj.predict(numpy.array([x_t[0]]))
