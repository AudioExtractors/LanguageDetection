import numpy
numpy.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import activity_l2, activity_l1l2
from keras.models import load_model
# from keras.layers.normalization import BatchNormalization  # Batch Normalization can be added
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import collections
import AppConfig


class BinaryClassify:
    def __init__(self, labels=-1):
        hidden_layers = AppConfig.getBinaryHiddenLayer()
        self.model = Sequential()
        if isinstance(hidden_layers, (collections.Sequence, numpy.ndarray)):
            self.model.add(Dense(hidden_layers[0], input_dim=AppConfig.selBinaryFeatures, activation='sigmoid'))
            # self.model.add(Dropout(0.2))
            for num in range(1, len(hidden_layers)):
                self.model.add(Dense(hidden_layers[num], activation='sigmoid'))
                # self.model.add(Dropout(0.2))
        else:
            self.model.add(Dense(hidden_layers, input_dim=AppConfig.selBinaryFeatures, activation='sigmoid'))
            # self.model.add(Dropout(0.2))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        if labels==-1:
            labels = range(2)
        self.index = labels
        self.label = LabelBinarizer().fit(labels)

    def train(self, X, Y):
        output = self.label.transform(Y)
        output = numpy.append(1 - output, output, axis=1)
        print output.sum(axis=0)
        X, output = shuffle(X, output, random_state=10)
        self.model.fit(X, output, batch_size=AppConfig.getBinaryBatchSize(),
                                 nb_epoch=AppConfig.getBinaryNumberEpochs())

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
            tp = ((probability[key] / total), self.index[key])
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
