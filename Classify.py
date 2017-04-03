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
import matplotlib.pyplot as plt


class Classify:
    def __init__(self):
        """
        :param hidden_layers:
        :param epoch:
        :return:
        """
        hidden_layers = AppConfig.getHiddenLayer()
        epoch = AppConfig.getTrainingDataSize()
        self.model = Sequential()
        if isinstance(hidden_layers, (collections.Sequence, numpy.ndarray)):
            self.model.add(Dense(hidden_layers[0], input_dim=AppConfig.selFeatures, activity_regularizer=activity_l2(),
                      activation='sigmoid'))
            # self.model.add(Dropout(0.2))
            # self.model.add(BatchNormalization())
            for num in range(1, len(hidden_layers)):
                self.model.add(Dense(hidden_layers[num], activation='sigmoid'))
                # self.model.add(Dropout(0.1))
                # self.model.add(BatchNormalization())
        else:
            # self.model.add(
            #     Dense(hidden_layers, activity_regularizer=activity_l2(),
            #           input_dim=13, activation='sigmoid'))
            self.model.add(
                Dense(hidden_layers, input_dim=AppConfig.selFeatures, activity_regularizer=activity_l2(),
                      activation='sigmoid'))
            # self.model.add(BatchNormalization())
        # self.model.add(Dense(2, activation='sigmoid'))
        self.model.add(Dense(AppConfig.getNumLanguages(), activation='sigmoid'))
        # self.model.add(BatchNormalization())
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])  # adam gave
        # better results, but adadelta used everywhere

    # def generator(self):
    #     sz = AudioIO.getFeatureDumpSize()
    #     for i in range(sz-1):
    #         X = numpy.load("Dump//dumpX_"+str(i)+".npy")
    #         Y = numpy.load("Dump//dumpY_"+str(i)+".npy")
    #         Ydash = []
    #         yield X, np_utils.to_categorical(Y, 2)

    def train(self, X, Y):
        """
        :param X: list of feature vector [ [0.2,0.4,0.8] , [0.22,0.65,0.12] , [0.99,0.45,0.35] ]
        :param Y: and their corresponding class labels [1,2,3]
        :return: nothing
        """
        # print np_utils.to_categorical(Y)
        # To disable printing add verbose=0
        # nb_epoch for number of epochs (Number of passes over complete Data)
        # batch_size for batch size
        # shuffle is true by default (shuffle batches)
        """self.model.fit_generator(self.generator(),
        samples_per_epoch=50000, nb_epoch=10)"""
        output = LabelBinarizer().fit(range(AppConfig.getNumLanguages())).transform(Y)
        X, output = shuffle(X, output, random_state=10)
        history = self.model.fit(X, output, batch_size=AppConfig.getBatchSize(),
                                 nb_epoch=AppConfig.getNumberEpochs())
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        # # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

    def predict(self, feature):
        """
        :param feature:
        :return: list of subcanditates with probabilities
        """
        prediction_vector = self.model.predict_proba(feature, verbose=0)
        probability = dict()
        total = len(prediction_vector)
        # For Sigmoid Neurons
        # for predictions in prediction_vector:
        #     sum = 0
        #     for values in predictions:
        #         sum += values
        #     for lang in range(len(predictions)):
        #         predictions[lang] = predictions[lang] / sum
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

    def load(self, name):
        del self.model
        self.model = load_model(name + ".h5")

    def save(self, name):
        self.model.save(name + ".h5")

# Predict Usage for a single feature Vector:
# print obj.predict(numpy.array([x_t[0]]))

# o = Classify()
# x_train = numpy.load("x_train1.npy")
# y_train = numpy.load("y_train1.npy")
# x_t = numpy.load("x_t1.npy")
# o.train(x_train, y_train)
# o.predict(x_t)
