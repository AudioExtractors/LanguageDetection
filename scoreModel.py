import Classifier
import AudioIO
import numpy as np
import Audio
import AppConfig
from matplotlib.pyplot import *
import datetime
import sys
import Classify


class scoreModel:
    def __init__(self, languages, featureSets, epoch):

        self.label = dict()
        self.languages = languages
        self.featureSets = featureSets
        self.epoch = epoch
        # self.classifier = Classifier.Classifier(AppConfig.getHiddenLayer(), AppConfig.getEpoch())
        self.classifier = Classify.Classify()
        for i, language in enumerate(languages):
            self.label[language] = i

    def populateFeatureVector(self):
        """
        :return: return list of number of files trained for each language
        """
        self.inputFeatureVector = []  # Feature Vector
        self.inputClassVector = []  # ClassVector
        languagesFeatures = []
        X = []
        Y = []
        noOfFilesTrained = []
        for language in self.languages:
            print language
            inputSize = 0
            flag = 0
            samples = AudioIO.getTrainingSamples(language, random="False")
            ct = 0
            for sample in samples:
                featureVector = sample.getContextFeatureVector()
                for frameFeature in featureVector:
                    if inputSize >= self.epoch:
                        noOfFilesTrained.append((language, sample.getIndex()))
                        # languagesFeatures.append(X)
                        flag = 1
                        break
                    X.append(frameFeature)
                    # print type(X), type(x[0])
                    # print (len(X) * len(X[0])*64) / float(1024 * 1024 * 8), "..MB"
                    Y.append(self.label.get(language))
                    inputSize += 1
                if flag == 1:
                    break
        # print X
        print "Fetched"
        X = self.normaliseFeatureVector(X)
        print "Normalised"
        print X
        self.assertFeatureVector(X, Y)
        # print X
        self.classifier.train(X, Y)
        return noOfFilesTrained

    def train(self):
        self.classifier.train(self.inputFeatureVector, self.inputClassVector)

    def assertFeatureVector(self, X, Y):
        if len(X) == len(Y):
            print "len Assert Pass", len(X)
        for frameFeature in X:
            if frameFeature.shape != X[0].shape:
                print "Dimension Assert Fail"
            for feature in frameFeature:
                if not feature >= 0.0 and feature <= 1.0:
                    print "fail"
        print "Dimension Assert Pas", X[0].shape

    def predict(self, audio):
        featureVector = audio.getContextFeatureVector()
        normFeatureVector = self.normaliseFeatureVector(featureVector)
        return self.classifier.predict(normFeatureVector)
        # return self.classifier.predict(featureVector)

    def normaliseFeatureVector(self, X):
        Xmin = np.min(X, axis=0)
        Xmax = np.max(X, axis=0)
        delta = np.subtract(X, Xmin)
        diff = np.subtract(Xmax, Xmin)
        for i, frame in enumerate(delta):
            for j, value in enumerate(frame):
                if diff[j] == 0.0:
                    delta[i][j] = 1.0
                else:
                    delta[i][j] = delta[i][j] / diff[j]
        return delta

    def plotFeature(self, language, type, fig, featNo, style, number):
        X = []
        figure(fig)
        if type == "Train":
            samples = AudioIO.getTrainingSamples(language, number)
            for sample in samples:
                featureVector = sample.getFeatureVector()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i == featNo:
                            X.append(feature)
        elif type == "Test":
            samples = AudioIO.getTestSamples(language, number)
            for sample in samples:
                featureVector = sample.getFeatureVector()
                for framefeature in featureVector:
                    for i, feature in enumerate(framefeature):
                        if i == featNo:
                            X.append(feature)
        plot(X, color=style[0], marker=style[1], alpha=style[2])

    def plotFeature2D(self, language, type, fig, featNo, featNo2, style, number):
        X = []
        Y = []
        figure(fig)
        if type == "Train":
            samples = AudioIO.getTrainingSamples(language, number)
            for sample in samples:
                featureVector = sample.getFeatureVector()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i == featNo:
                            X.append(feature)
                        if i == featNo2:
                            Y.append(feature)
        elif type == "Test":
            samples = AudioIO.getTestSamples(language, number)
            for sample in samples:
                featureVector = sample.getFeatureVector()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i == featNo:
                            X.append(feature)
                        if i == featNo2:
                            Y.append(feature)
        self.normaliseFeatureVector(X)
        plot(X, Y, color=style[0], marker=style[1], alpha=style[2])

    def showPlot(self):
        show()

    def analyse(self):
        """
        :return: list of percentage success with language
        """
        analysis = []
        bar_len = 60
        for language in self.languages:
            Total = AppConfig.getTestEpoch()
            success = 0
            print language
            for num in range(AppConfig.getTestEpoch()):
                progress = num * 100 / Total
                sys.stdout.write(str(progress) + " ")
                subcandidates = self.predict(Audio.Audio(AppConfig.getFilePathTest(language, num)))
                if subcandidates[0][1] == self.label[language]:
                    success += 1
            analysis.append((language, float(success * 100) / Total))
        return analysis

# ep = [3000,4000,5000,8000,10000,20000,50000]
"""ep=[9000]
hl=[(5, 2),(6, 2),(7, 2)]
J = []
K = []
for i in ep:
    AppConfig.epoch = i
    X=scoreModel(["en", "it"],["asd", "sdf", "asd"], AppConfig.getEpoch())
    Y = X.train()
    A = X.analyse()
    print A
    J.append(A[0][1])
    K.append(A[1][1])
    print "done epoch", i
plot(ep, J, "r-")
plot(ep, K, 'g-')
show()"""
a = datetime.datetime.now()
X = scoreModel(AppConfig.languages, ["asd", "sdf", "asd"], AppConfig.getEpoch())
print X.populateFeatureVector()
b = datetime.datetime.now()
c = b-a
print c.seconds
# X.train()
A = X.analyse()
print A
"""
X.plotFeature2D("en", "Train", 1, 1, 2, ("b", "o", 0.5), 100)
X.plotFeature2D("it", "Train", 1, 1, 2, ("g", "o", 0.3), 100)
X.plotFeature2D("en", "Test",1, 1, 2, ("pink", "o", 0.5), 2)
X.showPlot()"""
