import AudioIO
import Audio
import AppConfig
from matplotlib.pyplot import *
import datetime
import sys
import Classify
import psutil
import os
process = psutil.Process(os.getpid())


class scoreModel:
    def __init__(self, languages, epoch):
        self.label = dict()
        self.languages = languages
        # self.featureSets = featureSets
        self.epoch = epoch
        # self.classifier = Classifier.Classifier(AppConfig.getHiddenLayer(), AppConfig.getEpoch())
        self.classifier = Classify.Classify()
        self.inputFeatureVector = []  # Feature Vector
        self.inputClassVector = []  # ClassVector
        for i, language in enumerate(languages):
            self.label[language] = i
        self.mu = 0
        self.sigma = 0

    # Needs to be updated
    # def populateFeatureVector(self):
    #     """
    #     :return: return list of number of files trained for each language
    #     """
    #     languagesFeatures = []
    #     X = []
    #     Y = []
    #     noOfFilesTrained = []
    #     for language in self.languages:
    #         print "Fetching Data for ", language
    #         inputSize = 0
    #         flag = 0
    #         samples = AudioIO.getTrainingSamples(language, random="False")
    #         print "yes"
    #         ct = 0
    #         for sample in samples:
    #             featureVector = sample.getFullVector()
    #             for frameFeature in featureVector:
    #                 if inputSize >= self.epoch:
    #                     noOfFilesTrained.append((language, sample.getIndex()))
    #                     languagesFeatures.append(X)
    #                     flag = 1
    #                     break
    #             X.append(featureVector)
    #             print (len(X)*len(X[0])*64)/float(1024*1024*8), "..MB"
    #             Y.append(self.label.get(language))
    #             inputSize += 1
    #             if flag == 1:
    #                 break
    #
    #     print X
    #     print "Fetched Feature Vector.."
    #     X = self.normaliseFeatureVector(X)
    #     print "Normalised Feature Vector.."
    #     print "current memory usage : ", (process.memory_info().rss)/(1024*1024)
    #     self.assertFeatureVector(X, Y)
    #     print X
    #     self.classifier.train(np.array(X), Y)
    #     return noOfFilesTrained

    def train(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        # print "DumpSize: ", dumpSize
        for i in range(dumpSize):
            combineDumpLanguageFeature = np.array([])
            combineDumpLanguageLabel = np.array([])
            for language in self.languages:
                X = np.load("Dump//dumpX_"+language+str(i)+".npy")
                Y = np.load("Dump//dumpY_"+language+str(i)+".npy")
                if len(combineDumpLanguageFeature) == 0:
                    combineDumpLanguageFeature = X
                    combineDumpLanguageLabel = Y
                else:
                    combineDumpLanguageFeature = np.vstack((combineDumpLanguageFeature, X))
                    combineDumpLanguageLabel = np.concatenate((combineDumpLanguageLabel, Y))
            # print combineDumpLanguageLabel
            X, self.mu, self.sigma = self.normalise(combineDumpLanguageFeature)
            self.classifier.train(combineDumpLanguageFeature, combineDumpLanguageLabel)

    def createAudioDumps(self):
        for language in self.languages:
            AudioIO.dumpAudioFiles(language)

    # Needs to be updated
    def dumpFeatureVector(self):
        """
        :return: return list of number of files trained for each language
        """
        underFetching = True
        noOfFilesTrained = []
        for language in self.languages:
            X = []
            Y = []
            dumpLength = 0
            currentDumpSize = 0
            print "Fetching Data for ", language
            inputSize = 0
            flag = 0
            samples = AudioIO.getDumpTrainingSample(language)
            # print "current memory usage : ", (process.memory_info().rss)/(1024*1024)
            ct = 0
            gt = 0
            for sample in samples:
                # if ct % 100 == 0:
                #     print inputSize
                #     print ct
                ct += 1
                featureVector = sample.getAverageFeatureVector(std=True)
                if len(featureVector) > 0:
                    featuresPerFrame = len(featureVector[0])
                else:
                    continue
                for frameFeature in featureVector:
                    if inputSize >= self.epoch:
                        underFetching = False
                        print "Created dump:-"+str(dumpLength)
                        np.save("Dump\\dumpX_"+language+str(dumpLength), X)
                        np.save("Dump\\dumpY_"+language+str(dumpLength), Y)
                        # print "Created All Dumps.."
                        noOfFilesTrained.append((language, sample.getIndex()))
                        flag = 1
                        break
                    if currentDumpSize + featuresPerFrame > AppConfig.trainingBatchSize:
                        currentDumpSize = 0
                        print "Created dump:-"+language+str(dumpLength)
                        np.save("Dump\\dumpX_"+language+str(dumpLength), X)
                        np.save("Dump\\dumpY_"+language+str(dumpLength), Y)
                        dumpLength += 1
                        X = []
                        Y = []
                    currentDumpSize += featuresPerFrame
                    X.append(frameFeature)
                    Y.append(self.label.get(language))
                    inputSize += 1
                if flag == 1:
                    break
            # print "current memory usage :1", (process.memory_info().rss)/(1024*1024)
            # print "current memory usage :2", (process.memory_info().rss)/(1024*1024)
            if underFetching == True:
                print "Under Fetched Data Samples expected",self.epoch,"received",inputSize
        return noOfFilesTrained

    # def assertFeatureVector(self, X, Y):
    #     if len(X) == len(Y):
    #         print "Len Assert Pass", len(X)
    #     for frameFeature in X:
    #         if frameFeature.shape != X[0].shape:
    #             print "Dimension Assert Fail"
    #         for feature in frameFeature:
    #             if not(0.0 <= feature <= 1.0):
    #                 print "Fail"
    #     print "Dimension Assert Pass", X[0].shape

    def predict(self, audio):
        featureVector = audio.getAverageFeatureVector(std=True)
        normFeatureVector = self.normConv(featureVector, self.mu, self.sigma)
        return self.classifier.predict(normFeatureVector)

    # def normaliseFeatureVector(self, X):
    #     Xmin = np.min(X, axis=0)
    #     Xmax = np.max(X, axis=0)
    #     delta = np.subtract(X, Xmin)
    #     diff = np.subtract(Xmax, Xmin)
    #     for i, frame in enumerate(delta):
    #         for j, value in enumerate(frame):
    #             if diff[j] == 0.0:
    #                 delta[i][j] = 1.0
    #             else:
    #                 delta[i][j] = delta[i][j]/diff[j]
    #     return delta

    def normalise(self, X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0, ddof=1)
        X_norm = (X-mu)/sigma
        return X_norm, mu, sigma

    # Converts test data to a format on which NN was trained
    def normConv(self, X, mu, sigma):
        return (X-mu)/sigma

    # def plotFeature(self, language, type, fig, featNo, style, number):
    #     X = []
    #     figure(fig)
    #     if type == "Train":
    #         samples = AudioIO.getTrainingSamples(language, number)
    #         for sample in samples:
    #             featureVector = sample.getFeatureVector()
    #             for framefeature in featureVector:
    #                 for i, feature in enumerate(framefeature):
    #                     if i == featNo:
    #                         X.append(feature)
    #     elif type == "Test":
    #         samples = AudioIO.getTestSamples(language, number)
    #         for sample in samples:
    #             featureVector = sample.getFeatureVector()
    #             for framefeature in featureVector:
    #                 for i, feature in enumerate(framefeature):
    #                     if i == featNo:
    #                         X.append(feature)
    #     plot(X, color=style[0], marker=style[1], alpha=style[2])
    #
    # def plotFeature2D(self, language, type, fig, featNo, featNo2, style, number):
    #     X = []
    #     Y = []
    #     figure(fig)
    #     if type == "Train":
    #         samples = AudioIO.getTrainingSamples(language,number)
    #         for sample in samples:
    #             featureVector = sample.getFeatureVector()
    #             for framefeature in featureVector:
    #                 for i, feature in enumerate(framefeature):
    #                     if i == featNo:
    #                         X.append(feature)
    #                     if i == featNo2:
    #                         Y.append(feature)
    #     elif type == "Test":
    #         samples = AudioIO.getTestSamples(language, number)
    #         for sample in samples:
    #             featureVector = sample.getFeatureVector()
    #             for framefeature in featureVector:
    #                 for i, feature in enumerate(framefeature):
    #                     if i == featNo:
    #                         X.append(feature)
    #                     if i == featNo2:
    #                         Y.append(feature)
    #     self.normaliseFeatureVector(X)
    #     plot(X, Y, color=style[0], marker=style[1], alpha=style[2])
    #
    # def showPlot(self):
    #     show()

    def analyse(self):
        """
        :return: list of percentage success with language
        """
        analysis = []
        for language in self.languages:
            Total = AppConfig.getTestEpoch()
            success = 0
            print language
            for num in range(AppConfig.getTestEpoch()):
                progress = num*100/Total
                #sys.stdout.write(str(progress) + " ")
                try:
                    subcandidates = self.predict(Audio.Audio(AppConfig.getFilePathTest(language, num)))
                    if subcandidates[0][1] == self.label[language]:
                        success += 1
                except:
                    print "fail"
                    continue
            analysis.append((language, float(success*100)/Total))
        return analysis

# a = datetime.datetime.now()
X = scoreModel(AppConfig.languages, AppConfig.getTrainingDataSize())
# X.populateFeatureVector()
X.createAudioDumps()
files = X.dumpFeatureVector()
X.train()
# b = datetime.datetime.now()
# c = b-a
# print c.seconds
print X.analyse()
