import AudioIO
import AppConfig
from matplotlib.pyplot import *
import datetime
import sys
import Classify
import psutil
import os
import confusion_matrix
from binary_classify import BinaryClassify
from feature_selection import FeatureSelection
from feature_normalise import FeatureNormalise
process = psutil.Process(os.getpid())


class scoreModel:
    def __init__(self, languages, epoch):
        self.languages = languages
        self.epoch = epoch
        self.classifier = Classify.Classify()
        self.bClassifiers = {}
        for i in xrange(len(languages) - 1):
            for j in xrange(i + 1, len(languages)):
                self.bClassifiers[(languages[i], languages[j])] = BinaryClassify(labels=[i, j])
        self.inputFeatureVector = []  # Feature Vector
        self.inputClassVector = []  # ClassVector
        self.sel = FeatureSelection(AppConfig.getNumLanguages(), AppConfig.getNumFeatures() *
                                    AppConfig.getNumberOfAverageStats() * AppConfig.getContextWindowSize(),
                                    k=AppConfig.selFeatures)
        self.norm = FeatureNormalise(AppConfig.getNumFeatures() * AppConfig.getNumberOfAverageStats() *
                      AppConfig.getContextWindowSize())

    def createAudioDumps(self):
        for language in self.languages:
            print "Dumping Training Audio Samples of", language
            AudioIO.dumpTrainFiles(language)
        for language in self.languages:
            print "Dumping Testing Audio Samples of", language
            AudioIO.dumpTestFiles(language)

    def dumpFeatureVector(self):
        underFetching = True
        # noOfFilesTrained = []
        for language in self.languages:
            X = []
            Y = []
            dumpLength = 0
            currentDumpSize = 0
            print "Fetching Data for", language
            inputSize = 0
            flag = 0
            samples = AudioIO.getDumpTrainingSample(language)
            ct = 0
            for sample in samples:
                ct += 1
                featureVector = sample.getAverageFeatureVector(std=AppConfig.includeStd)
                if len(featureVector) > 0:
                    featuresPerFrame = len(featureVector[0])
                else:
                    continue
                for frameFeature in featureVector:
                    if inputSize >= self.epoch:
                        underFetching = False
                        print "Created dump:-"+str(dumpLength)
                        np.save("Dump\\dumpX_" + language + str(dumpLength), X)
                        np.save("Dump\\dumpY_" + language + str(dumpLength), Y)
                        # noOfFilesTrained.append((language, sample.getIndex()))
                        flag = 1
                        break
                    if currentDumpSize + featuresPerFrame > AppConfig.trainingBatchSize:
                        currentDumpSize = 0
                        print "Created dump:-"+language + str(dumpLength)
                        np.save("Dump\\dumpX_"+language + str(dumpLength), X)
                        np.save("Dump\\dumpY_"+language + str(dumpLength), Y)
                        dumpLength += 1
                        X = []
                        Y = []
                    currentDumpSize += featuresPerFrame
                    X.append(frameFeature)
                    Y.append(self.languages.index(language))
                    inputSize += 1
                if flag == 1:
                    break
            if underFetching == True:
                print "Under Fetched Data Samples expected", self.epoch, "received", inputSize
            else:
                print "Fetched Data Samples expected", inputSize, "and ", ct, "number of files have been dumped"
        # return noOfFilesTrained

    def normFeature(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        for i in range(dumpSize):
            for language in self.languages:
                X = np.load("Dump//dumpX_" + language + str(i) + ".npy")
                self.norm.batchData(X)
        self.norm.fit()

    def selectFeature(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        for i in range(dumpSize):
            for language in self.languages:
                X = np.load("Dump//dumpX_" + language + str(i) + ".npy")
                y = np.load("Dump//dumpY_" + language + str(i) + ".npy")
                self.sel.batchData(self.norm.transform(X), y)
        self.sel.fit()

    def train(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        for i in range(dumpSize):
            combined_feature = np.array([])
            combined_label = np.array([])
            for language in self.languages:
                X = np.load("Dump//dumpX_" + language + str(i) + ".npy")
                y = np.load("Dump//dumpY_" + language + str(i) + ".npy")
                X = self.norm.transform(X)
                if len(combined_feature) == 0:
                    combined_feature = X
                    combined_label = y
                else:
                    combined_feature = np.vstack((combined_feature, X))
                    combined_label = np.concatenate((combined_label, y))
            X = self.sel.transform(combined_feature)
            self.classifier.train(X, combined_label)

    def binaryTrain(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        combined_language_feature = {}
        combined_language_label = {}
        languages = self.languages
        for language in languages:
            combined_language_feature[language] = np.array([])
            combined_language_label[language] = np.array([])
            for i in range(dumpSize):
                X = np.load("Dump//dumpX_" + language + str(i) + ".npy")
                y = np.load("Dump//dumpY_" + language + str(i) + ".npy")
                if len(combined_language_feature[language]) == 0:
                    combined_language_feature[language] = X
                    combined_language_label[language] = y
                else:
                    combined_language_feature[language] = np.vstack((combined_language_feature[language], X))
                    combined_language_label[language] = np.concatenate((combined_language_label[language], y))
            combined_language_feature[language] = self.norm.transform(combined_language_feature[language])
        masks = np.load("Dump\\confusion_matrix.npy").item()
        for i in xrange(len(languages) - 1):
            for j in xrange(i + 1, len(languages)):
                X_combined = np.vstack((combined_language_feature[languages[i]],
                                        combined_language_feature[languages[j]]))
                y_combined = np.concatenate((combined_language_label[languages[i]],
                                             combined_language_label[languages[j]]))
                print "Training for: ", (languages[i], languages[j])
                X_combined = X_combined[:, masks[(languages[i], languages[j])]]
                self.bClassifiers[(languages[i], languages[j])].train(X_combined, y_combined)

    def predict(self, audio):
        featureVector = audio.getAverageFeatureVector(std=AppConfig.includeStd)
        normFeatureVector = self.norm.transform(featureVector)
        subcandidates = self.classifier.predict(self.sel.transform(normFeatureVector))
        language1 = self.languages[subcandidates[0][1]]
        language2 = self.languages[subcandidates[1][1]]
        if subcandidates[0][1] > subcandidates[1][1]:
            language1, language2 = language2, language1
        masks = np.load("Dump\\confusion_matrix.npy").item()
        finalcandidates = self.bClassifiers[(language1, language2)].predict(normFeatureVector[:, masks[(language1,
                                                                                                       language2)]])
        label_sum = subcandidates[0][1] + subcandidates[1][1]
        total_label = len(self.languages)
        for i in xrange(2, total_label):
            finalcandidates = finalcandidates.append(subcandidates[i])
            label_sum += subcandidates[i][1]
        if label_sum != (total_label*(total_label-1))/2:
            raise ValueError("Unexpected Output Candidates", finalcandidates)
        return finalcandidates

    def analyse(self):
        fileName = os.path.join(AppConfig.logs_base_dir, "Log" + str(datetime.datetime.now().strftime("%m%d-%H-%M-%S")))
        log = open(fileName+".txt", 'w')
        confusionMatrix = {}
        for language in self.languages:
            for language2 in self.languages:
                confusionMatrix[(language, language2)] = 0
        totalCount = {}
        for language in self.languages:
            totalCount[language] = 0

        analysis = []
        for language in self.languages:
            log.write("Testing for samples in " + language + "\n")
            Total = AppConfig.getTestEpoch()
            success = 0
            completefailure = 0
            total = 0
            print language
            samples = AudioIO.getDumpTestSample(language)
            for sample in samples:
                total += 1
                subcandidates = self.predict(sample)
                key = (language, self.languages[subcandidates[0][1]])
                confusionMatrix[key] += 1
                if self.languages[subcandidates[0][1]] == language:
                    success += 1
                    log.write("[Correct]"+sample.getIndex() + " " + str(subcandidates) + "\n")
                else:
                    if self.languages[subcandidates[2][1]] == language:
                        completefailure += 1
                    log.write("[Wrong]"+sample.getIndex() + " " + str(subcandidates) + "\n")
            totalCount[language] = total
            analysis.append((language, float(success*100)/Total))
            analysis.append((language, 100.0 - float(completefailure * 100) / Total))
        sys.stdout.write("     ")

        for language in self.languages:
            print language+"   ",
        for language in self.languages:
            sys.stdout.write("\n"+language+"   ")
            for language2 in self.languages:
                print "%d" % confusionMatrix[(language, language2)]+"   ",
        print "\n"

        log.write("Confusion Matrix on number of Samples:")
        log.write(str(confusionMatrix) + "\n" + "\n")
        for language in self.languages:
                for language2 in self.languages:
                    confusionMatrix[(language, language2)] /= float(totalCount[language])
        log.write("Confusion Matrix on Percentage of Samples:")
        log.write(str(confusionMatrix))
        print "See logs in -", fileName
        return analysis

    def saveNN(self, name):
        self.classifier.save(os.path.join(AppConfig.NN_save_dir, name))

    def loadNN(self, name):
        self.classifier.load(os.path.join(AppConfig.NN_save_dir, name))

    def saveBinaryNN(self, name):
        for i in xrange(len(self.languages) - 1):
            for j in xrange(i + 1, len(self.languages)):
                self.bClassifiers[(self.languages[i], self.languages[j])].save(os.path.join(AppConfig.NN_save_dir, name
                                                                                            + str(i) + str(j)))

    def loadBinaryNN(self, name):
        for i in xrange(len(self.languages) - 1):
            for j in xrange(i + 1, len(self.languages)):
                self.bClassifiers[(self.languages[i], self.languages[j])].load(os.path.join(AppConfig.NN_save_dir, name
                                                                                            + str(i) + str(j)))

if __name__ == "__main__":
    X = scoreModel(AppConfig.languages, AppConfig.getTrainingDataSize())
    # X.createAudioDumps()
    # X.dumpFeatureVector()
    # confusion_matrix.dumpConfusionMatrix()
    X.normFeature()
    X.selectFeature()
    X.train()
    # X.saveNN("NN")
    # X.loadNN("NN")
    X.binaryTrain()
    print X.analyse()
