import AudioIO
import AppConfig
from matplotlib.pyplot import *
import datetime
import sys
import Classify
from sklearn.linear_model import SGDClassifier
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
        if AppConfig.includeBaseline:
            self.baselineClassfier = SGDClassifier(warm_start=True)
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
            print "Done"
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
            if AppConfig.includeBaseline:
                self.baselineClassfier.fit(X, combined_label)
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
        if AppConfig.includeBaseline:
            subcandidatesbaseline = self.baselineClassfier.predict(self.sel.transform(normFeatureVector))
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
            finalcandidates.append(subcandidates[i])
            label_sum += subcandidates[i][1]
        if label_sum != (total_label*(total_label-1))/2:
            raise ValueError("Unexpected Output Candidates", finalcandidates)
        if AppConfig.includeBaseline:
            return (subcandidatesbaseline, finalcandidates)
        else:
            return finalcandidates

    def analyse(self):
        ##LOGGING
        fileName = os.path.join(AppConfig.logs_base_dir, "Log" + str(datetime.datetime.now().strftime("%m%d-%H-%M-%S")))
        log = open(fileName+".txt", 'w')
        log.write("Parameters:\n")
        log.write("windowSize             :" + str(AppConfig.windowSize) + "\n")
        log.write("windowHop              :" + str(AppConfig.windowHop) + "\n")
        log.write("contextWindowSize      :" + str(AppConfig.contextWindowSize) + "\n")
        log.write("averageFramesPerSample :" + str(AppConfig.averageFramesPerSample) + "\n")
        log.write("includeStd             :" + str(AppConfig.includeStd) + "\n")
        log.write("hiddenLayer            :" + str(AppConfig.hiddenLayer) + "\n")
        log.write("batch_size             :" + str(AppConfig.batch_size) + "\n")
        log.write("nb_epoch               :" + str(AppConfig.nb_epoch) + "\n")
        log.write("selFeatures            :" + str(AppConfig.selFeatures) + "\n")
        log.write("binaryHiddenLayer      :" + str(AppConfig.binaryHiddenLayer) + "\n")
        log.write("binary_batch_size      :" + str(AppConfig.binary_batch_size) + "\n")
        log.write("binary_nb_epoch        :" + str(AppConfig.binary_nb_epoch) + "\n")
        log.write("selBinaryFeatures      :" + str(AppConfig.selBinaryFeatures) + "\n")
        ##LOGGING END

        confusionMatrix = {}
        for language in self.languages:
            for language2 in self.languages:
                confusionMatrix[(language, language2)] = 0
        if AppConfig.includeBaseline:
            confusionMatrixBaseline = {}
            for language in self.languages:
                for language2 in self.languages:
                    confusionMatrixBaseline[(language, language2)] = 0
        totalCount = {}
        for language in self.languages:
            totalCount[language] = 0

        analysis = []
        for language in self.languages:
            log.write("\nTesting for samples in " + language + "\n")
            Total = AppConfig.getTestEpoch()
            success = 0
            completefailure = 0
            total = 0
            print language
            samples = AudioIO.getDumpTestSample(language)
            for sample in samples:
                total += 1
                if AppConfig.includeBaseline:
                    subcandidatesbaseline, subcandidates = self.predict(sample)
                else:
                    subcandidates = self.predict(sample)
                if AppConfig.includeBaseline:
                    key = (language, self.languages[subcandidatesbaseline[0]])
                    confusionMatrixBaseline[key] += 1
                key = (language, self.languages[subcandidates[0][1]])
                confusionMatrix[key] += 1
                if self.languages[subcandidates[0][1]] == language:
                    success += 1
                    log.write("[Correct] "+sample.getIndex() + " " + str(subcandidates) + "\n")
                else:
                    if self.languages[subcandidates[0][1]] != language and self.languages[subcandidates[1][1]] != language:
                        completefailure += 1
                    log.write("[Wrong]   "+sample.getIndex() + " " + str(subcandidates) + "\n")
            totalCount[language] = total
            analysis.append((language, float(success*100)/Total))
            analysis.append((language, 100.0 - float(completefailure * 100) / Total))

        confusionMatrixTemp = []
        if AppConfig.includeBaseline:
            confusionMatrixTempBaseline = []
        print "\nConfusion Matrix:"
        sys.stdout.write("      ")
        for language in self.languages:
            print language + "    ",
        for language in self.languages:
            sys.stdout.write("\n" + language + "   ")
            if AppConfig.includeBaseline:
                confusionMatrixTempRowBaseline = []
            confusionMatrixTempRow = []
            for language2 in self.languages:
                print "%3d" % confusionMatrix[(language, language2)] + "   ",
                if AppConfig.includeBaseline:
                    confusionMatrixTempRowBaseline.append(confusionMatrixBaseline[(language, language2)])
                confusionMatrixTempRow.append(confusionMatrix[(language, language2)])
            if AppConfig.includeBaseline:
                confusionMatrixTempBaseline.append(confusionMatrixTempRowBaseline)
            confusionMatrixTemp.append(confusionMatrixTempRow)
        print "\n"

        print "Analysis:"
        confusionMatrixTemp = np.array(confusionMatrixTemp)
        FP = confusionMatrixTemp.sum(axis=0) - np.diag(confusionMatrixTemp)
        FN = confusionMatrixTemp.sum(axis=1) - np.diag(confusionMatrixTemp)
        TP = np.diag(confusionMatrixTemp)
        TN = confusionMatrixTemp.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        ACC = (TP+TN)/(TP+FP+FN+TN)
        PRE = TP / (TP + FP)
        REC = TP / (TP + FN)
        if AppConfig.includeBaseline:
            confusionMatrixTempBaseline = np.array(confusionMatrixTempBaseline)
            FPBaseline = confusionMatrixTempBaseline.sum(axis=0) - np.diag(confusionMatrixTempBaseline)
            FNBaseline = confusionMatrixTempBaseline.sum(axis=1) - np.diag(confusionMatrixTempBaseline)
            TPBaseline = np.diag(confusionMatrixTempBaseline)
            TNBaseline = confusionMatrixTempBaseline.sum() - (FPBaseline + FNBaseline + TPBaseline)
            FPBaseline = FPBaseline.astype(float)
            FNBaseline = FNBaseline.astype(float)
            TPBaseline = TPBaseline.astype(float)
            TNBaseline = TNBaseline.astype(float)
            ACCBaseline = (TPBaseline + TNBaseline) / (TPBaseline + FPBaseline + FNBaseline + TNBaseline)
            PREBaseline = TPBaseline / (TPBaseline + FPBaseline)
            RECBaseline = TPBaseline / (TPBaseline + FNBaseline)
            print "           Baseline SGD Classifier Model    Neural Network Hybrid Model"
            print "Accuracy : %.2f" % (np.mean(ACCBaseline) * 100) + "%" + "                           %.2f" % (np.mean(ACC) * 100) + "%"
            print "Precision: %.2f" % (np.mean(PREBaseline) * 100) + "%" + "                           %.2f" % (np.mean(PRE) * 100) + "%"
            print "Recall   : %.2f" % (np.mean(RECBaseline) * 100) + "%" + "                           %.2f" % (np.mean(REC) * 100) + "%"
            print ""
        else:
            print "Neural Network Hybrid Model"
            print "Accuracy : %.2f" % (np.mean(ACC) * 100) + "%"
            print "Precision: %.2f" % (np.mean(PRE) * 100) + "%"
            print "Recall   : %.2f" % (np.mean(REC) * 100) + "%"
            print ""

        ##LOGGING
        log.write("\nConfusion Matrix on number of Samples:\n")
        log.write(str(confusionMatrix) + "\n" + "\n")
        for language in self.languages:
                for language2 in self.languages:
                    confusionMatrix[(language, language2)] /= float(totalCount[language])
        log.write("Confusion Matrix on Percentage of Samples:\n")
        log.write(str(confusionMatrix) + "\n" + "\n")
        log.write("Final Analysis" + "\n")
        log.write(str(analysis))
        print "See logs in -", fileName
        ##LOGGINGEND

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
    X.loadNN("NN")
    # X.binaryTrain()
    # X.saveBinaryNN("Binary")
    X.loadBinaryNN("Binary")
    analysis = X.analyse()
