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
        # self.featureSets = featureSets
        self.epoch = epoch
        # self.classifier = Classifier.Classifier(AppConfig.getHiddenLayer(), AppConfig.getEpoch())
        self.classifier = Classify.Classify()
        self.bClassifiers = {}
        for i in xrange(len(languages) - 1):
            for j in xrange(i + 1, len(languages)):
                self.bClassifiers[(languages[i], languages[j])] = BinaryClassify(labels=[i,j])
        self.inputFeatureVector = []  # Feature Vector
        self.inputClassVector = []  # ClassVector
        #TODO: replace 78 with number of features and put 20 in appconfig as number of feture selected
        self.sel = FeatureSelection(AppConfig.getNumLanguages(), AppConfig.getNumFeatures() * AppConfig.getNumberOfAverageStats() *
                      AppConfig.getContextWindowSize(), k=AppConfig.selFeatures)
        self.norm = FeatureNormalise(AppConfig.getNumFeatures() * AppConfig.getNumberOfAverageStats() *
                      AppConfig.getContextWindowSize())

    def normFeature(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        for i in range(dumpSize):
            for language in self.languages:
                X = np.load("Dump//dumpX_"+language+str(i)+".npy")
                self.norm.batchData(X)
        self.norm.fit()

    def selectFeature(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        for i in range(dumpSize):
            for language in self.languages:
                X = np.load("Dump//dumpX_"+language+str(i)+".npy")
                y = np.load("Dump//dumpY_"+language+str(i)+".npy")
                self.sel.batchData(self.norm.transform(X), y)
        self.sel.fit()
        #print self.sel.mask

    def train(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        # print "DumpSize: ", dumpSize
        for i in range(dumpSize):
            combined_feature = np.array([])
            combined_label = np.array([])
            for language in self.languages:
                X = np.load("Dump//dumpX_"+language+str(i)+".npy")
                y = np.load("Dump//dumpY_"+language+str(i)+".npy")
                X = self.norm.transform(X)
                if len(combined_feature) == 0:
                    combined_feature = X
                    combined_label = y
                else:
                    combined_feature = np.vstack((combined_feature, X))
                    combined_label = np.concatenate((combined_label, y))
            # print combineDumpLanguageLabel
            # print i, len(combineDumpLanguageFeature)
            #X_norm = self.norm.transform(combined_feature)
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
                X = np.load("Dump//dumpX_"+language+str(i)+".npy")
                y = np.load("Dump//dumpY_"+language+str(i)+".npy")
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
                X_combined = np.vstack((combined_language_feature[languages[i]], combined_language_feature[languages[j]]))
                y_combined = np.concatenate((combined_language_label[languages[i]], combined_language_label[languages[j]]))
                print "Training for: ",(languages[i], languages[j])
                X_combined = X_combined[:, masks[(languages[i], languages[j])]]
                self.bClassifiers[(languages[i], languages[j])].train(X_combined,y_combined)

    def createAudioDumps(self):
        for language in self.languages:
            print "Dumping Audio Samples of", language
            AudioIO.dumpAudioFiles(language)

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
                    Y.append(self.languages.index(language))
                    inputSize += 1
                if flag == 1:
                    break
            # print "current memory usage :1", (process.memory_info().rss)/(1024*1024)
            # print "current memory usage :2", (process.memory_info().rss)/(1024*1024)
            if underFetching == True:
                print "Under Fetched Data Samples expected", self.epoch, "received", inputSize
            else:
                print "Fetched Data Samples expected", inputSize,"and ",ct,"number of files have been dumped"
        return noOfFilesTrained


    def predict(self, audio):
        featureVector = audio.getAverageFeatureVector(std=True)
        normFeatureVector = self.norm.transform(featureVector)
        subcandidates = self.classifier.predict(self.sel.transform(normFeatureVector))
        language1 = self.languages[subcandidates[0][1]]
        language2 = self.languages[subcandidates[1][1]]
        if subcandidates[0][1] > subcandidates[1][1]:
            language1, language2 = language2, language1
        masks = np.load("Dump\\confusion_matrix.npy").item()
        finalcandidates = self.bClassifiers[(language1,language2)].predict(normFeatureVector[:, masks[(language1, language2)]])
        finalcandidates.append(subcandidates[2])
        if finalcandidates[0][1]+finalcandidates[1][1]+finalcandidates[2][1] != 3:
            raise ValueError("Unexpected Output Candidates", finalcandidates)
        return finalcandidates

    def analyse(self):
        """
        :return: list of percentage success with language
        """
        fileName=os.path.join(AppConfig.logs_base_dir,"Log"+str(datetime.datetime.now().strftime("%m%d-%H-%M-%S")))
        log=open(fileName+".txt",'w')
        confusionMatrix={}
        for language in self.languages:
            for language2 in self.languages:
                confusionMatrix[(language,language2)]=0

        totalCount={}
        for language in self.languages:
            totalCount[language] = 0

        analysis = []
        for language in self.languages:
            log.write("Testing for samples in "+language+"\n")
            Total = AppConfig.getTestEpoch()
            success = 0
            completefailure = 0
            total=0
            print language
            samples = AudioIO.getDumpTestSample(language)
            for sample in samples:
 #               try:
                total += 1
                subcandidates = self.predict(sample)
                key=(language,self.languages[subcandidates[0][1]])
                confusionMatrix[key] += 1
                if self.languages[subcandidates[0][1]] == language:
                    success += 1
                    log.write("[Correct]"+sample.getIndex()+" "+str(subcandidates)+"\n")
                else:
                    if self.languages[subcandidates[2][1]] == language:
                        completefailure += 1
                    log.write("[Wrong]"+sample.getIndex()+" "+str(subcandidates)+"\n")
#                except:
#                    print "fail"
#                    continue
            totalCount[language] = total
            analysis.append((language, float(success*100)/Total))
            analysis.append((language, 100.0 - float(completefailure * 100) / Total))

        #print confusionMatrix
        sys.stdout.write("     ")
        for language in self.languages:
            print language+"   ",
        for language in self.languages:

            sys.stdout.write("\n"+language+"   ")
            for language2 in self.languages:
                print "%d" % confusionMatrix[(language,language2)]+"   ",
        print "\n"

        log.write("Confusion Matrix on number of Samples:")
        log.write(str(confusionMatrix)+"\n"+"\n")
        for language in self.languages:
                for language2 in self.languages:
                    confusionMatrix[(language,language2)] /=float(totalCount[language])
        log.write("Confusion Matrix on Percentage of Samples:")
        log.write(str(confusionMatrix))
        print "See logs in -",fileName
        return analysis

    def saveNN(self, name):
        self.classifier.save(name)

    def loadNN(self, name):
        self.classifier.load(name)

# a = datetime.datetime.now()
X = scoreModel(AppConfig.languages, AppConfig.getTrainingDataSize())
# X.populateFeatureVector()
# X.createAudioDumps()
#files = X.dumpFeatureVector()
# print "Files",files
# print AppConfig.getNumFeatures()*AppConfig.
confusion_matrix.dumpConfusionMatrix()
X.normFeature()
X.selectFeature()
X.train()
X.saveNN("NN")
# X.loadNN("NN")
X.binaryTrain()
# X.selectFeature()
# b = datetime.datetime.now()
# c = b-a
# print c.seconds
print X.analyse()
