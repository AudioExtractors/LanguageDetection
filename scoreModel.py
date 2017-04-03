import AudioIO
import AppConfig
from matplotlib.pyplot import *
import datetime
import sys
import Classify
import psutil
import os
from feature_selection import FeatureSelection
from feature_normalise import FeatureNormalise
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
        #TODO: replace 78 with number of features and put 20 in appconfig as number of feture selected
        self.sel = FeatureSelection(AppConfig.getNumLanguages(), 78, 20)
        self.norm = FeatureNormalise(78)

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
            # X_norm = self.norm.transform(combineDumpLanguageFeature) this will normalise data
            # X = self.sel.transform(X_norm) this will eliminate some coloumns
            # self.classifier.train(X, combineDumpLanguageLabel)

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
                print "Under Fetched Data Samples expected", self.epoch, "received", inputSize
            else:
                print "Fetched Data Samples expected", inputSize
        return noOfFilesTrained


    def predict(self, audio):
        featureVector = audio.getAverageFeatureVector(std=True)
        normFeatureVector = self.norm.transform(featureVector)
        return self.classifier.predict(normFeatureVector)

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
            total=0
            print language
            samples = AudioIO.getDumpTestSample(language)
            for sample in samples:
                try:
                    total += 1
                    subcandidates = self.predict(sample)
                    key=(language,self.languages[subcandidates[0][1]])
                    confusionMatrix[key] += 1
                    if subcandidates[0][1] == self.label[language]:
                        success += 1
                        log.write("[Correct]"+sample.getIndex()+" "+str(subcandidates)+"\n")
                    else:
                        log.write("[Wrong]"+sample.getIndex()+" "+str(subcandidates)+"\n")
                except:
                    print "fail"
                    continue
            totalCount[language] = total
            analysis.append((language, float(success*100)/Total))

        print confusionMatrix
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


        return analysis

# a = datetime.datetime.now()
X = scoreModel(AppConfig.languages, AppConfig.getTrainingDataSize())
# X.populateFeatureVector()
# X.createAudioDumps()
#files = X.dumpFeatureVector()
# print "Files",files
# print AppConfig.getNumFeatures()*AppConfig.
X.normFeature()
X.selectFeature()
X.train()
#X.selectFeature()
# b = datetime.datetime.now()
# c = b-a
# print c.seconds
#print X.analyse()
