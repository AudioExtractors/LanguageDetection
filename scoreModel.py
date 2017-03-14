import Classifier
import AudioIO
import numpy as np
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
    def __init__(self, languages,featureSets, epoch):

        self.label = dict()
        self.languages = languages
        self.featureSets = featureSets
        self.epoch = epoch
        # self.classifier = Classifier.Classifier(AppConfig.getHiddenLayer(), AppConfig.getEpoch())
        self.classifier = Classify.Classify()
        self.inputFeatureVector = []  # Feature Vector
        self.inputClassVector = []  # ClassVector
        for i,language in enumerate(languages):
            self.label[language] = i

    def populateFeatureVector(self):
        """
        :return: return list of number of files trained for each language
        """
        languagesFeatures = []
        X = []
        Y = []
        noOfFilesTrained = []
        for language in self.languages:
            print "Fetching Data for ", language
            inputSize = 0
            flag = 0
            samples = AudioIO.getTrainingSamples(language, random="False")
            print "yes"
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
                    # print type(X),type(x[0])
                    # print (len(X)*len(X[0])*64)/float(1024*1024*8), "..MB"
                    Y.append(self.label.get(language))
                    inputSize += 1
                if flag == 1:
                    break

        # print X
        print "Fetched Feature Vector.."
        X = self.normaliseFeatureVector(X)
        print "Normalised Feature Vector.."
        print "current memory usage : ", (process.memory_info().rss)/(1024*1024)
        self.assertFeatureVector(X,Y)
        #print X
        """self.classifier.train(X,Y)"""
        return noOfFilesTrained

    def train(self):

        dumpSize=AudioIO.getFeatureDumpSize()
        print "DumpSize: ",dumpSize

        for i in range(dumpSize):
            combineDumpLanguageFeature=np.array([])
            combineDumpLanguageLabel=np.array([])
            for language in self.languages:
                X=np.load("Dump//dumpX_"+language+str(i)+".npy")
                Y=np.load("Dump//dumpY_"+language+str(i)+".npy")
                if len(combineDumpLanguageFeature)==0:
                    combineDumpLanguageFeature=X
                    combineDumpLanguageLabel=Y
                else:
                    combineDumpLanguageFeature=np.vstack((combineDumpLanguageFeature,X))
                    combineDumpLanguageLabel=np.concatenate((combineDumpLanguageLabel,Y))
            print combineDumpLanguageLabel
            self.classifier.train(combineDumpLanguageFeature,combineDumpLanguageLabel)


    def dumpFeatureVector(self):
        """
        :return: return list of number of files trained for each language
        """
        languagesFeatures = []
        underFetching=True
        noOfFilesTrained = []
        for language in self.languages:
            X = []
            Y = []
            dumpLength = 0
            currentDumpSize = 0
            print "Fetching Data for ", language
            inputSize = 0
            flag = 0
            samples = AudioIO.getTrainingSamples(language, random="False")
            print "current memory usage : ", (process.memory_info().rss)/(1024*1024)
            ct = 0
            gt=0
            for sample in samples:
                print ct
                ct=ct+1
                featureVector=sample.getAverageFeatureVector(std=True)
                print featureVector.shape
                if len(featureVector)>0:
                    featuresPerFrame=len(featureVector[0])
                else:
                    continue
                for frameFeature in featureVector:
                    if inputSize>=self.epoch:
                        underFetching=False
                        print "Created dump:-"+str(dumpLength)
                        X=self.normaliseFeatureVector(X)
                        np.save("Dump\\dumpX_"+language+str(dumpLength),X)
                        np.save("Dump\\dumpY_"+language+str(dumpLength),Y)
                        print "Created All Dumps.."
                        noOfFilesTrained.append((language,sample.getIndex()))
                        flag=1
                        break
                    if currentDumpSize + featuresPerFrame > AppConfig.trainingBatchSize:
                        currentDumpSize=0
                        X=self.normaliseFeatureVector(X)
                        print "Created dump:-"+language+str(dumpLength)
                        np.save("Dump\\dumpX_"+language+str(dumpLength),X)
                        np.save("Dump\\dumpY_"+language+str(dumpLength),Y)
                        dumpLength+=1
                        X=[]
                        Y=[]
                    currentDumpSize+=featuresPerFrame
                    X.append(frameFeature)
                    Y.append(self.label.get(language))
                    inputSize += 1
                if flag == 1:
                    break
            print "current memory usage :1", (process.memory_info().rss)/(1024*1024)
            samples=[]
            print "current memory usage :2", (process.memory_info().rss)/(1024*1024)
            if underFetching==True:
                print "Under Fetched Data Samples"
        # print X
        return noOfFilesTrained

    def assertFeatureVector(self,X,Y):
        if len(X)==len(Y):
            print "len Assert Pass",len(X)
        for frameFeature in X:
            if frameFeature.shape != X[0].shape:
                print "Dimension Assert Fail"
            for feature in frameFeature:
                if not(feature >= 0.0 and feature <= 1.0):
                    print "fail"
        print "Dimension Assert Pas", X[0].shape

    def predict(self, audio):
        featureVector = audio.getAverageFeatureVector(std=True)
        #normFeatureVector = self.normaliseFeatureVector(featureVector)
        return self.classifier.predict(featureVector)
        # return self.classifier.predict(featureVector)

    def normaliseFeatureVector(self,X):
        Xmin=np.min(X,axis=0)
        Xmax=np.max(X,axis=0)
        delta=np.subtract(X,Xmin)
        diff=np.subtract(Xmax,Xmin)
        for i, frame in enumerate(delta):
            for j, value in enumerate(frame):
                if diff[j] == 0.0:
                    delta[i][j] = 1.0
                else:
                    delta[i][j] = delta[i][j]/diff[j]
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
            samples = AudioIO.getTrainingSamples(language,number)
            for sample in samples:
                featureVector = sample.getFeatureVector()
                for framefeature in featureVector:
                    for i, feature in enumerate(framefeature):
                        if i == featNo:
                            X.append(feature)
                        if i == featNo2:
                            Y.append(feature)
        elif type == "Test":
            samples = AudioIO.getTestSamples(language, number)
            for sample in samples:
                featureVector = sample.getFeatureVector()
                for framefeature in featureVector:
                    for i, feature in enumerate(framefeature):
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
                progress = num*100/Total
                sys.stdout.write(str(progress) + " ")
                subcandidates = self.predict(Audio.Audio(AppConfig.getFilePathTest(language, num)))
                if subcandidates[0][1] == self.label[language]:
                    success += 1
            analysis.append((language, float(success*100)/Total))
            print ""
        return analysis

a = datetime.datetime.now()
X = scoreModel(AppConfig.languages, ["asd", "sdf", "asd"], AppConfig.getTrainingDataSize())
#X.populateFeatureVector()
#X.dumpFeatureVector()
#print AppConfig.getNumFeatures()*AppConfig.getContextWindowSize()
X.train()
b=datetime.datetime.now()
c=b-a
print c.seconds
print X.analyse()