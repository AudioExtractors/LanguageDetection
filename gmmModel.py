import AudioIO
import Audio
import AppConfig
from matplotlib.pyplot import *
import datetime
import sys
import Classify
import psutil
import os
from sklearn.mixture import GaussianMixture
process = psutil.Process(os.getpid())

class languageMixtureModel:
    def __init__(self):
        self.label = dict()
        self.languages = AppConfig.languages
        self.epoch = AppConfig.getTrainingDataSize()
        for i, language in enumerate(self.languages):
            self.label[language] = i
        self.mixtureModel=GaussianMixture(n_components=AppConfig.gmmComponents,covariance_type='full',verbose=1)
        self.mu = 0
        self.sigma = 0
        self.mp={}



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
            ct = 0
            gt = 0
            for sample in samples:
                ct += 1
                tempWindowHop=AppConfig.windowHop
                tempWindowSize=AppConfig.windowSize
                AppConfig.windowHop=AppConfig.gmmWindowHop
                AppConfig.windowSize=AppConfig.gmmWindowSize
                featureVector = sample.getAverageFeatureVector(std=False)
                AppConfig.gmmWindowHop=tempWindowHop
                AppConfig.gmmWindowSize=tempWindowSize
                if len(featureVector) > 0:
                    featuresPerFrame = len(featureVector[0])
                else:
                    continue
                for frameFeature in featureVector:
                    if inputSize >= self.epoch:
                        print "Received Total",inputSize,"Samples"
                        underFetching = False
                        print "Created dump:-"+str(dumpLength)
                        np.save("gmm\\Dump\\dumpX_"+language+str(dumpLength), X)
                        np.save("gmm\\Dump\\dumpY_"+language+str(dumpLength), Y)
                        # print "Created All Dumps.."
                        noOfFilesTrained.append((language, sample.getIndex()))
                        flag = 1
                        break
                    if currentDumpSize + featuresPerFrame > AppConfig.trainingBatchSize:
                        currentDumpSize = 0
                        print "Created dump:-"+language+str(dumpLength)
                        np.save("gmm\\Dump\\dumpX_"+language+str(dumpLength), X)
                        np.save("gmm\\Dump\\dumpY_"+language+str(dumpLength), Y)
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

    def train(self):
        dumpSize = AudioIO.getFeatureDumpSize()
        for i in range(dumpSize):
            combineDumpLanguageFeature = np.array([])
            combineDumpLanguageLabel = np.array([])
            for language in self.languages:
                X = np.load("gmm//Dump//dumpX_"+language+str(i)+".npy")
                Y = np.load("gmm//Dump//dumpY_"+language+str(i)+".npy")
                if len(combineDumpLanguageFeature) == 0:
                    combineDumpLanguageFeature = X
                    combineDumpLanguageLabel = Y
                else:
                    combineDumpLanguageFeature = np.vstack((combineDumpLanguageFeature, X))
                    combineDumpLanguageLabel = np.concatenate((combineDumpLanguageLabel, Y))
            X, self.mu, self.sigma = self.normalise(combineDumpLanguageFeature)
            Y=combineDumpLanguageLabel
            self.mixtureModel.fit(X)
            Ydash=self.mixtureModel.predict(X)
            count=[0 for i in range(0,AppConfig.gmmComponents+1)]
            for i in range(1,len(Ydash)):
                #if (Y[i]==Y[i-1]):
                key=(Y[i],Ydash[i])
                count[Ydash[i]]+=1
                if key in self.mp:
                    self.mp[key]+=1
                else:
                    self.mp[key]=1
            for k in self.mp:
                self.mp[k]=float(self.mp[k])/float(count[k[1]])
            print self.mp

    def predict(self, audio):
        featureVector = audio.getAverageFeatureVector(std=False)
        normFeatureVector = self.normConv(featureVector, self.mu, self.sigma)
        Y = self.mixtureModel.predict(normFeatureVector)
        subCandidates=[]
        for i in range(1,len(Y)):
            for language in self.languages:
                L=self.label.get(language)
                key = (L, Y[i])
                if key in self.mp:
                    subCandidates.append((self.mp[key],L))
                else:
                    subCandidates.append((0.00,L))
            subCandidates.sort()
            subCandidates.reverse()
        return subCandidates

    def normalise(self, X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0, ddof=1)
        X_norm = (X-mu)/sigma
        return X_norm, mu, sigma

    # Converts test data to a format on which NN was trained
    def normConv(self, X, mu, sigma):
        return (X-mu)/sigma

    def analyse(self):
        """
        :return: list of percentage success with language
        """
        analysis = []
        for language in self.languages:
            Total = AppConfig.getTestEpoch()
            success = 0
            print language
            samples=AudioIO.getDumpTestSample(language)
            for sample in samples:
                try:
                    subcandidates = self.predict(sample)
                    if subcandidates[0][1] == self.label[language]:
                        success += 1
                except:
                    print "fail"
                    continue
            analysis.append((language, float(success*100)/Total))
        return analysis

X = languageMixtureModel()
#F=X.dumpFeatureVector()
#print F
X.train()
A=X.analyse()
print A

