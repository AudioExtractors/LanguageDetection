import Classifier
import AudioIO
import numpy as np
import Audio
import AppConfig
from matplotlib.pyplot import *
class scoreModel:
    def __init__(self,languages,featureSets,epoch):
        self.label=dict()
        self.languages=languages
        self.featureSets=featureSets
        self.epoch=epoch
        self.classifier=Classifier.Classifier(AppConfig.getHiddenLayer(),AppConfig.getEpoch())
        for i,language in enumerate(languages):
            self.label[language]=i
    def train(self):
        """
        :return: return list of number of files trained for each language
        """
        ENG=[]
        ENGN=[]
        X=[]
        Y=[]
        noOfFilesTrained=[]
        for language in self.languages:
            inputSize=0
            flag=0
            samples=AudioIO.getTrainingSamples(language)
            for sample in samples:
                featureVector=sample.getFeatureVector2()
                for frameFeature in featureVector:
                    if(inputSize>self.epoch):
                        noOfFilesTrained.append((language,sample.getIndex()))
                        flag=1
                        break
                    X.append(frameFeature)
                    Y.append(self.label.get(language))
                    inputSize+=1
                if flag==1:
                    break
        X=np.array(X)
        Y=np.array(Y)
        X=self.normaliseFeatureVector(X)
        self.classifier.train(X,Y)
        return noOfFilesTrained
    def predict(self,audio):
        featureVector=audio.getFeatureVector2()
        #return self.classifier.predict(self.normaliseFeatureVector(featureVector))
        return self.classifier.predict(featureVector)
    def normaliseFeatureVector(self,X):
        Xmin=np.min(X,axis=0)
        Xmax=np.max(X,axis=0)
        delta=np.subtract(X,Xmin)
        diff=np.subtract(Xmax,Xmin)

        for i,frame in enumerate(delta):
            for j,value in enumerate(frame):
                if diff[j] == 0.0:
                    delta[i][j]=1.0
                else:

                    delta[i][j]=delta[i][j]/diff[j]
        print "delta",delta
        print "diff",diff
        return delta
    def plotFeature(self,language,type,fig,featNo,color,number):
        X=[]
        figure(fig)
        if type=="Train":
            samples=AudioIO.getTrainingSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector2()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
        elif type=="Test":
            samples=AudioIO.getTestSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector2()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
        print X
        plot(X,color)
    def plotFeature2D(self,language,type,fig,featNo,featNo2,color,number):
        X=[]
        Y=[]
        figure(fig)
        if type=="Train":
            samples=AudioIO.getTrainingSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector2()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
                        if i==featNo2:
                            Y.append(feature)
        elif type=="Test":
            samples=AudioIO.getTestSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector2()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
                        if i==featNo2:
                            Y.append(feature)

        plot(X,Y,color)
    def showPlot(self):
        show()
    def analyse(self):
        """

        :return: list of percentage success with language
        """
        analysis=[]
        for language in self.languages:
            Total=AppConfig.getTestEpoch()
            success=0
            for num in range(AppConfig.getTestEpoch()):
                subcandidates=self.predict(Audio.Audio(AppConfig.getFilePathTest(language,num)))
                if subcandidates[0][1]==self.label[language]:
                    success+=1
            analysis.append((language,float(success*100/Total)))
        return analysis


X=scoreModel(["en","it"],["asd","sdf","asd"],AppConfig.getEpoch())
"""Y=X.train()
print Y"""
X.plotFeature2D("en","Train",1,1,10,"rd",100)
X.plotFeature2D("it","Train",1,1,10,"gd",100)
X.plotFeature2D("it","Test",1,1,10,"bd",100)
X.showPlot()

