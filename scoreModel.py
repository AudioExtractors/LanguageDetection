import Classifier
import AudioIO
import numpy as np
import Audio
import AppConfig
from matplotlib.pyplot import *
import datetime
class scoreModel:
    def __init__(self,languages,featureSets,epoch):


        self.label=dict()
        self.languages=languages
        self.featureSets=featureSets
        self.epoch=epoch
        self.classifier=Classifier.Classifier(AppConfig.getHiddenLayer(),AppConfig.getEpoch())
        for i,language in enumerate(languages):
            self.label[language]=i



    def populateFeatureVector(self):
        """
        :return: return list of number of files trained for each language
        """
        self.inputFeatureVector=[]#Feature Vector
        self.inputClassVector=[]#ClassVector
        languagesFeatures=[]
        X=[]
        Y=[]
        noOfFilesTrained=[]
        for language in self.languages:
            inputSize=0
            flag=0
            samples=AudioIO.getTrainingSamples(language)
            ct=0
            for sample in samples:
                featureVector=sample.getContextFeatureVector()
                for frameFeature in featureVector:
                    if(inputSize>=self.epoch):
                        noOfFilesTrained.append((language,sample.getIndex()))
                        #languagesFeatures.append(X)
                        flag=1
                        break
                    X.append(frameFeature)
                    Y.append(self.label.get(language))
                    inputSize+=1
                if flag==1:
                    break

        #print X
        X=self.normaliseFeatureVector(X)
        #print X
        self.classifier.train(X,Y)
        return noOfFilesTrained
    def train(self):
        self.classifier.train(self.inputFeatureVector,self.inputClassVector)


    def predict(self,audio):
        featureVector=audio.getContextFeatureVector()
        normFeatureVector=self.normaliseFeatureVector(featureVector)
        return self.classifier.predict(normFeatureVector)
        #return self.classifier.predict(featureVector)


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
        return delta





    def plotFeature(self,language,type,fig,featNo,style,number):
        X=[]
        figure(fig)
        if type=="Train":
            samples=AudioIO.getTrainingSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
        elif type=="Test":
            samples=AudioIO.getTestSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
        plot(X,color=style[0],marker=style[1],alpha=style[2])


    def plotFeature2D(self,language,type,fig,featNo,featNo2,style,number):
        X=[]
        Y=[]
        figure(fig)
        if type=="Train":
            samples=AudioIO.getTrainingSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
                        if i==featNo2:
                            Y.append(feature)
        elif type=="Test":
            samples=AudioIO.getTestSamples(language,number)
            for sample in samples:
                featureVector=sample.getFeatureVector()
                for framefeature in featureVector:
                    for i,feature in enumerate(framefeature):
                        if i==featNo:
                            X.append(feature)
                        if i==featNo2:
                            Y.append(feature)
        self.normaliseFeatureVector(X)
        plot(X,Y,color=style[0],marker=style[1],alpha=style[2])


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
            analysis.append((language,float(success*100)/Total))
        return analysis

#ep=[3000,4000,5000,8000,10000,20000,50000]
"""ep=[9000]
hl=[(5,2),(6,2),(7,2)]
J=[]
K=[]
for i in ep:
    AppConfig.epoch=i
    X=scoreModel(["en","it"],["asd","sdf","asd"],AppConfig.getEpoch())
    Y=X.train()
    A=X.analyse()
    print A
    J.append(A[0][1])
    K.append(A[1][1])
    print "done epoch",i
plot(ep,J,"r-")
plot(ep,K,'g-')
show()"""
a=datetime.datetime.now()
X=scoreModel(["en","it","de"],["asd","sdf","asd"],AppConfig.getEpoch())
print X.populateFeatureVector()
b=datetime.datetime.now()
c=b-a
print c.seconds
#X.train()
print X.analyse()
"""
X.plotFeature2D("en","Train",1,1,2,("b","o",0.5),100)
X.plotFeature2D("it","Train",1,1,2,("g","o",0.3),100)
X.plotFeature2D("en","Test",1,1,2,("pink","o",0.5),2)
X.showPlot()"""

