import Classifier
import AudioIO
import numpy as np
import Audio
import AppConfig
class scoreModel:
    def __init__(self,languages,featureSets,epoch):
        self.label=dict()
        self.languages=languages
        self.featureSets=featureSets
        self.epoch=epoch
        self.classifier=Classifier.Classifier()
        for i,language in enumerate(languages):
            self.label[language]=i
    def train(self):
        X=[]
        Y=[]
        inputSize=0
        flag=0
        for language in self.languages:
            samples=AudioIO.getTrainingSamples(language)
            for sample in samples:
                featureVector=sample.getFeatureVector()
                for frameFeature in featureVector:
                    if(inputSize>self.epoch):
                        flag=1
                        break
                    X.append(frameFeature)
                    Y.append(self.label.get(language))
                    inputSize+=1
                if flag==1:
                    break
            if flag==1:
                break
        X=np.array(X)
        Y=np.array(Y)
        X=self.normaliseFeatureVector(X)
        self.classifier.train(X,Y)
    def predict(self,audio):
        featureVector=audio.getFeatureVector()
        print self.classifier.predict(self.normaliseFeatureVector(featureVector))
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

X=scoreModel(["en","de"],["asd","sdf","asd"],5)
X.train()
X.predict(Audio.Audio(AppConfig.getTestSamplePath("en",100)))

