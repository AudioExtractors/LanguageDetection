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
    def __init__(self,languages,featureSets,epoch):

        self.label=dict()
        self.languages=languages
        self.featureSets=featureSets
        self.epoch=epoch
        #self.classifier=Classifier.Classifier(AppConfig.getHiddenLayer(),AppConfig.getEpoch())
        self.classifier=Classify.Classify()
        self.inputFeatureVector=[]#Feature Vector
        self.inputClassVector=[]#ClassVector
        for i,language in enumerate(languages):
            self.label[language]=i



    def populateFeatureVector(self):
        """
        :return: return list of number of files trained for each language
        """

        languagesFeatures=[]
        X=[]
        Y=[]
        noOfFilesTrained=[]
        for language in self.languages:
            print "Fetching Data for ",language
            inputSize=0
            flag=0
            samples=AudioIO.getTrainingSamples(language,random="False")
            print "yes"
            ct=0
            for sample in samples:
                featureVector=sample.getContextFeatureVector()
                for frameFeature in featureVector:
                    if inputSize>=self.epoch:
                        noOfFilesTrained.append((language,sample.getIndex()))
                        #languagesFeatures.append(X)
                        flag=1
                        break
                    X.append(frameFeature)
                    print(process.memory_info().rss)/(1024*1024)
                    #print type(X),type(x[0])
                    #print (len(X)*len(X[0])*64)/float(1024*1024*8), "..MB"
                    Y.append(self.label.get(language))
                    inputSize+=1
                if flag==1:
                    break

        #print X
        print "Fetched Feature Vector.."


        X=self.normaliseFeatureVector(X)
        print "Normalised Feature Vector.."
        print "current memory usage : ", (process.memory_info().rss)/(1024*1024)
        print X
        self.assertFeatureVector(X,Y)
        #print X
        self.classifier.train(X,Y)
        return noOfFilesTrained


    def train(self):
        self.classifier.train(self.inputFeatureVector,self.inputClassVector)


    def dumpFeatureVector(self):
        """
        :return: return list of number of files trained for each language
        """
        languagesFeatures=[]
        X=[]
        Y=[]
        noOfFilesTrained=[]
        dumpLength=0
        currentDumpSize=0
        for language in self.languages:
            print "Fetching Data for ",language
            inputSize=0
            flag=0
            samples=AudioIO.getTrainingSamples(language,random="False")
            ct=0
            for sample in samples:
                featureVector=sample.getContextFeatureVector()
                print len(featureVector)
                if len(featureVector)>0:
                    featuresPerFrame=len(featureVector[0])
                else:
                    continue
                for frameFeature in featureVector:
                    print "Current DS",currentDumpSize
                    print "Compare",inputSize,self.epoch
                    if inputSize>=self.epoch:
                        noOfFilesTrained.append((language,sample.getIndex()))
                        #languagesFeatures.append(X)
                        flag=1
                        break
                    if currentDumpSize + featuresPerFrame > AppConfig.trainingBatchSize:
                        currentDumpSize=0
                        print "Created dumpX_"+str(dumpLength)
                        print "Created dumpY_"+str(dumpLength)
                        np.save("dumpX_"+str(dumpLength),X)
                        np.save("dumpY_"+str(dumpLength),Y)
                        dumpLength+=1
                        X=[]
                        Y=[]
                    currentDumpSize+=featuresPerFrame
                    X.append(frameFeature)
                    Y.append(self.label.get(language))
                    inputSize+=1
                if flag==1:
                    break

        #print X
        print "Created dumpX_"+str(dumpLength)
        print "Created dumpY_"+str(dumpLength)
        np.save("dumpX_"+str(dumpLength),X)
        np.save("dumpY_"+str(dumpLength),Y)
        print "Created Dump.."

        """
        X=self.normaliseFeatureVector(X)
        print "Normalised Feature Vector.."
        print "current memory usage : ", (process.memory_info().rss)/(1024*1024)
        print X
        self.assertFeatureVector(X,Y)
        #print X
        self.classifier.train(X,Y)"""
        return noOfFilesTrained


    def assertFeatureVector(self,X,Y):
        if len(X)==len(Y):
            print "len Assert Pass",len(X)
        for frameFeature in X:
            if frameFeature.shape!=X[0].shape:
                print "Dimension Assert Fail"
            for feature in frameFeature:
                if not(feature>=0.0 and feature <=1.0):
                    print "fail"
        print "Dimension Assert Pas",X[0].shape
    def predict(self,audio):
        featureVector=audio.getContextFeatureVector()
        normFeatureVector=self.normaliseFeatureVector(featureVector)
        return self.classifier.predict(normFeatureVector)
        #return self.classifier.predict(featureVector)


    def normaliseFeatureVector(self,X):
        Xmin=np.min(X,axis=0)
        print(process.memory_info().rss)/(1024*1024)
        Xmax=np.max(X,axis=0)
        delta=np.subtract(X,Xmin)
        diff=np.subtract(Xmax,Xmin)
        print(process.memory_info().rss)/(1024*1024)
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
        bar_len = 60
        for language in self.languages:
            Total=AppConfig.getTestEpoch()
            success=0
            print language
            for num in range(AppConfig.getTestEpoch()):
                progress=num*100/Total
                sys.stdout.write(str(progress)+" ")
                subcandidates=self.predict(Audio.Audio(AppConfig.getFilePathTest(language,num)))
                if subcandidates[0][1]==self.label[language]:
                    success+=1
            analysis.append((language,float(success*100)/Total))
        return analysis

a=datetime.datetime.now()
X = scoreModel(AppConfig.languages, ["asd", "sdf", "asd"], AppConfig.getEpoch())

print X.dumpFeatureVector()
b=datetime.datetime.now()
c=b-a
print c.seconds

