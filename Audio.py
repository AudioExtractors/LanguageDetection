import AppConfig
import numpy
import numpy as np
from scipy.io import wavfile
from pyAudioAnalysis import audioFeatureExtraction


class Audio:
    def __init__(self, path):
        self.path = path
        self.singleFrame = []
        self.allFrames = []
        self.index = path
        (self.fs, self.signal) = wavfile.read(path)
        self.contextFeatureVector=[]

        self.noFrames = len(self.signal)

        self.featureVectorSize=-1
        self.contextFeatureVectorSize=-1

    def getIndex(self):
        return self.index

    def getFeatureVector(self):
        featureVector = audioFeatureExtraction.stFeatureExtraction(self.signal, self.fs, AppConfig.getWindowSize(),
                                                                   AppConfig.getWindowHop())
        for frames in numpy.transpose(featureVector):
            self.singleFrame = []
            for requiredFeature in AppConfig.getFeaturesNumbers():
                self.singleFrame.append(frames[requiredFeature])
            self.allFrames.append(self.singleFrame)
        self.featureVectorSize=len(self.allFrames)
        return numpy.array(self.allFrames)

    def getContextFeatureVector(self):
        featureVector=self.getFeatureVector()
        contextFeatureVector=self.makeContextWindows(featureVector)
        self.contextFeatureVectorSize=contextFeatureVector.shape
        self.contextFeatureVector=contextFeatureVector
        return contextFeatureVector

    def makeContextWindows(self,languageFeature):
        contextFeature=[]
        noOfFrames=len(languageFeature)
        for i in range(noOfFrames):
            start=i
            end=i+AppConfig.getContextWindowSize()
            if end>=noOfFrames:
                break
            context=languageFeature[start:end]
            context=np.reshape(context,context.size)
            contextFeature.append(context)
        contextFeature=np.array(contextFeature)
        return contextFeature

    def getNoOfFrames(self):
        return self.noFrames
"""
X=Audio(AppConfig.getFilePathTraining("en",22))
Z=X.getContextFeatureVector()
print X.featureVectorSize
print X.contextFeatureVectorSize"""