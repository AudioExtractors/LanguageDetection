import AppConfig
import numpy
from scipy.io import wavfile
from pyAudioAnalysis import audioFeatureExtraction


class Audio:
    def __init__(self, path):
        self.path = path
        self.singleFrame = []
        self.allFrames = []
        self.index = path
        (self.fs, self.signal) = wavfile.read(path)
        self.noFrames = len(self.signal)

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
        return numpy.array(self.allFrames)

    def getNoOfFrames(self):
        return self.noFrames

X=Audio(AppConfig.getFilePathTraining("en",22))
Z=X.getFeatureVector()
"""X=Audio(AppConfig.getFilePathTraining("en",100))
print X
FV=X.getFeatureVector()
print FV
print FV.shape
FV2=X.getFeatureVector2()
print FV2
print FV2.shape"""
