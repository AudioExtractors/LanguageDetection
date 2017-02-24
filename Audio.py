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
            for requiredFeature in AppConfig.featureNumbers:
                self.singleFrame.append(frames[requiredFeature])
            self.allFrames.append(self.singleFrame)
        return numpy.array(self.allFrames)

    def getNoOfFrames(self):
        return self.noFrames

#     def getFeatureVector2(self):
#         featureVector=audioFeatureExtraction.stFeatureExtraction(self.signal,self.fs,AppConfig.getWindowSize(),AppConfig.getWindowHop())
#         filteredFeatureVector=[]
#         for i,feature in enumerate(featureVector):
#             if i>=8 and i<=20:
#                 filteredFeatureVector.append(feature)
#         self.featureVector=filteredFeatureVector
#         self.featureVector=filteredFeatureVector
#         return numpy.transpose(filteredFeatureVector)
#
# """X=Audio(AppConfig.getFilePathTraining("en",100))
# print X
# FV=X.getFeatureVector()
# print FV
# print FV.shape
# FV2=X.getFeatureVector2()
# print FV2
# print FV2.shape"""
