import AppConfig
import numpy
from scipy.io import wavfile
from pyAudioAnalysis import audioFeatureExtraction
class Audio:
    def __init__(self, path):
        self.path = path
        self.singleFrame = []
        self.allFrames = []
        (self.fs, self.signal) = wavfile.read(path)
        self.noFrames=len(self.signal)

    def getFeatureVector(self):
        featureVector = audioFeatureExtraction.stFeatureExtraction(self.signal, self.fs, AppConfig.getWindowSize(),
                                                                   AppConfig.getWindowHop())
        for frames in featureVector:
            self.singleFrame = []
            for requiredFeature in AppConfig.featureNumbers:
                self.singleFrame.append(frames[requiredFeature])
            self.allFrames.append(self.singleFrame)
        return numpy.array(self.allFrames)
    def getFeatureVector2(self):
        featureVector=audioFeatureExtraction.stFeatureExtraction(self.signal,self.fs,AppConfig.getWindowSize(),AppConfig.getWindowHop())
        filteredFeatureVector=[]
        for i,feature in enumerate(featureVector):
            if i>=8 and i<=20:
                filteredFeatureVector.append(feature)
        self.featureVector=filteredFeatureVector
        print type(featureVector)
        return numpy.transpose(filteredFeatureVector)
    def getNoOfFrames(self):
        return self.noFrames
"""X=Audio(AppConfig.getFilePathTraining("en",34))
print X
FV=X.getFeatureVector()
print FV
print FV.shape
FV2=X.getFeatureVector2()
print FV2
print FV2.shape"""