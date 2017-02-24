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

    def getFeatureVector(self):
        featureVector = audioFeatureExtraction.stFeatureExtraction(self.signal, self.fs, AppConfig.getWindowSize(),
                                                                   AppConfig.getWindowHop())
        for frames in featureVector:
            self.singleFrame = []
            for requiredFeature in AppConfig.featureNumbers:
                self.singleFrame.append(frames[requiredFeature])
            self.allFrames.append(self.singleFrame)
        return numpy.array(self.allFrames)
