import AppConfig
from scipy.io import wavfile
from pyAudioAnalysis import audioFeatureExtraction
class Audio:
    def __init__(self,path):
        self.path=path
        (self.fs,self.signal)=wavfile.read(path)
    def populateFeatureVector(self):
        featureVector=audioFeatureExtraction.stFeatureExtraction(self.signal,self.fs,AppConfig.getWindowSize(),AppConfig.getWindownHop())
    def getFeatureVector(self):
