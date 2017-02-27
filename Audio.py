import AppConfig
import numpy
import numpy as np
from scipy.io import wavfile
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as pp
from pydub.playback import play
from pydub import AudioSegment
import librosa
class Audio:
    def __init__(self, path):
        self.path = path
        self.singleFrame = []
        self.allFrames = []
        self.index = path
        (self.fs, self.signal) = wavfile.read(path)
        segments=aS.silenceRemoval(self.signal, self.fs, 0.020, 0.020, smoothWindow = 1.0, Weight = 0.4, plot = False)
        self.voicedSignal=np.array([],dtype=np.int16)

        for segment in segments:
            voicedStart=int(segment[0]*self.fs)
            voicedEnd=int(segment[1]*self.fs)
            self.voicedSignal=np.append(self.voicedSignal,self.signal[voicedStart:voicedEnd])

        """song = AudioSegment.from_wav(AppConfig.getFilePathTraining("en",23))
        play((self.signal,self.fs))"""

        if self.fs != 16000:
            print "sampling Error.."
            return

        self.contextFeatureVector = []
        self.noFrames = len(self.signal)
        self.featureVectorSize = -1
        self.contextFeatureVectorSize = -1

    def getFsAndSignal(self):
        return self.fs, self.signal

    def getIndex(self):
        return self.index

    def getFeatureVector(self):
        featureVectorTemp = audioFeatureExtraction.stFeatureExtraction(self.signal, self.fs, AppConfig.getWindowSize(),
                                                                       AppConfig.getWindowHop())
        Deltas = []
        Deltas2 = []
        for mfcc in range(8, 21):
            delta = librosa.feature.delta(featureVectorTemp[mfcc])
            delta2 = librosa.feature.delta(featureVectorTemp[mfcc], order=2)
            Deltas.append(delta)
            Deltas2.append(delta2)
        Deltas = numpy.array(Deltas)
        Deltas2 = numpy.array(Deltas2)
        featureVectorTemp = numpy.transpose(featureVectorTemp)
        Deltas = numpy.transpose(Deltas)
        Deltas2 = numpy.transpose(Deltas2)
        featureVector = []
        for frames in range(0, len(featureVectorTemp)):
            Temp = numpy.append(featureVectorTemp[frames], Deltas[frames])
            Temp = numpy.append(Temp, Deltas2[frames])
            featureVector.append(Temp)
        featureVector = numpy.array(featureVector)
        for frames in featureVector:
            self.singleFrame = []
            for requiredFeature in AppConfig.getFeaturesNumbers():
                self.singleFrame.append(frames[requiredFeature])
            self.allFrames.append(self.singleFrame)
        self.featureVectorSize = len(self.allFrames)
        return numpy.array(self.allFrames)

    def getContextFeatureVector(self):
        featureVector = self.getFeatureVector()
        contextFeatureVector = self.makeContextWindows(featureVector)
        self.contextFeatureVectorSize = contextFeatureVector.shape
        self.contextFeatureVector = contextFeatureVector
        return contextFeatureVector

    def makeContextWindows(self, languageFeature):
        contextFeature = []
        noOfFrames = len(languageFeature)
        for i in range(noOfFrames):
            start = i
            end = i + AppConfig.getContextWindowSize()
            if end >= noOfFrames:
                break
            context = languageFeature[start:end]
            context = np.reshape(context, context.size)
            contextFeature.append(context)
        contextFeature = np.array(contextFeature)
        return contextFeature

    def getNoOfFrames(self):
        return self.noFrames