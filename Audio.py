import AppConfig
import numpy
import numpy as np
from scipy.io import wavfile
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioSegmentation as aS
import os
import librosa


class Audio:
    def __init__(self, path, signal=None):
        self.path = path
        self.singleFrame = []
        self.allFrames = []
        self.index = path
        if signal is None:
            path_list = path.split(os.sep)
            indexOfLanguage = 1
            indexOfName = 3
            self.language = path_list[indexOfLanguage]
            self.name = path_list[indexOfName]
            (self.fs, signal) = wavfile.read(path)
            # segments = aS.silenceRemoval(signal, self.fs, 0.020, 0.020, smoothWindow=1.0, Weight=0.4, plot=False)
            # self.voicedSignal = np.array([], dtype=np.int16)
            # for segment in segments:
            #     voicedStart = int(segment[0]*self.fs)
            #     voicedEnd = int(segment[1]*self.fs)
            #     self.voicedSignal = np.append(self.voicedSignal, signal[voicedStart:voicedEnd])
            # self.noFrames = len(self.voicedSignal)
            # self.signal = self.voicedSignal
            self.signal = signal
            self.noFrames = len(self.signal)
        else:
            self.signal = signal
            self.fs = 16000
            self.noFrames = len(self.signal)
        if AppConfig.fixedAudioLength != 0.0:
            mid = self.noFrames/2
            halfFrame = int(self.fs*AppConfig.fixedAudioLength*0.5)
            left = max(0, mid-halfFrame)
            right = min(mid+halfFrame, self.noFrames)
            self.signal = self.signal[left:right]
            self.noFrames = len(self.signal)
        if self.fs != 16000:
            print "sampling Error.."
            return
        self.contextFeatureVector = []
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
        for mfcc in range(0, 13):
            delta = librosa.feature.delta(featureVectorTemp[mfcc], width=3)
            delta2 = librosa.feature.delta(featureVectorTemp[mfcc], width=3, order=2)
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
        # self.contextFeatureVector = contextFeatureVector   # commented to save memory
        return contextFeatureVector

    def makeContextWindows(self, languageFeature):
        contextFeature = []
        noOfFrames = len(languageFeature)
        for i in range(noOfFrames):
            start = i
            end = i + AppConfig.getContextWindowSize()
            if end > noOfFrames:
                break
            context = languageFeature[start:end]
            context = np.reshape(context, context.size)
            contextFeature.append(context)
        contextFeature = np.array(contextFeature)
        return contextFeature

    def getAverageFeatureVector(self, std=False):  # std true means standard deviation to be included as well
        featureVector = self.getFeatureVector()
        featureVector = self.makeContextWindows(featureVector)
        if featureVector.size == 0:
            return featureVector
        averageFeatureVector = self.makeAverageWindows(featureVector, AppConfig.averageFramesPerSample)
        if std == True:
            stdFeatureVector = self.makeAverageWindows(featureVector, AppConfig.averageFramesPerSample, std=True)
            if averageFeatureVector.shape != stdFeatureVector.shape:
                print "Average Features cannot be concatenated because of shape error"
                print stdFeatureVector.shape, averageFeatureVector.shape
            averageFeatureVector = np.concatenate((averageFeatureVector, stdFeatureVector), axis=1)
        return averageFeatureVector

    def makeAverageWindows(self, languageFeature, averageFramesPerSample, std=False):
        averageFeature = []
        noOfFrames = len(languageFeature)
        averagingWindowSize = max(1, noOfFrames/averageFramesPerSample)
        start = 0
        end = averagingWindowSize
        for i in range(noOfFrames):
            if start >= noOfFrames:
                break
            averagingWindow = languageFeature[start:end]
            if std == True:
                averagingWindow = averagingWindow.std(axis=0)
            else:
                averagingWindow = averagingWindow.mean(axis=0)

            averageFeature.append(averagingWindow)
            start = end
            if start + 2 * averagingWindowSize > noOfFrames:
                end = noOfFrames
            else:
                end = start + averagingWindowSize
        averageFeature = np.array(averageFeature)
        return averageFeature

    def getNoOfFrames(self):
        return self.noFrames

    def publish(self, dir):
        path = os.path.join(dir, self.language, self.name)
        np.save(path, self.signal)
