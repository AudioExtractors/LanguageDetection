import os
epoch=100
base_dir = "Data"
windowSize = 1000  # In number of frames
windowHop = 500  # In number of frames
languages = ["en", "de", "it", "es", "ru", "fr"]
numFeatures = 34
featureNames = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread',
                'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4',
                'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13',
                'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5',
                'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10',
                'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']
featureNumbers = [i for i in range(numFeatures)]  # Can be changed accordingly


def getFilePathTraining(language, number):
    range_start = number - number % 100
    folder = str(range_start) + "-" + (str(range_start+99))
    path = os.path.join(base_dir, language, folder, language + "_train" + str(number) + ".wav")
    return path


def getFeaturesNumbers():
    return featureNumbers


def getWindowSize():
    return windowSize


def getWindowHop():
    return windowHop
def getEpoch():
    return epoch
def getFeatureSet():
    return featureNames
def getLanguages():
    return languages
