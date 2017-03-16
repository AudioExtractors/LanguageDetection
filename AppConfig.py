import os
epoch = 800
trainingDataSize = 1100  # must be greater than 2*contextWindow
base_dir = "Data"
test_base_dir = "Test"
dump_base_dir="Dump"
hiddenLayer = (10)  # approx (2/3)*len(featureSet)*contextWindow
windowSize = 1000  # In number of frames
windowHop = 500  # In number of frames
languages = ["en", "de"]
test_epoch = 10
contextWindowSize = 1  # -x/2 to +x/2 number of frames
maxTrainingSamples = 500
maxTestSamples = 500
trainingBatchSize=78*10000 #78 features * 100 samples
averageFramesPerSample=1 #each clip treated as one sample by average out
featureNames = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread',
                'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4',
                'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13',
                'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5',
                'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10',
                'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation', "MFCC Delta 1", "MFCC Delta 2",
                "MFCC Delta 3", "MFCC Delta 4", "MFCC Delta 5", "MFCC Delta 6", "MFCC Delta 7", "MFCC Delta 8",
                "MFCC Delta 9", "MFCC Delta 10", "MFCC Delta 11", "MFCC Delta 12", "MFCC Delta 13", "MFCC Delta Delta 1"
                , "MFCC Delta Delta 2", "MFCC Delta Delta 3", "MFCC Delta Delta 4", "MFCC Delta Delta 5",
                "MFCC Delta Delta 6", "MFCC Delta Delta 7", "MFCC Delta Delta 8", "MFCC Delta Delta 9",
                "MFCC Delta Delta 10", "MFCC Delta Delta 11", "MFCC Delta Delta 12", "MFCC Delta Delta 13"]


# featureNumbers = [i for i in range(34)]  # Can be changed accordingly
featureNumbers = [i for i in range(8, 21)]
for i in range(34, 60):
    featureNumbers.append(i)


def getFilePathTraining(language, number):
    range_start = number - number % 100
    folder = str(range_start) + "-" + (str(range_start+99))
    path = os.path.join(base_dir, language, folder, language + "_train" + str(number) + ".wav")
    return path


def getFilePathTest(language, number):
    range_start = number - number % 100
    folder = str(range_start) + "-" + (str(range_start+99))
    path = os.path.join(test_base_dir, language, folder, language + "_test" + str(number) + ".wav")
    return path


def getNumFeatures():
    return 78

    #return len(featureNumbers)


def getNumLanguages():
    return len(languages)


def getFeaturesNumbers():
    return featureNumbers


def getWindowSize():
    return windowSize


def getWindowHop():
    return windowHop


def getTrainingDataSize():
    return trainingDataSize


def getFeatureSet():
    return featureNames


def getLanguages():
    return languages


def getHiddenLayer():
    return hiddenLayer


def getTestEpoch():
    return test_epoch

def getContextWindowSize():
    return contextWindowSize