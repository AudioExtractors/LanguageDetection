import os
trainingDataSize = 1199*80  # must be greater than 2*contextWindow
hiddenLayer = (15)  # approx (2/3)*len(featureSet)*contextWindow
windowSize = 400  # In number of frames
windowHop = 160  # In number of frames
languages = ["de", "it", "en"]
test_epoch = 300
contextWindowSize = 4  # -x/2 to +x/2 number of frames
maxTrainingSamples = 1210
maxTestSamples = 1210
trainingBatchSize = 7000000  # 78 features * 100 samples
averageFramesPerSample = 1000000000000  # each clip treated as one sample by average out
batch_size = 32
nb_epoch = 20
numberOfAverageStats = 2
fixedAudioLength = 1.5  # seconds
selFeatures = 30
####GMM Features
gmmWindowSize = 400
gmmWindowHop = 250
gmmComponents = 50
####
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
                "MFCC Delta Delta 10", "MFCC Delta Delta 11", "MFCC Delta Delta 12", "MFCC Delta Delta 13", "Formant 1",
                "Formant 2", "Formant 3", "Formant 4", "Formant 5"]


# featureNumbers = [i for i in range(34)]  # Can be changed accordingly
featureNumbers = []
# featureNumbers.append(0)
# featureNumbers.append(1)
for i in range(9, 21):
    featureNumbers.append(i)
for i in range(35, 47):
    featureNumbers.append(i)
# for i in range(48, 60):
#     featureNumbers.append(i)
# for i in range(60, 62):
#     featureNumbers.append(i)

base_dir = "Data"
test_base_dir = "Test"
dump_train_dir = "Samples"
dump_test_dir = "TestDump"
dump_base_dir = "Dump"
logs_base_dir = "logs"
gmmLogs_base_dir = os.path.join("logs", "gmm")


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


def getFilePathTrainingDump(language, number):
    range_start = number - number % 100
    folder = str(range_start) + "-" + (str(range_start+99))
    path = os.path.join(dump_train_dir, language, language+"_train"+str(number)+".wav"+".npy")
    return path


def getNumFeatures():
    return len(featureNumbers)


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


def getBatchSize():
    return batch_size


def getNumberEpochs():
    return nb_epoch


def getNumberOfAverageStats():
    return numberOfAverageStats
