import os

languages = ["ch", "fr"]

# Feature Extraction
featureNames = ["MFCC 1", "MFCC 2", "MFCC 3", "MFCC 4", "MFCC 5", "MFCC 6", "MFCC 7", "MFCC 8", "MFCC 9", "MFCC 10",
                "MFCC 11", "MFCC 12", 'MFCC 13', "MFCC Delta 1", "MFCC Delta 2", "MFCC Delta 3", "MFCC Delta 4",
                "MFCC Delta 5", "MFCC Delta 6", "MFCC Delta 7", "MFCC Delta 8", "MFCC Delta 9", "MFCC Delta 10",
                "MFCC Delta 11", "MFCC Delta 12", "MFCC Delta 13", "MFCC Delta Delta 1", "MFCC Delta Delta 2",
                "MFCC Delta Delta 3", "MFCC Delta Delta 4", "MFCC Delta Delta 5", "MFCC Delta Delta 6",
                "MFCC Delta Delta 7", "MFCC Delta Delta 8", "MFCC Delta Delta 9", "MFCC Delta Delta 10",
                "MFCC Delta Delta 11", "MFCC Delta Delta 12", "MFCC Delta Delta 13"]
featureNumbers = []
for i in range(0, 13):
    featureNumbers.append(i)
for i in range(13, 26):
    featureNumbers.append(i)
for i in range(26, 39):
    featureNumbers.append(i)
windowSize = 400  # In number of frames
windowHop = 100  # In number of frames
contextWindowSize = 5  # -x/2 to +x/2 number of frames
averageFramesPerSample = 1  # each clip treated as one sample by average out
includeStd = True
includeBaseline = False
if includeStd == True:
    numberOfAverageStats = 2
else:
    numberOfAverageStats = 1
fixedAudioLength = 0.0  # In seconds

# Data Size
trainingDataSize = 1105
test_epoch = 236
trainingBatchSize = 1000000000000
maxTrainingSamples = 1183
maxTestSamples = 253

# Directories
base_dir = "Data"
test_base_dir = "Test"
dump_train_dir = "Samples"
dump_test_dir = "TestDump"
dump_base_dir = "Dump"
logs_base_dir = "logs"
gmmLogs_base_dir = os.path.join(logs_base_dir, "gmm")
NN_save_dir = "NNFiles"

# Initial NN Characteristics
hiddenLayer = (12)  # approx (2/3) * len(featureSet) * contextWindow
batch_size = 1
nb_epoch = 30
selFeatures = 180

# Binary NN Characteristics
binaryHiddenLayer = (22)
binary_batch_size = 10
binary_nb_epoch = 30
selBinaryFeatures = 380

# GMM Features
gmmWindowSize = 400
gmmWindowHop = 250
gmmComponents = 50


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


def getLanguages():
    return languages


def getHiddenLayer():
    return hiddenLayer


def getBinaryHiddenLayer():
    return binaryHiddenLayer


def getTestEpoch():
    return test_epoch


def getContextWindowSize():
    return contextWindowSize


def getBatchSize():
    return batch_size


def getBinaryBatchSize():
    return binary_batch_size


def getNumberEpochs():
    return nb_epoch


def getBinaryNumberEpochs():
    return binary_nb_epoch


def getNumberOfAverageStats():
    return numberOfAverageStats
