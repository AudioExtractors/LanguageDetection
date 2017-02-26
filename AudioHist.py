from pyAudioAnalysis import audioFeatureExtraction
import numpy
import matplotlib.pyplot as plt
import os

featureNames = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux',
         'Spectral Rolloff', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11',
         'MFCC 12', 'MFCC 13', 'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5', 'Chroma Vector 6',
         'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10', 'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']
languages = ["en", "de", "ru", "es", "fr"]
AvgFeatures = ["Mean", " Standard Deviation"]
Features = [[[] for x in range(len(featureNames) * len(AvgFeatures))] for y in range(len(languages))]
dir_path = os.path.dirname(os.path.realpath(__file__))
Folder = ""

for language in languages:
    Folder += language + ", "
Folder = Folder[0:-2]
save_dir = os.path.join(dir_path, "Histograms", "Features Histogram " + "(" + Folder + ")\\")

for i in range(len(languages)):
    dirList = [dir[0] for dir in os.walk(os.path.join(dir_path, "Data", languages[i]))]
    dirList = dirList[1:]
    for dirs in dirList:
        [mtFeatures, _, _] = audioFeatureExtraction.dirWavFeatureExtractionNoAveraging(os.path.join(dir_path, "Data", languages[i], dirs), 2.0, 1.0, 0.05, 0.025)
        for times in mtFeatures:
            count = 0
            for features in times:
                Features[i][count].append(features)
                count += 1

# Plot Colors in order of languages Array -> Blue, Orange, Green, Red, Purple
for numOfAvgFeatures in range(len(AvgFeatures)):
    for numOfStats in range(len(featureNames)):
        plt.figure(numOfStats)
        plt.title(AvgFeatures[numOfAvgFeatures] + " " + featureNames[numOfStats])
        for i in range(len(languages)):
            H, binEdges = numpy.histogram(Features[i][numOfStats + len(featureNames) * numOfAvgFeatures], 20)
            binCentres = 0.5 * (binEdges[1:] + binEdges[:-1])
            plt.plot(binCentres, H)
        plt.savefig(save_dir + "figure_" + str(numOfStats) + "-" + str(numOfAvgFeatures) + ".png")
        plt.close()
