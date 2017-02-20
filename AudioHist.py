from pyAudioAnalysis import audioFeatureExtraction
import numpy
import matplotlib.pyplot as plt
import os

names = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy',
         'Spectral Centroid', 'Spectral Spread',
         'Spectral Entropy', 'Spectral Flux',
         'Spectral Rolloff', 'MFCC 1', 'MFCC 2',
         'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6',
         'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
         'MFCC 11', 'MFCC 12', 'MFCC 13',
         'Chroma Vector 1', 'Chroma Vector 2',
         'Chroma Vector 3', 'Chroma Vector 4',
         'Chroma Vector 5', 'Chroma Vector 6',
         'Chroma Vector 7', 'Chroma Vector 8',
         'Chroma Vector 9', 'Chroma Vector 10',
         'Chroma Vector 11', 'Chroma Vector 12',
         'Chroma Deviation']

dir_path = os.path.dirname(os.path.realpath(__file__))
dir1 = os.path.join(dir_path, "DATASET", "ENG0-99")
dir2 = os.path.join(dir_path, "DATASET", "DUTCH0-99")

Feature1 = [[] for x in range(68)]
Feature2 = [[] for x in range(68)]

[mtFeatures1, _, _] = audioFeatureExtraction.dirWavFeatureExtractionNoAveraging(dir1, 2.0, 1.0, 0.05, 0.025)
for times in mtFeatures1:
    count = 0
    for features in times:
        Feature1[count].append(features)
        count += 1

[mtFeatures2, _, _] = audioFeatureExtraction.dirWavFeatureExtractionNoAveraging(dir2, 2.0, 1.0, 0.05, 0.025)
for times in mtFeatures2:
    count = 0
    for features in times:
        Feature2[count].append(features)
        count += 1

for numOfStats in range(34):
    plt.figure(numOfStats)
    plt.title("Mean " + names[numOfStats])
    H1, binEdges1 = numpy.histogram(Feature1[numOfStats], 20)
    binCentres1 = 0.5 * (binEdges1[1:] + binEdges1[:-1])
    H2, binEdges2 = numpy.histogram(Feature2[numOfStats], 20)
    binCentres2 = 0.5 * (binEdges2[1:] + binEdges2[:-1])
    plt.plot(binCentres1, H1)
    plt.plot(binCentres2, H2)
    plt.show()

for numOfStats in range(34):
    plt.figure(numOfStats)
    plt.title("Standard Deviation " + names[numOfStats])
    H1, binEdges1 = numpy.histogram(Feature1[numOfStats + 34], 20)
    binCentres1 = 0.5 * (binEdges1[1:] + binEdges1[:-1])
    H2, binEdges2 = numpy.histogram(Feature2[numOfStats + 34], 20)
    binCentres2 = 0.5 * (binEdges2[1:] + binEdges2[:-1])
    plt.plot(binCentres1, H1)
    plt.plot(binCentres2, H2)
    plt.show()
