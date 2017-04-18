import numpy as np
from AppConfig import *
import Audio
import AppConfig
import random as rd


def getTrainingSamples(language, rng=None, maxEpoch=AppConfig.getTrainingDataSize(), max=AppConfig.maxTrainingSamples,
                       random="False"):
    samples = []
    if random == "True":
        if rng is None:
            randomNumbers = rd.sample(range(0, AppConfig.maxTrainingSamples), max)
        else:
            randomNumbers = rd.sample(range(rng[0], min(AppConfig.maxTrainingSamples, rng[1])), min(max,
                                                                                                    rng[1]-rng[0]-1))
        for i in randomNumbers:
            randomSample = Audio.Audio(AppConfig.getFilePathTraining(language, i))
            samples.append(randomSample)
        return samples
    if rng is None:  # most imp
        for i in range(max):
            sample = Audio.Audio(AppConfig.getFilePathTraining(language, i))
            samples.append(sample)
    else:
        for i in range(rng[0], min(rng[1], rng[0]+max)):
            sample = Audio.Audio(AppConfig.getFilePathTraining(language, i))
            samples.append(sample)
    return samples


def dumpTrainFiles(language):
    samples = getTrainingSamples(language)
    for sample in samples:
        sample.publish(AppConfig.dump_train_dir)


def dumpTestFiles(language):
    samples = getTestSamples(language)
    for sample in samples:
        sample.publish(AppConfig.dump_test_dir)


def getDumpTestSample(language):
    samples = []
    for files in os.walk(os.path.join(AppConfig.dump_test_dir, language)):
        for file in files[2]:
            path = os.path.join(str(files[0]), file)
            sig = np.load(path)
            audio = Audio.Audio(path, sig)
            samples.append(audio)
    return samples


def getDumpTrainingSample(language):
    samples = []
    for files in os.walk(os.path.join(AppConfig.dump_train_dir, language)):
        for file in files[2]:
            path = os.path.join(str(files[0]), file)
            sig = np.load(path)
            audio = Audio.Audio(path, sig)
            samples.append(audio)
    return samples


def getTestSamples(language, rng=None, max=AppConfig.test_epoch, random="False"):
    samples = []
    if random == "True":
        if rng is None:
            randomNumbers = rd.sample(range(0, AppConfig.maxTestSamples), max)
        else:
            randomNumbers = rd.sample(range(rng[0], min(AppConfig.maxTestSamples, rng[1])), min(max, rng[1]-rng[0]-1))
        for i in randomNumbers:
            randomSample = Audio.Audio(AppConfig.getFilePathTest(language, i))
            samples.append(randomSample)
        return samples
    if rng is None:  # most imp
        for i in range(max):
            sample = Audio.Audio(AppConfig.getFilePathTest(language, i))
            samples.append(sample)
    else:
        for i in range(rng[0], min(rng[1], rng[0]+max)):
            sample = Audio.Audio(AppConfig.getFilePathTest(language, i))
            samples.append(sample)
    return samples


def getFeatureDumpSize():
    dumpList = []
    for i in os.walk("Dump"):
        for dumps in i[2]:
            dumpList.append(i[0]+"\\"+dumps)
    return len(dumpList)/(2*AppConfig.getNumLanguages())
