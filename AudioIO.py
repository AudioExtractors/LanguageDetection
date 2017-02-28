import os
import tarfile
# import python_speech_features as psf
from pydub import AudioSegment
from scipy import signal
import numpy as np
from pydub.playback import play
import scipy.io.wavfile as wavfile
from pydub.playback import play
from matplotlib.pyplot import plot,subplot,xlabel,ylabel,show,figure
import matplotlib.pyplot as plt
from scipy import fft,arange
from AppConfig import *
from pyAudioAnalysis import audioFeatureExtraction
import Audio
import AppConfig
import random as rd
#TUTS
#os.mkdir('')
#os.mkdirs('')
#os.listdir()
#os.getcwd()
#os.chdir()
#song = AudioSegment.from_mp3('vocal.mp3')
#data = numpy.fromstring(song._data, numpy.int16)

def read(file):
    x=wavfile.read(file)
    sound=x[1]
    fs=x[0]
    samples=len(sound)
    time=5
    #time=float(samples)/fs
    #print time
    timestamp=[time*float(i)/samples for i in range(0,samples)]
    figure(1)
    xlabel('Time(sec)')
    ylabel('Magnitude')
    plot(timestamp,sound[0:samples])

def spectrum(file):
    x=wavfile.read(file)

    y=x[1]
    Fs=x[0]
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range
    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    figure(2)
    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')
    show()
def preprocess():
    files=[]
    for i in os.walk('dataset'):
        for f in i[2]:
            files.append(str(i[0])+"\\"+f)
    L=len(files)
    ct=0
    for num,f in enumerate(files):
        if(".flac" not in f):
            continue
        print str(num),L
        if ct%100==0:
            D=str(ct)+"-"+str(ct+99)
            os.mkdir("DNEW/"+D)
        sound = AudioSegment.from_file(f, format="flac")
        sound.export("DNEW/"+D+"/eng_train"+str(ct)+".wav",format = "wav")
        ct=ct+1
def filter():

    for i in os.walk("DNEW"):
        for num,f in enumerate(i[2]):
            x=wavfile.read(str(i[0])+"\\"+f)
            y=x[1]
            Fs=x[0]
            n = len(y) # length of the signal
            k = arange(n)
            T = n/Fs
            frq = k/T # two sides frequency range
            frq = frq[range(n/2)] # one side frequency range
            Y = fft(y)/n # fft computing and normalization
            Y = Y[range(n/2)]
            fr=np.dot(abs(Y),frq)/sum(abs(Y))
            print fr
            """mx=-1
            for k,j in enumerate(Y):
                if abs(j)>mx:
                    mx=abs(j)
                    mf=frq[k]
            if mf>500:
                print mf,f
            else:
                print f"""
def MFCCExtract(file,num,color):
    x=wavfile.read(file)
    feat=psf.mfcc(x[1],x[0])
    X=[]
    for f in feat:
        X.append(f[0])
    Y=[]
    for f in feat:
        Y.append(f[1])
    figure(num)
    xlabel("MFCC Coeff 1")
    ylabel("MFCC Coeff 2")
    plot(X,Y,color)
    plt.axis([0,50,-70,40])
    plt.grid(True)
def spectrogramPlot(file,fig):
    (fs,x)=wavfile.read(file)
    plt.figure(fig)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=1024, Fs=fs, noverlap=900)
    plt.axis([1,4,0,1000])
def getFeature():

    (fs,signal)=wavfile.read(getFilePathTraining("de",80))
    ft = audioFeatureExtraction.stFeatureExtraction(signal,fs,1000,500)
    print 1/ft[0]
def getTrainingSamples(language,rng=None,maxEpoch=AppConfig.getEpoch(),max=AppConfig.maxTrainingSamples,random="True"):
    samples=[]
    error=10
    loadedFrames=0
    maxFrames=(maxEpoch)*(AppConfig.getWindowHop()+error)
    if random=="True":
        if rng is None:
            randomNumbers=rd.sample(range(0,AppConfig.maxTrainingSamples),max)
        else:
            randomNumbers=rd.sample(range(rng[0],min(AppConfig.maxTrainingSamples,rng[1])),min(max,rng[1]-rng[0]-1))
        for i in randomNumbers:
            randomSample=Audio.Audio(AppConfig.getFilePathTraining(language,i))
            samples.append(randomSample)
            loadedFrames+=randomSample.getNoOfFrames()
            if loadedFrames>maxFrames:
                break
        return samples
    if rng is None:
        for i in range(max):
            sample=Audio.Audio(AppConfig.getFilePathTraining(language,i))
            samples.append(sample)
            loadedFrames+=sample.getNoOfFrames()
            if loadedFrames>maxFrames:
                break
    else:
        for i in range(rng[0],min(rng[1],rng[0]+max)):
            sample=Audio.Audio(AppConfig.getFilePathTraining(language,i))
            samples.append(sample)
            loadedFrames+=sample.getNoOfFrames()
            if loadedFrames>maxFrames:
                break
    return samples
def getTestSamples(language,rng=None,maxEpoch=AppConfig.getEpoch(),max=AppConfig.maxTestSamples,random="True"):
    samples=[]
    loadedFrames=0
    error=10
    maxFrames=(maxEpoch)*(AppConfig.getWindowHop()+error)
    if random=="True":
        if rng is None:
            randomNumbers=rd.sample(range(0,AppConfig.maxTestSamples),max)
        else:
            randomNumbers=rd.sample(range(rng[0],min(AppConfig.maxTestSamples,rng[1])),min(max,rng[1]-rng[0]-1))
        for i in randomNumbers:
            randomSample=Audio.Audio(AppConfig.getFilePathTest(language,i))
            samples.append(randomSample)
            loadedFrames+=randomSample.getNoOfFrames()
            if loadedFrames>maxFrames:
                break

        return samples
    if rng is None:
        for i in range(max):
            sample=Audio.Audio(AppConfig.getFilePathTest(language,i))
            samples.append(sample)
            loadedFrames+=sample.getNoOfFrames()
            if loadedFrames>maxFrames:
                break
    else:
        for i in range(rng[0],min(rng[1],rng[0]+max)):
            sample=Audio.Audio(AppConfig.getFilePathTest(language,i))
            samples.append(sample)
            loadedFrames+=sample.getNoOfFrames()
            if loadedFrames>maxFrames:
                break
    return samples
def getFeatureDumpSize():
    dumpList=[]
    for i in os.walk("Dump"):
        print i[0],i[2]
        for dumps in i[2]:
            dumpList.append(i[0]+"\\"+dumps)
    return len(dumpList)/2

if __name__ == "__main__":

    """
    x=getTestSamples("de",random="True",rng=(500,700),max=3,maxEpoch=100000)
    print len(x)
    for i in x:
        print i.getIndex()
    """

    """file="DNEW3//1100-1199//dutch_train1119.wav"
    #read("DNEW//0-99//eng_train7.wav")
    #spectrum(file)
    #filter()
    MFCCExtract(file,1,'rd')
    file2="dataset2//prac.wav"
    #MFCCExtract(file2,1,'gd')

    file3="dataset2//prac2.wav"
    #MFCCExtract(file3,1,'bd')


    file4="DNEW//100-199//eng_train145.wav"
    MFCCExtract(file4,1,'gd')"""
    #spectrogramPlot("Trash//Dutch//400-499//dutch_train403.wav",1)
    #spectrogramPlot("Trash//DutchNoise//29.wav",2)