import matplotlib.pyplot as plt
import numpy as np
import wave
import AppConfig
import scoreModel
import Audio

path = 'C:\\Users\\win 8.1\\Desktop\\1.wav'
spf = wave.open(path, 'r')
fs = spf.getframerate()
X = scoreModel.scoreModel(AppConfig.languages, AppConfig.getTrainingDataSize())
X.normFeature()
X.selectFeature()
# X.train()
X.loadNN("NN")
# X.binaryTrain()
X.loadBinaryNN("Binary")
plt.figure(1)
plt.title('Signal Wave')
color = ['r', 'g', 'b']
numSeconds = 2
while spf.tell() != spf.getnframes():
    signal = spf.readframes(fs * numSeconds)
    signal = np.fromstring(signal, 'Int16')
    Time = np.linspace(float(spf.tell() - numSeconds * fs) / fs, float(spf.tell()) / fs, num=len(signal))
    results = X.predict(Audio.Audio(None, signal=signal))
    lang = results[0][1]
    plt.plot(Time, signal, color[lang])
plt.show()
