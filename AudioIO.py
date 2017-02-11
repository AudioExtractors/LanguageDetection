import os
from pydub import AudioSegment
import numpy
import scipy
from pydub.playback import play
import scipy.io.wavfile as wavfile
from pydub.playback import play
import matplotlib.pyplot as plt
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
    time=float(samples)/fs
    print time
    timestamp=[time*float(i)/samples for i in range(0,samples)]
    plt.plot(timestamp,sound[0:samples])
    plt.show()

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
if __name__ == "__main__":
    read("DNEW//0-99//eng_train7.wav")