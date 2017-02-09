import os
from pydub import AudioSegment
import numpy
#os.mkdir('')
#os.mkdirs('')
#os.listdir()
#os.getcwd()
#os.chdir()
files=[]
#song = AudioSegment.from_mp3('vocal.mp3')
#data = numpy.fromstring(song._data, numpy.int16)

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
