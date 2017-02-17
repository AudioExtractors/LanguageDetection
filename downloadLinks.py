import wget
import re
f=open('in.txt')
x=f.read()
y=x.split('\n')
ct=51
for g in y:
    sp=g.split(" ")
    wget.download("http://www.repository.voxforge1.org/downloads/Dutch/Trunk/Audio/Main/16kHz_16bit/"+sp[4],out="F:\BTP\dataset2")
    if ct>200:
        break
    ct=ct+1
    print ct
