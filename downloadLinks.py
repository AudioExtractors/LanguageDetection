import wget
import re
import os,tarfile
import shutil
def extract():
    files=[]
    for i in os.walk('dataset2'):
        for f in i[2]:
            files.append(str(i[0])+"\\"+f)
    L=len(files)
    print files
    ct=0
    max=len(files)
    for num,f in enumerate(files):
        if(".tgz" not in f):
            continue
        tar=tarfile.open(f)
        tar.extractall("DNEW2")
        print num,max

def recompile():
    files=[]
    for i in os.walk('DNEW2'):
        for f in i[2]:
            files.append(str(i[0])+"\\"+f)
    L=len(files)
    print files
    ct=0
    max=len(files)
    for num,f in enumerate(files):

        if(".wav" not in f):
            continue
        if(ct%100==0):
            dir=str(ct)+"-"+str(ct+99)
            os.mkdir("DNEW3\\"+dir+"\\")
        shutil.copy(f,"DNEW3\\"+dir+"\\"+"dutch_train"+str(ct)+".wav")
        print ct
        ct=ct+1
def download():
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