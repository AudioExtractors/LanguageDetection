import wget
import re
import os,tarfile
import shutil
def extract():
    files=[]
    for i in os.walk('Raw\\DutchRawTest'):
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
        tar.extractall("Test\\de_raw")
        print num,max

def recompile():
    files=[]
    for i in os.walk('Test\\en_raw'):
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
            os.mkdir("Test\\en\\"+dir+"\\")
        shutil.copy(f,"Test\\en\\"+dir+"\\"+"en_test"+str(ct)+".wav")
        print ct
        ct=ct+1
def recompile2():
    files=[]
    for i in os.walk('Data\\it_raw'):
        for f in i[2]:
            files.append(str(i[0])+"\\"+f)
    L=len(files)
    print files
    ct=998
    max=len(files)
    dir="900-999"
    for num,f in enumerate(files):

        if(".wav" not in f):
            continue
        if(ct%100==0):
            dir=str(ct)+"-"+str(ct+99)
            os.mkdir("Data\\it\\"+dir+"\\")
        shutil.copy(f,"Data\\it\\"+dir+"\\"+"it_train"+str(ct)+".wav")
        print ct
        ct=ct+1

def download():
    f=open('english.txt')
    x=f.read()
    y=x.split('\n')
    ct=0
    for g in y:
        sp=g.split(" ")
        wget.download("http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/"+sp[4],out="F:\BTP\EnglishRaw")
        ct=ct+1
        print ct
if __name__ == "__main__":
    recompile2()
