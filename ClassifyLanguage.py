from sklearn.neural_network import MLPClassifier
import os
import python_speech_features as psf
import scipy.io.wavfile as wavfile
import numpy as np
from sklearn.externals import joblib

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4), random_state=1)
X=[]
Y=[]
def getSample(language,num):
    fnum=num/100
    fnum*=100
    if language=='ENG':
        folder="DNEW//"+str(fnum)+"-"+str(fnum+99)+"//"+"eng_train"+str(num)+".wav"
        return folder
    if language=='DUTCH':
        folder="DNEW3//"+str(fnum)+"-"+str(fnum+99)+"//"+"dutch_train"+str(num)+".wav"
        return folder
def train(lang="ENG",classi=1,sample=500000):
    ct=0
    for i in range(0,1000):
        x=wavfile.read(getSample(lang,i))
        feat=psf.mfcc(x[1],x[0])
        for frame_feat in feat:
            X.append(frame_feat)
            Y.append(classi)
            ct=ct+1
            if ct==sample:
                break
        if ct==sample:
            print i
            break
def pred(file):
    x=wavfile.read(file)
    feat=psf.mfcc(x[1],x[0])
    return clf.predict(feat)
if __name__=="__main__":
    try:
        clf= joblib.load('Model3.pkl')
    except:
        train(lang="ENG",classi=1)
        train(lang="DUTCH",classi=2)
        clf.fit(X,Y)
        joblib.dump(clf, 'Model3.pkl')

    print "Start"
    PASSE = 0
    PASSD = 0
    count =0
    for i in range(1200,1400):
        pred_arr=pred(getSample("ENG",i))
        englishness= np.count_nonzero(pred_arr==1)*100/len(pred_arr)
        if englishness>=50:
            PASSE=PASSE+1
        count=count+1
    for i in range(1200,1400):
        pred_arr=pred(getSample("DUTCH",i))
        englishness= np.count_nonzero(pred_arr==1)*100/len(pred_arr)
        if englishness<50:
            PASSD=PASSD+1
    print float(PASSE)*100/(count)
    print float(PASSD)*100/(count)
