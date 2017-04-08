import numpy as np
import AppConfig
import AudioIO
from feature_selection import FeatureSelection
from feature_normalise import FeatureNormalise

def normF(languages, dumpSize):
    norm = FeatureNormalise(AppConfig.getNumFeatures() * AppConfig.getNumberOfAverageStats() * AppConfig.getContextWindowSize())
    for language in languages:
        for i in xrange(dumpSize):
            X = np.load("Dump//dumpX_"+language+str(i)+".npy")
            norm.batchData(X)
    norm.fit()
    return norm

def selF(sel, norm, lang, dumpSize):
    for i in xrange(dumpSize):
        X = np.load("Dump//dumpX_"+lang+str(i)+".npy")
        y = np.load("Dump//dumpY_"+lang+str(i)+".npy")
        sel.batchData(norm.transform(X), y)

# Create confusion matrix dump
if __name__ == "__main__":
   dumpSize = AudioIO.getFeatureDumpSize()
   masks = {}
   languages = AppConfig.languages
   norm = normF(languages, dumpSize)
   for i in xrange(len(languages)-1):
       for j in xrange(i+1,len(languages)):
           print languages[i],languages[j]
           sel = FeatureSelection(2, AppConfig.getNumFeatures() * AppConfig.getNumberOfAverageStats() *
                                  AppConfig.getContextWindowSize(), labels=[i,j], k=AppConfig.selFeatures)
           selF(sel,norm,languages[i],dumpSize)
           selF(sel,norm,languages[j],dumpSize)
           sel.fit()
           masks[(languages[i], languages[j])] = sel.mask
           masks[(languages[j], languages[i])] = sel.mask
           print sel.mask
   np.save("Dump\\confusion_matrix", masks)
   #print masks
           
           
           
