import Classifier
import AudioIO
class scoreModel:
    def __init__(self,languages,featureSets,epoch):
        self.label=dict()
        self.languages=languages
        self.featureSets=featureSets
        self.epoch=epoch
        self.classifier=Classifier.Classifier()
        for i,language in enumerate(languages):
            self.label[language]=i
    def train(self):
        X=[]
        Y=[]
        inputSize=0
        flag=0
        for language in self.languages:
            samples=AudioIO.getSampleList(language)
            for sample in samples:
                featureVector=sample.getFeatureVectorNormalised()
                for frameFeature in featureVector:
                    if(inputSize>self.epoch):
                        flag=1
                        break
                    X.append(frameFeature)
                    Y.append(self.label.get(language))
                    inputSize+=1
                if flag==1:
                    break
            if flag==1:
                break




