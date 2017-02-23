from scoreModel import scoreModel,predict
import resolutionModel
import AppConfig
class LIDModel:
    def __init__(self,languages,featureSet):
        """
        :param languages:list of languages
        :param featureSet: list of features to be used in Level 1 Classifiers
        :return:
        """
        self.languages=languages
        self.featureSet=featureSet
        self.RM=resolutionModel()
    def train(self,epoch):
        self.SM=scoreModel(self.languages,self.featureSet,epoch)
        self.SM.train()

    def predict(self,audio):
        """
        :param audio: instance of audio object
        :return:
        """
        subcandidates=self.SM.predict(audio)
        predictedLanguage=self.RM.predict(audio,subcandidates)
        return predictedLanguage





