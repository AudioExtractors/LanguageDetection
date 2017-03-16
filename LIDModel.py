from scoreModel import scoreModel,predict
import resolutionModel
import AppConfig
class LIDModel:
    def __init__(self):
        """
        :param languages:list of languages
        :param featureSet: list of features to be used in Level 1 Classifiers
        :return:
        """
        self.languages=AppConfig.languages
        self.featureSet=AppConfig.featureSet
        self.RM=resolutionModel()
    def train(self,epoch):
        self.SM=scoreModel(self.languages, self.featureSet, AppConfig.trainingDataSize)
        self.SM.train()

    def predict(self,audio):
        """
        :param audio: instance of audio object
        :return:
        """
        subcandidates=self.SM.predict(audio)
        predictedLanguage=self.RM.predict(audio,subcandidates)
        return predictedLanguage





