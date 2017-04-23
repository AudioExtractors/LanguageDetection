import AppConfig
import scoreModel
import confusion_matrix
WS=[400,410]#window size
WH=[100,100]#window hop
CWS=[1,1]#context WS
AFPS=[1,1]#Average frames per sample
ISTD=[True,True]#is std included
#above four change dump as well so keep equal values consecutive to each other so dumps are not recreated again
HL=[(11),(12,3)]#hidden layer
BS=[32,30]#batch size
NE=[26,30]#epoch
SF=[39,30]
sz=len(WS)
def change(i):

    if i==0:
        return True

    if WS[i]!=WS[i-1]:
        return True

    if WH[i]!=WH[i-1]:
        return True

    if CWS[i]!=CWS[i-1]:
        return True

    if AFPS[i]!=AFPS[i-1]:
        return True

    if ISTD[i]!=ISTD[i-1]:
        return True

    return False

for i in range(sz):
    AppConfig.windowSize = WS[i]
    AppConfig.windowHop = WH[i]
    AppConfig.contextWindowSize = CWS[i]
    AppConfig.averageFramesPerSample = AFPS[i]
    AppConfig.includeStd = ISTD[i]
    if AppConfig.includeStd == True:
        AppConfig.numberOfAverageStats = 2
    else:
        AppConfig.numberOfAverageStats = 1
    AppConfig.hiddenLayer = HL[i]
    AppConfig.batch_size = BS[i]
    AppConfig.nb_epoch = NE[i]
    AppConfig.selFeatures = SF[i]

    X = scoreModel.scoreModel(AppConfig.languages, AppConfig.getTrainingDataSize())
    print "Iteration: ",i
    if change(i):
        pass
        #X.dumpFeatureVector()
    confusion_matrix.dumpConfusionMatrix()
    X.normFeature()
    X.selectFeature()
    X.train()
    X.binaryTrain()
    print "Iteration: ",i
    X.analyse()

