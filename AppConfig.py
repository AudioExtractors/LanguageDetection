base_dir="Data"
languages=["en","de","it","es","ru","fr"]
def getFilePathTraining(language,number):
    range_start=number-number%100
    folder=str(range_start)+"-"+(str(range_start+99))
    path=base_dir+"//"+language+"//"+folder+"//"+language+"_train"+str(number)+".wav"
    return path
