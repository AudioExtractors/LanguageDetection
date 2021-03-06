import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import subprocess
import wave
import scoreModel
import AppConfig
import numpy as np
import Audio
import math


class FileWidget(QWidget):
    def __init__(self):
        super(FileWidget, self).__init__()
        self.figure = plt.figure(figsize=(5, 5))
        self.figure.suptitle('Audio Signal')
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_xlabel('Time')
        self.axis.set_ylabel('Amplitude')
        self.canvas.setParent(self)
        self.vbox = QVBoxLayout(self)
        self.vbox.addWidget(self.canvas)


class MainWidget(QWidget):
    def __init__(self):
        super(MainWidget, self).__init__()
        self.setWindowTitle('Voice Recognizer')
        self.setWindowIcon(QIcon('Icon\\icon.jpg'))
        self.font = QFont()
        self.font.setPointSize(9)
        self.text = QLabel('Select file to Analyse', self)
        self.text.setFont(self.font)
        self.text.setGeometry(312, 0, 140, 20)
        self.finalpredict = QLabel("", self)
        self.finalpredictfont = QFont()
        self.finalpredictfont.setBold(True)
        self.finalpredictfont.setUnderline(True)
        self.finalpredict.setFont(self.finalpredictfont)
        self.finalpredict.setGeometry(304, 80, 150, 20)
        self.fileName = QLineEdit(self)
        self.fileName.setReadOnly(True)
        self.fileName.setGeometry(10, 30, 660, 40)
        self.button = QPushButton('Browse', self)
        self.button.clicked.connect(self.getFile)
        self.button.setGeometry(680, 30, 60, 40)
        self.Widget = FileWidget()
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.Widget)
        self.vbox = QVBoxLayout()
        self.vbox.addStretch(3)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)
        self.resize(750, 630)
        self.show()
        AppConfig.includeBaseline = False
        self.classifier = self.loadNN()

    def updateText(self, file):
        self.fileName.setText(file)

    def getFile(self):
        file = QFileDialog.getOpenFileName(self, 'Open File', os.path.join(os.environ['USERPROFILE'], 'Desktop'),
                                           "mp3 files (*.mp3)")[0]
        self.updateText(os.path.basename(file))
        self.Widget.axis.cla()
        wavFile = self.createWav(file)
        self.processFile(wavFile)

    def clearFile(self, wavfile):
        os.remove(wavfile)

    def processFile(self, file):
        spf = wave.open(file, 'r')
        self.plotFigure(spf, self.classifier)
        spf.close()
        spf = wave.open(file, 'r')
        self.finalPredict(spf, self.classifier)
        spf.close()
        self.clearFile(file)

    def createWav(self, file):
        wavFile = os.path.join(os.path.dirname(file), os.path.splitext(file)[0] + ".wav")
        subprocess.call(
            ["C:\\Users\\win 8.1\\Desktop\\Dataset Download\\ffmpeg-20170411-f1d80bc-win64-static\\bin\\ffmpeg", "-i",
             file, "-ac", "1", "-ar", "16000", "-y", wavFile])
        return wavFile

    def loadNN(self):
        classifier = scoreModel.scoreModel(AppConfig.languages, AppConfig.getTrainingDataSize())
        classifier.normFeature()
        classifier.selectFeature()
        classifier.loadNN("NN")
        classifier.loadBinaryNN("Binary")
        return classifier

    def plotFigure(self, spf, classifier):
        fs = spf.getframerate()
        color = ['r', 'g', 'b']
        numSeconds = 6
        spfTellPrev = 0
        while spf.tell() != spf.getnframes():
            spfTellPrev = spf.tell()
            signal = spf.readframes(fs * numSeconds)
            signal = np.fromstring(signal, 'Int16')
            Time = np.linspace(float(float(spfTellPrev) / float(fs)), float(float(spf.tell()) / float(fs)), num=len(signal))
            results = classifier.predict(Audio.Audio(None, signal=signal))
            lang = results[0][1]
            self.Widget.axis.plot(Time, signal, color[lang])
        handles = []
        for i in range(len(AppConfig.getLanguages())):
            patch = mpatches.Patch(color=color[i], label=AppConfig.languages[i])
            handles.append(patch)
        self.Widget.axis.legend(handles=handles)
        self.Widget.axis.set_xlabel('Time')
        self.Widget.axis.set_ylabel('Amplitude')
        self.Widget.canvas.draw()

    def finalPredict(self, spf, classifier):
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        finalprediction = classifier.predict(Audio.Audio(None, signal=signal))
        self.finalpredict.setText("Predicted language: " + AppConfig.languages[finalprediction[0][1]])


def main():
    app = QApplication(sys.argv)
    ex = MainWidget()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
