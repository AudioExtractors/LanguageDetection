import pyaudio
import sys
import time
import wave
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scoreModel
import AppConfig
import numpy as np
import Audio
import warnings

warnings.filterwarnings("ignore", ".*GUI is implemented.*")


def data_to_array(data):
    return np.frombuffer(data, dtype=np.int16).reshape((-1, ))


class RealTime:
    def __init__(self, languages):
        plt.rcParams['toolbar'] = 'None'
        self.format = pyaudio.paInt16
        self.chunk = 16000
        self.fs = 16000
        self.classifier = scoreModel.scoreModel(languages, AppConfig.getTrainingDataSize())
        self.classifier.normFeature()
        self.classifier.selectFeature()
        self.classifier.loadNN("NN")
        self.classifier.loadBinaryNN("Binary")
        if len(sys.argv) > 1:
            self.wf1 = wave.open(sys.argv[1], 'rb')
            self.wf2 = wave.open(sys.argv[1], 'rb')
        self.color = ['r', 'g', 'b']
        self.time = 0
        AppConfig.includeBaseline = False
        self.handles = []
        for i in range(len(AppConfig.getLanguages())):
            patch = mpatches.Patch(color=self.color[i], label=AppConfig.languages[i])
            self.handles.append(patch)

        # instantiate PyAudio (1)
        self.p = pyaudio.PyAudio()

        def out_callback(in_data, frame_count, time_info, status):
            data = self.wf2.readframes(frame_count)
            return (data, pyaudio.paContinue)

        if len(sys.argv) > 1:
            self.out_stream = self.p.open(
                    format=self.format,
                    channels=1,
                    rate=self.fs,
                    output=True,
                    stream_callback=out_callback
                )
        # define callback (2)
        def callback(in_data, frame_count, time_info, status):
            self.time += 1
            if len(sys.argv) > 1:
                in_data = self.wf1.readframes(frame_count)
            if len(in_data) <= 0:
                return None, pyaudio.paComplete
            samples = data_to_array(in_data)
            result = self.classifier.predict(Audio.Audio(None, signal=samples))
            Time = np.linspace(self.time - 1, self.time, num=len(samples))
            plt.ion()
            plt.plot(Time, samples, color=self.color[result[0][1]])
            plt.legend(handles=self.handles, loc='upper left')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            fig = plt.figure(plt.gcf().number)
            fig.canvas.set_window_title("Audio Detection")
            plt.ylim((-20000, 20000))
            plt.pause(0.001)
            # if self.time % 5 == 0:
            #     plt.clf()
            return in_data, pyaudio.paContinue

        # open stream (3)
        self.stream = self.p.open(
                    format=self.format,
                    channels=1,
                    frames_per_buffer=self.chunk,
                    rate=self.fs,
                    stream_callback=callback,
                    input=True
                )

    def run(self):
        # start the stream (4)
        self.stream.start_stream()
        if len(sys.argv) > 1:
            self.out_stream.start_stream()
        # wait for stream to finish (5)
        try:
            while self.stream.is_active():
                time.sleep(0.001)
        except KeyboardInterrupt:
            pass

    def __del__(self):
        # stop stream (6)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if len(sys.argv) > 1:
            self.out_stream.stop_stream()
            self.out_stream.close()
        # close PyAudio (7)
        self.pyaudio.terminate()

if __name__ == "__main__":
    if len(sys.argv)>1:
        print("Input source provided: %s" % sys.argv[1])
        h = sys.argv[1].split('.')
        if h[-1] != 'wav':
            print "Conerting source to .wav file."
            h[-1] = 'wav'
            dest = '.'.join(h)
            P = subprocess.call(
                ["C:\\Users\\ADITYA\\Downloads\\Telegram Desktop\\ffmpeg-20170411-f1d80bc-win64-static\\bin\\ffmpeg",
                 "-i", sys.argv[1], "-ac", "1", "-ar", "16000", "-y", dest])
            sys.argv[1] = dest
            print "Conversion complete."
    else:
        print("Using default device input source")
    X = RealTime(AppConfig.languages)
    X.run()
