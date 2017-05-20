import pyaudio
import time
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
        self.color = ['r', 'g', 'b']
        self.time = 0
        AppConfig.includeBaseline = False
        self.handles = []
        for i in range(len(AppConfig.getLanguages())):
            patch = mpatches.Patch(color=self.color[i], label=AppConfig.languages[i])
            self.handles.append(patch)

        # instantiate PyAudio (1)
        self.p = pyaudio.PyAudio()

        # define callback (2)
        def callback(in_data, frame_count, time_info, status):
            self.time += 1
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
            plt.pause(0.001)
            if self.time % 5 == 0:
                plt.clf()
            return None, pyaudio.paContinue

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

        # wait for stream to finish (5)
        try:
            while self.stream.is_active():
                time.sleep(0.001)
        except Exception as e:
            raise e
        except KeyboardInterrupt:
            pass

    def __del__(self):
        # stop stream (6)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        # close PyAudio (7)
        self.pyaudio.terminate()

if __name__ == "__main__":
    X = RealTime(AppConfig.languages)
    X.run()
