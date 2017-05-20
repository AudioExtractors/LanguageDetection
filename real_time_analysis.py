import pyaudio
import numpy as np
import time
import scoreModel
import AppConfig
import Audio

def data_to_array(data):
    return (np.frombuffer(data, dtype=np.int16).reshape((-1, )))

class RealTime:
    def __init__(self, languages):
        self.languages = languages
        self.format = pyaudio.paInt16
        self.chunk = 16000
        self.fs = 16000
        self.classifier = scoreModel.scoreModel(languages, AppConfig.getTrainingDataSize())
        self.classifier.normFeature()
        self.classifier.selectFeature()
        self.classifier.loadNN("NN")
        self.classifier.loadBinaryNN("Binary")

        # instantiate PyAudio (1)
        self.p = pyaudio.PyAudio()

        # define callback (2)
        def callback(in_data, frame_count, time_info, status):
            samples = data_to_array(in_data)
            result = self.classifier.predict(Audio.Audio(None, signal=samples))
            print self.languages[result[0][1]]
            return (None, pyaudio.paContinue)

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
        print "Say!(Press Ctrl+C to exit)"

        # wait for stream to finish (5)
        try:
            while self.stream.is_active():
                time.sleep(0.1)
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

X = RealTime(AppConfig.languages)
X.run()