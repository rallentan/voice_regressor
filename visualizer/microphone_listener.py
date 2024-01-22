import pyaudio
import numpy as np
import threading
import collections

class MicrophoneListener:
    def __init__(self, chunk=1024, format=pyaudio.paInt16, channels=1, rate=44100, buffer_duration=0.5):
        self.CHUNK = chunk
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self.BUFFER_DURATION = buffer_duration  # Duration to keep in buffer, in seconds

        self.microphone = pyaudio.PyAudio()
        self.stream = None
        self.buffer = collections.deque(maxlen=int(rate / chunk * buffer_duration))
        self.lock = threading.Lock()
        self.listening = False

    def terminate(self):
        self.listening = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.microphone.terminate()

    def list_input_devices(self):
        info = self.microphone.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        for i in range(0, num_devices):
            if (self.microphone.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", self.microphone.get_device_info_by_host_api_device_index(0, i).get('name'))

    def start_listening(self, input_device_index=None):
        self.stream = self.microphone.open(format=self.FORMAT,
                                           channels=self.CHANNELS,
                                           rate=self.RATE,
                                           input=True,
                                           frames_per_buffer=self.CHUNK,
                                           input_device_index=input_device_index)
        self.listening = True
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _record_loop(self):
        while self.listening:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            with self.lock:
                self.buffer.append(np.frombuffer(data, dtype=np.int16))

    def get_audio(self):
        with self.lock:
            if not self.buffer:
                # Buffer is empty, return an empty array or handle as needed
                return None
            audio_data = np.concatenate(list(self.buffer))

        # Normalize the data to the range [-1, 1], as librosa.load does
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        # Return only the last 0.5 seconds of audio
        return audio_data[-int(self.RATE * self.BUFFER_DURATION):]
