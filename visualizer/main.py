import os
import soundfile as sf
from ml_model.inference_engine import InferenceEngine
from visualizer.dynamic_graph_ui import DynamicGraphUI
import tkinter as tk
import time

from visualizer.microphone_listener import MicrophoneListener

BASE_MODEL_FILENAME = 'trained_voice_model'


def main():
    root = tk.Tk()
    app = DynamicGraphUI(root)
    engine = InferenceEngine(BASE_MODEL_FILENAME)
    microphone = MicrophoneListener(rate=44100, buffer_duration=0.5)
    microphone.list_input_devices()
    microphone.start_listening(input_device_index=1)
    time.sleep(0.5)

    while True:
        try:
            audio_sample = microphone.get_audio()
            if audio_sample is None:
                time.sleep(0.1)
                continue

            # Save audio sample to a file
            #sf.write(f"audio_sample.wav", audio_sample, microphone.RATE)

            prediction = engine.predict_one(audio_sample)
            #x, y = prediction[0], prediction[1]
            x = prediction[0]

            x = (x * 2 - 1)
            #y = (y * 2 - 1) / 3

            #app.update_dot_position(x, y)
            app.update_dot_position(x, 0)

            time.sleep(0.1)

            root.update_idletasks()
            root.update()
        except tk.TclError:
            # This error occurs when the Tkinter window is closed
            break

    microphone.terminate()


if __name__ == '__main__':
    main()
