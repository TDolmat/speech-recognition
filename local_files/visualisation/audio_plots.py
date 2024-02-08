import torch
import torchaudio
import librosa

import matplotlib.pyplot as plt

class AudioPlots():
    def __init__(self, DEFAULT_SAMPLE_RATE = 32000):
        self.DEFAULT_SAMPLE_RATE = DEFAULT_SAMPLE_RATE


    def plot_audio(self, waveform_audio=None, spectrogram_audio=None, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.DEFAULT_SAMPLE_RATE

        if waveform_audio is not None and spectrogram_audio is not None: # Both waveform and spectrogram are specified
            fig, axs = plt.subplots(2, 1)

            self.plot_waveform(waveform_audio, sample_rate, title="Original waveform", axis=axs[0])
            self.plot_spectrogram(spectrogram_audio[0], title="Spectrogram", axis=axs[1])

            # fig.tight_layout()
        elif waveform_audio is not None: # Only waveform is specified
            self.plot_waveform(waveform_audio, sample_rate, title="Waveform")

        elif spectrogram_audio is not None: # Only spectrogram is specified
            self.plot_spectrogram(spectrogram_audio[0], title="spectrogram")

        else:
            print("Nothing diplayable provided")
            return
        
        plt.show()

    def plot_waveform(self, waveform, sample_rate, title="Waveform", axis=None, xlabel=None, ylabel=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        if axis is None:
            _, axis = plt.subplots(num_channels, 1)
        axis.plot(time_axis, waveform[0], linewidth=1)
        axis.grid(True)
        axis.set_xlim([0, time_axis[-1]])
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)


    def plot_spectrogram(self, specgram, title=None, axis=None, xlabel=None, ylabel=None):
        if axis is None:
            _, axis = plt.subplots(1, 1)
        if title is not None:
            axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")