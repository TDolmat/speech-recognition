from data_processing.text_processing import TextProcessing
import torchaudio
from visualisation.audio_plots import AudioPlots
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.mel_spectogram_function = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)

    def forward(self, x):
        x = self.mel_spectogram_function(x)  # mel spectrogram
        x = np.log(x + 1e-14)  # logrithmic, add small value to avoid inf
        return x
    

class MelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(MelSpec, self).__init__()
        self.mel_spectogram_function = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)

    def forward(self, x):
        return self.mel_spectogram_function(x)  # mel spectrogram

class SpecAugment(nn.Module):
    def __init__(self, frequency_mask, time_mask):
        super(SpecAugment, self).__init__()

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=frequency_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

    def forward(self, x):
        for _ in range(self.torch_random_value_in_range(1, 3)):
            x = self.specaug(x)
        return x
    
    def torch_random_value_in_range(self, beggining, end):
        return int((torch.rand(1, 1).item() * 100) % (end - beggining + 1)) + beggining

torch.random.manual_seed(7)

sample_speech = "PATH" 

speech_waveform, sample_rate = torchaudio.load(sample_speech)

spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)
spec = spectrogram(speech_waveform)
spec = spectrogram(speech_waveform)

log_mel_spectogram = LogMelSpec(sample_rate=sample_rate, n_mels=128,  win_length=160)
log_mel_spec = log_mel_spectogram(speech_waveform)

mel_spectogram = MelSpec(sample_rate=sample_rate, n_mels=128,  win_length=160)
mel_spec = mel_spectogram(speech_waveform)

time_masking = torchaudio.transforms.TimeMasking(time_mask_param=100)
freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=800)

time_masked = time_masking(spec)
freq_masked = freq_masking(spec)

sa = SpecAugment(20, 60)

fig, axs = plt.subplots(6, 1)

ap = AudioPlots(sample_rate)
ap.plot_waveform(speech_waveform, sample_rate, axis=axs[0], title="Original waveform", xlabel="Time (s)", ylabel="Amplitude")
ap.plot_spectrogram(spec[0], axis=axs[1], title="Spectrogram", xlabel="Time samples", ylabel="Frequency (Hz)")
ap.plot_spectrogram(time_masked[0], axis=axs[2], title="Spectrogram Augmentation - Time masking", xlabel="Time samples", ylabel="Frequency (Hz)")
ap.plot_spectrogram(freq_masked[0], axis=axs[3], title="Spectrogram Augmentation - Frequency masking", xlabel="Time samples", ylabel="Frequency (Hz)")
ap.plot_spectrogram(mel_spec[0], axis=axs[4], title="Mel Spectrogram", xlabel="Time samples", ylabel="Frequency (Mel scale)")
ap.plot_spectrogram(log_mel_spec[0], axis=axs[5], title="Logarithmic Mel Spectrogram", xlabel="Time samples", ylabel="Frequency (Mel scale)")

fig.tight_layout(pad=-1.5)

plt.show()