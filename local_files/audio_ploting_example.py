from data_processing.text_processing import TextProcessing
import torchaudio
from visualisation.audio_plots import AudioPlots
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from IPython.display import Audio


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

    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


def preview(sp, rate=16000):
    ispec = torchaudio.transforms.InverseSpectrogram()
    waveform = ispec(sp)

    return Audio(waveform[0].numpy().T, rate=rate)


sample_speech = "PATH" 

speech_waveform, sample_rate = torchaudio.load(sample_speech)

spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)
spec = spectrogram(speech_waveform)
spec = spectrogram(speech_waveform)


log_mel_spectogram = LogMelSpec(sample_rate=sample_rate, n_mels=128,  win_length=160)
log_mel_spec = log_mel_spectogram(speech_waveform)

mel_spectogram = MelSpec(sample_rate=sample_rate, n_mels=128,  win_length=160)
mel_spec = mel_spectogram(speech_waveform)

torch.random.manual_seed(4)

time_masking = torchaudio.transforms.TimeMasking(time_mask_param=100)
freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=800)

time_masked = time_masking(spec)
freq_masked = freq_masking(spec)



# sa = SpecAugment(sample_rate, 3, 15, 70)
# github_sol = sa(spec)

fig, axs = plt.subplots(4, 1)

ap = AudioPlots(sample_rate)
ap.plot_waveform(speech_waveform, sample_rate, axis=axs[0], title="Original waveform", xlabel="Time (s)", ylabel="Amplitude")
ap.plot_spectrogram(spec[0], axis=axs[1], title="Spectrogram", xlabel="Time samples", ylabel="Frequency (Hz)")
ap.plot_spectrogram(time_masked[0], axis=axs[2], title="Spectrogram Augmentation - Time masking", xlabel="Time samples", ylabel="Frequency (Hz)")
ap.plot_spectrogram(freq_masked[0], axis=axs[3], title="Spectrogram Augmentation - Frequency masking", xlabel="Time samples", ylabel="Frequency (Hz)")
# ap.plot_spectrogram(mel_spec[0], axis=axs[4], title="Mel Spectrogram", xlabel="Time samples", ylabel="Frequency (Mel scale)")
# ap.plot_spectrogram(log_mel_spec[0], axis=axs[5], title="Logarithmic Mel Spectrogram", xlabel="Time samples", ylabel="Frequency (Mel scale)")



fig.tight_layout(pad=-1.5)
plt.subplot_tool()
# plt.subplots_adjust(hspace=1.3)
# plt.subplots_adjust(top=0.1, bottom=0.05)

plt.show()