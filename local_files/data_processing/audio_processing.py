import torchaudio
import numpy as np
import torch.nn as nn
import torch

class LogMelSpec(nn.Module):

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(LogMelSpec, self).__init__()
        self.mel_spectogram_function = torchaudio.transforms.MelSpectrogram(
                            sample_rate=sample_rate, n_fft=n_fft,
                            win_length=win_length, hop_length=hop_length, 
                            n_mels=n_mels
                            )

    def forward(self, x):
        x = self.mel_spectogram_function(x)  # mel spectrogram
        x = np.log(x + 1e-14)  # logrithmic, add small value to avoid inf
        return x
    
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