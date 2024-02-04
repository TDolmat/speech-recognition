import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
import config as cg
from torch.utils.data import DataLoader

from visualisation.audio_plots import AudioPlots

from data_processing.text_processing import TextProcessing
from data_processing.audio_processing import LogMelSpec, SpecAugment

# TODO: Add to thesis https://www.youtube.com/watch?v=Z7YM-HAz-IY


class Data(torch.utils.data.Dataset):
    def __init__(self, csv_path, log_exception=True):
        print(f"\nLoading data CSV file from: {csv_path}\n")
        self.data = pd.read_csv(csv_path, sep='\t')

        self.log_exception = log_exception

        self.audio_transforms = torch.nn.Sequential(
            LogMelSpec(sample_rate=cg.SAMPLE_RATE,
                       n_fft=cg.N_FFT, 
                       n_mels=cg.N_MELS,
                       win_length=cg.WIN_LENGTH, 
                       hop_length=cg.HOP_LENGTH),
            SpecAugment(rate=cg.SPECAUG_RATE, 
                        policy=cg.SPECAUG_POLICY, 
                        freq_mask=cg.FREQUENCY_MASK, 
                        time_mask=cg.TIME_MASK)
        )

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, index):

        file_path = None
        try:
            text = self.data.iloc[index].text # Column is named "text"
            label = TextProcessing.text_to_int_sequence(text) # Text as sequence of ints
            label_len = len(label)

            file_path = self.data.iloc[index].file # Column is named "file"
            waveform, samplerate = torchaudio.load(file_path)

            spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2

            # FOR TESTING PURPOSES
            # print("\n\n")
            # print(text)
            # print(label)
            # ap = AudioPlots(sample_rate)
            # ap.plot_audio(waveform_audio=waveform, spectrogram_audio=spectrogram)

            if spec_len < label_len:
                raise Exception(f'spectrogram len ({spec_len}) is smaller than label len ({label_len})') # spectrogram length must be higher than label length so that audio is longer than written form
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s'%file_path)
            if spectrogram.shape[2] > cg.MAX_SPECTROGRAM_SIZE:
                raise Exception('spectrogram to big. size %s'%spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s'%file_path)

        except Exception as e:
            if self.log_exception:
                print(str(e), file_path, text)
                return self.__getitem__(index - 1 if index != 0 else index + 1) # Returning previous item (we have to assume that at least first element was correct, if not we will have a loop)
        return spectrogram, label, spec_len, label_len


def collate_fn_padd(batch):
    spectrograms = []
    labels = []
    spectrogram_lengths = []
    label_lengths = []
    for (spectrogram, label, spectrogram_length, label_length) in batch:
        if spectrogram is None:
            continue


        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1)) 
        # Squeeze gets rid of first size parameter beacuse spectrograms in this program are single channel 
        # FROM: torch.Size([1, 128, 514])    ->    TO: torch.Size([128, 514])
        # 
        # Transposing so that the first parameter will be the number of elements (in our example 514), 
        # because 128 is number of mels and we are doing this so we can use pad_sequence 

        labels.append(torch.Tensor(label))
        spectrogram_lengths.append(spectrogram_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    # pad_sequence adds zeros  for elements which are smaller than the biggest one, then we unsqueze it 
    # to get back to this 1 in front of 128, 514 and transpose it to get back from 
    # [514, 128] to original [128, 514], and we end up with shape: torch.Size([2, 1, 128, 514])
    # Where: 
    #   2 is number of elements in a batch 
    #   1 is a number of channels of audio
    #   128 is number of mels
    #   514 is a number of time sequences of this audio (max)
    
    # because batch_first is true number of batches is a first parameter: [2, 1, 128, 514]
    # otherwise it will be like this: [514, 1, 128, 2]



    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    spectrogram_lengths = spectrogram_lengths

    label_lengths = label_lengths
    # ## compute mask
    # mask = (batch != 0).cuda(gpu)
    # return batch, lengths, mask
    
    # print(f"\n\n\nDATALOADER: 0 - {spectrograms.shape}\n1 - {labels.shape}\n2 - {spectrogram_lengths}\n3 - {label_lengths}\n\n\n\n")

    return spectrograms, labels, spectrogram_lengths, label_lengths


def testing():
    # TESTING
    d = Data("/Users/tomasz/repos/thesis/dataset/test/datasets-csv/train.csv")

    # # ===== HOW DATA WORKS =====
    # # first_element = d[0]
    # # print(f"""Dataset (Data):
    # # Spectrogram (shape): {first_element[0].shape} [channels, number of mels - x axis, number of time sequences - y axis]
    # # Label: {first_element[1]}
    # # Spec len: {first_element[2]}
    # # Label len: {first_element[3]}
    # # """)


    dl = DataLoader(dataset=d,
                batch_size=10,
                collate_fn=collate_fn_padd)

    # # ===== HOW GETTING FROM BATCH WORKS =====
    # i = 0
    for batch in dl:
    #     print(i)
    #     # if i == 0:
        if True:
            spectrograms, labels, spectrogram_lengths, label_lengths = batch
    #         # To get element from batch
    #         batch_element_number = 0

            print(spectrograms.shape)

    #         # print(spectrograms[batch_element_number].shape)
    #         # print(labels[batch_element_number].shape)
    #         # print(spectrogram_lengths[batch_element_number])
    #         # print(label_lengths[batch_element_number])
    #         print("\n")
        
    #     i += 1
    
if __name__ == '__main__':
    testing()

