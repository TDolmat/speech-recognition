import torch
import torch.nn as nn
from torch.nn import functional as F
import config as cg


class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super(TransposeLayer, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class SpeechRecognition(nn.Module):
    def __init__(self):
        super(SpeechRecognition, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(cg.N_MELS, cg.N_MELS, kernel_size=cg.KERNEL_SIZE, stride=cg.STRIDE, padding=cg.KERNEL_SIZE//cg.STRIDE),  # in_channels, out_channels, kernel_size, stride=1, padding=padding
            TransposeLayer(1, 2),
            nn.LayerNorm(cg.N_MELS),
            nn.GELU(),
            nn.Dropout(cg.DROPOUT),
        )
        self.dense = nn.Sequential(
            nn.Linear(cg.N_MELS, cg.MAIN_SIZE),    # in_features, out_features
            nn.LayerNorm(cg.MAIN_SIZE),
            nn.GELU(),
            nn.Dropout(cg.DROPOUT),
            nn.Linear(cg.MAIN_SIZE, cg.MAIN_SIZE),
            nn.LayerNorm(cg.MAIN_SIZE),
            nn.GELU(),
            nn.Dropout(cg.DROPOUT),
        )
        self.lstm = nn.LSTM(input_size=cg.MAIN_SIZE, hidden_size=cg.LSTM_HIDDEN_SIZE,
                            num_layers=cg.LSTM_NUMBER_OF_LAYERS, dropout=cg.LSTM_DROPOUT,
                            bidirectional=cg.LSTM_BIDIRECTIONAL)
        self.final_transformations = nn.Sequential(
            nn.LayerNorm(cg.LSTM_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(cg.DROPOUT),
        )

        self.final_fc = nn.Linear(cg.LSTM_HIDDEN_SIZE, cg.NUMBER_OF_CLASSES) # final fully connected
        self.log_softmax = nn.LogSoftmax(dim=2)    # https://www.baeldung.com/cs/softmax-vs-log-softmax


    def _init_hidden(self, batch_size):
        n, hs = cg.LSTM_NUMBER_OF_LAYERS, cg.LSTM_HIDDEN_SIZE
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))

    def forward(self, x, hidden):        
        if cg.LSTM_BIDIRECTIONAL:
            directions = 2
        else:
            directions = 1 
        
        # initial hidden state for each element in the input sequence
        h_0 = torch.zeros(directions * cg.LSTM_NUMBER_OF_LAYERS, cg.BATCH_SIZE, cg.LSTM_HIDDEN_SIZE)   
        #  initial cell state for each element in the input sequence 
        c_0 = torch.zeros(directions * cg.LSTM_NUMBER_OF_LAYERS, cg.BATCH_SIZE, cg.LSTM_HIDDEN_SIZE)   

        # TODO: Maybe, for h_0 and c_0 you can do .to(device)
        
        
        # x.shape - batch, number_of_channels_of_audio, feature, time 
        # feature is equal to n_feats which is actually n_mels
        # More about the shape in dataset.py:123
        
        x = x.squeeze(1)  # batch, feature, time - removing unnecessary dimention for num_of_channels
        x = self.cnn(x) # batch, time, feature
        x = self.dense(x) # batch, time, feature
        x = x.transpose(0, 1) # time, batch, feature
        x, (h_n, c_n) = self.lstm(x, hidden)

        x = self.final_transformations(x)  
        x = self.final_fc(x) # (time, batch, n_class)
        output = self.log_softmax(x)
        return output

