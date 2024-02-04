MAX_EPOCHS = 400
ON_MAC = False


# PATHS
TRAIN_PATH = ""
VALIDATE_PATH = ""
SAVE_MODEL_PATH = ""

# DATA
BATCH_SIZE = 64
VALID_EVERY = 1000 // BATCH_SIZE

# DATALOADER
NUM_WORKERS = 1

# AUDIO
SAMPLE_RATE = 32000

# MEL LOG SPECTROGRAM
N_MELS = 128
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 512
MAX_SPECTROGRAM_SIZE = 1650

# SPECTROGRAM AUGMENTATION
SPECAUG_RATE = 0.5
SPECAUG_POLICY = 3
TIME_MASK = 60
FREQUENCY_MASK = 20

# TEXT
NUMBER_OF_CLASSES = 29 # number of label clases (characters)
BLANK_CHARACTER_INDEX = 28

# MODEL
DROPOUT = 0.1
MAIN_SIZE = 128

# CNN
KERNEL_SIZE = 10
STRIDE = 2

# LSTM
LSTM_HIDDEN_SIZE = 512
LSTM_NUMBER_OF_LAYERS = 1
LSTM_DROPOUT = 0.0
LSTM_BIDIRECTIONAL = False

DEVICE = "mps" # Temportary, should be in model:    device = 'cuda' if torch.cuda.is_available() else 'cpu'