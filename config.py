DATASET_PATH ="./UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
TEST_PATH = "./UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
SINGLE_TEST_PATH = "./UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"

MODEL_PATH = "./model.hdf5"
TRAIN_MODEL = False

EPOCHS = 1

CLIP_LEN = 10
STRIDE = 1
DIM = (CLIP_LEN, 256, 256)
BATCH_SIZE = 2
N_CHANNELS = 1
SHUFFLE = True