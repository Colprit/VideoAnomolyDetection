DATASET_PATH ="./UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
TEST_PATH = "./UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
TEST = "002"
SINGLE_TEST_PATH = f"./UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test{TEST}"

VERSION = "001"
MODEL_DIR = f"./models/{VERSION}"
MODEL_PATH = f"{MODEL_DIR}/model.hdf5"
FIG_PATH = f'{MODEL_DIR}/test_figures'
TRAIN_MODEL = False

EPOCHS = 1

CLIP_LEN = 10
STRIDE = 1
DIM = (CLIP_LEN, 256, 256)
BATCH_SIZE = 4
N_CHANNELS = 1
SHUFFLE = True