from os import listdir
from os.path import join, isdir
import numpy as np
from PIL import Image

from config import *


def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 256 X 256
    sequence_size: int
        The size of the lstm sequence
    Returns
    -------
    list
        A list of clips , 10 frames each
    """
    clips = []
    size = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    count = 0
    for start in range(stride):
        for i in range(start, size, stride):
            clip[count, :, :, 0] = frames_list[i]
            count += 1
            if count == sequence_size:
                clips.append(np.copy(clip))
                count = 0
    
    return clips


def get_training_set():
    """
    Returns
    -------
    list
            A list of training sequences of shape (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    clips = []
    # loop over the training folders (Train000,Train001,..)
    for folder in sorted(listdir(DATASET_PATH)):
        directory_path = join(DATASET_PATH, folder)
        if isdir(directory_path):
            all_frames = []
            # loop over all the images in the folder (0.tif, 1.tif, ..., 199.tif)
            for file in sorted(listdir(directory_path)):
                img_path = join(directory_path, file)
                if img_path.endswith('tif'):
                    img = Image.open(img_path).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            # get the 10-frame sequences from the list of images after applying data augmentation
            for stride in range(1,2):
                clips.extend(get_clips_by_stride(
                    stride=stride,
                    frames_list=all_frames,
                    sequence_size=10
                ))
    
    return clips

def get_single_test():
    size = 200
    test = np.zeros(shape=(size, 256, 256, 1))
    count = 0
    for file in sorted(listdir(SINGLE_TEST_PATH)):
        if file.endswith('tif'):
            img = Image.open(join(SINGLE_TEST_PATH, file)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[count, :, :, 0] = img
            count += 1

    return test